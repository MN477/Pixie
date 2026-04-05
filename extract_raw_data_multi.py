"""
extract_raw_data_multi.py
=========================
Multi-person landmark extraction — optimized for low latency.

Key optimizations vs. original:
  1. Merged detection + pose: YOLOv8-pose.track() does person detection,
     ByteTrack tracking, AND 17-keypoint extraction in ONE model call.
     Eliminates YOLO11m entirely.
  2. Frame stride: process every Nth frame (configurable).
  3. Inference resolution cap: YOLO infers at fixed lower res (e.g. 480px).

Pipeline per frame:
  1. YOLOv8-pose.track(frame) → bboxes + track IDs + 17 keypoints
  2. Write body keypoints to CSV (full-frame coords, no remapping)
  3. For each person: extract body crop from bbox
  4. YOLO-face on body crop → face bbox (conf > 0.5)
  5. Expand face 25%, save crop, queue for OpenFace
  6. 6DRepNet on face crop → pitch, yaw, roll → CSV
  7. OpenFace processes face crops in background thread

Outputs:
  raw_body_multi.csv, raw_head_pose_multi.csv,
  raw_action_units_multi.csv, raw_gaze_multi.csv, face_crops/

Usage:
    python extract_raw_data_multi.py
"""

import csv
import gc
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sixdrepnet import SixDRepNet

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
INPUT_SOURCE = "testing_vid/on_vid yawing 1.mp4"

BODY_OUTPUT      = "raw_body_multi.csv"
HEAD_POSE_OUTPUT = "raw_head_pose_multi.csv"
AU_OUTPUT        = "raw_action_units_multi.csv"
GAZE_OUTPUT      = "raw_gaze_multi.csv"
FACE_CROPS_DIR   = "face_crops"
OPENFACE_OUT_DIR = "openface_output"

# Models (YOLO11m removed — YOLOv8-pose handles detection + tracking + pose)
POSE_MODEL_PATH      = "yolov8n-pose.pt"
FACE_YOLO_MODEL_PATH = "yolov11m-face.pt"

# OpenFace
OPENFACE_DIR = r"C:\Users\mouss\Documents\OpenFace_2.2.0_win_x86"
OPENFACE_EXE = os.path.join(OPENFACE_DIR, "FaceLandmarkImg.exe")

# ── Performance tuning ──
FRAME_STRIDE        = 1     # process every Nth frame (1 = all, 2 = half, etc.)
INFERENCE_SIZE      = 480   # YOLO inference resolution (lower = faster)
EXPAND_RATIO        = 0.20  # body bbox expansion for face detection crop
OPENFACE_BATCH_SIZE = 300  # OpenFace batch trigger size

# COCO 17 keypoint names
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# AU columns output by OpenFace
AU_INTENSITY_COLS = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
    "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
    "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
]
AU_BINARY_COLS = [
    "AU01_c", "AU02_c", "AU04_c", "AU05_c", "AU06_c", "AU07_c",
    "AU09_c", "AU10_c", "AU12_c", "AU14_c", "AU15_c", "AU17_c",
    "AU20_c", "AU23_c", "AU25_c", "AU26_c", "AU28_c", "AU45_c",
]

# Gaze columns output by OpenFace
GAZE_COLS = [
    "gaze_0_x", "gaze_0_y", "gaze_0_z",
    "gaze_1_x", "gaze_1_y", "gaze_1_z",
    "gaze_angle_x", "gaze_angle_y"
]

FILENAME_PATTERN = re.compile(r"frame_(\d+)_track_(\d+)")


# ──────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────
def get_device():
    """Auto-detect GPU, fall back to CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def expand_bbox(x1, y1, x2, y2, frame_h, frame_w, expand=EXPAND_RATIO):
    """Expand a bounding box by `expand` ratio and clip to frame bounds."""
    w = x2 - x1
    h = y2 - y1
    pad_w = w * expand / 2
    pad_h = h * expand / 2

    x1 = max(0, int(x1 - pad_w))
    y1 = max(0, int(y1 - pad_h))
    x2 = min(frame_w, int(x2 + pad_w))
    y2 = min(frame_h, int(y2 + pad_h))

    return x1, y1, x2, y2


# ──────────────────────────────────────────────
# OPENFACE BACKGROUND WORKER
# ──────────────────────────────────────────────
class OpenFaceWorker:
    """Background thread that processes batches of face crops through OpenFace."""

    def __init__(self, face_crops_dir, openface_out_dir, batch_size=OPENFACE_BATCH_SIZE):
        self.face_crops_dir = face_crops_dir
        self.openface_out_dir = openface_out_dir
        self.batch_size = batch_size
        self.pending_crops = []
        self.batch_count = 0
        self.lock = threading.Lock()
        self.task_queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.total_processed = 0

    def start(self):
        os.makedirs(self.openface_out_dir, exist_ok=True)
        self.thread.start()
        print("[OpenFace] Background worker started")

    def add_crop(self, filename):
        with self.lock:
            self.pending_crops.append(filename)
            if len(self.pending_crops) >= self.batch_size:
                batch = self.pending_crops.copy()
                self.pending_crops.clear()
                self.batch_count += 1
                batch_id = self.batch_count
                self.task_queue.put(("batch", batch_id, batch))

    def flush_and_stop(self):
        with self.lock:
            if self.pending_crops:
                self.batch_count += 1
                batch_id = self.batch_count
                batch = self.pending_crops.copy()
                self.pending_crops.clear()
                self.task_queue.put(("batch", batch_id, batch))
        self.task_queue.put(("stop", None, None))
        self.thread.join(timeout=600)
        print(f"[OpenFace] Worker stopped. Total crops processed: {self.total_processed}")

    def _worker_loop(self):
        while True:
            action, batch_id, data = self.task_queue.get()
            if action == "stop":
                break
            elif action == "batch":
                self._process_batch(batch_id, data)

    def _process_batch(self, batch_id, filenames):
        batch_dir = os.path.join(self.face_crops_dir, f"_batch_{batch_id}")
        os.makedirs(batch_dir, exist_ok=True)

        for fname in filenames:
            src = os.path.join(self.face_crops_dir, fname)
            dst = os.path.join(batch_dir, fname)
            if os.path.exists(src):
                os.rename(src, dst)

        batch_out_dir = os.path.join(self.openface_out_dir, f"batch_{batch_id}")
        os.makedirs(batch_out_dir, exist_ok=True)

        cmd = [
            OPENFACE_EXE,
            "-fdir", os.path.abspath(batch_dir),
            "-out_dir", os.path.abspath(batch_out_dir),
            "-aus",
            "-gaze",
            "-multi_view", "1",
        ]

        print(f"[OpenFace] Processing batch {batch_id} ({len(filenames)} crops)...")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=OPENFACE_DIR, timeout=300
            )
            if result.returncode != 0:
                print(f"[OpenFace] Batch {batch_id} warning: exit code {result.returncode}")
            else:
                print(f"[OpenFace] Batch {batch_id} complete")
        except subprocess.TimeoutExpired:
            print(f"[OpenFace] Batch {batch_id} timed out!")
        except Exception as e:
            print(f"[OpenFace] Batch {batch_id} error: {e}")

        self.total_processed += len(filenames)

        for fname in filenames:
            src = os.path.join(batch_dir, fname)
            dst = os.path.join(self.face_crops_dir, fname)
            if os.path.exists(src):
                os.rename(src, dst)

        try:
            os.rmdir(batch_dir)
        except OSError:
            pass


#------------------------------
# OPENFACE CSV MERGING
#------------------------------
def merge_openface_outputs(openface_out_dir, au_csv, gaze_csv):
    """Parse all OpenFace batch output CSVs and merge into two clean CSVs."""
    au_rows = []
    gaze_rows = []

    for root, dirs, files in os.walk(openface_out_dir):
        for csv_file in files:
            if not csv_file.endswith(".csv"):
                continue

            csv_path = os.path.join(root, csv_file)
            basename = os.path.splitext(csv_file)[0]
            match = FILENAME_PATTERN.search(basename)
            if not match:
                continue

            frame_id = int(match.group(1))
            track_id = int(match.group(2))

            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cleaned = {k.strip(): v.strip() for k, v in row.items()}
                    confidence = float(cleaned.get("confidence", 0))
                    success_val = cleaned.get("success")
                    success = int(success_val) if success_val is not None else 1

                    common_dict = {
                        "frame_id": frame_id,
                        "track_id": track_id,
                        "confidence": f"{confidence:.4f}",
                        "success": success,
                    }

                    out_au = dict(common_dict)
                    out_gaze = dict(common_dict)

                    if success:
                        for au_col in AU_INTENSITY_COLS + AU_BINARY_COLS:
                            out_au[au_col] = cleaned.get(au_col, "")
                        for gz_col in GAZE_COLS:
                            out_gaze[gz_col] = cleaned.get(gz_col, "")
                    else:
                        for au_col in AU_INTENSITY_COLS + AU_BINARY_COLS:
                            out_au[au_col] = ""
                        for gz_col in GAZE_COLS:
                            out_gaze[gz_col] = ""

                    au_rows.append(out_au)
                    gaze_rows.append(out_gaze)

    au_rows.sort(key=lambda r: (r["frame_id"], r["track_id"]))
    gaze_rows.sort(key=lambda r: (r["frame_id"], r["track_id"]))

    au_fieldnames = ["frame_id", "track_id", "confidence", "success"] + AU_INTENSITY_COLS + AU_BINARY_COLS
    gaze_fieldnames = ["frame_id", "track_id", "confidence", "success"] + GAZE_COLS

    with open(au_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=au_fieldnames)
        writer.writeheader()
        writer.writerows(au_rows)

    with open(gaze_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=gaze_fieldnames)
        writer.writeheader()
        writer.writerows(gaze_rows)

    print(f"[OpenFace] Merged {len(au_rows)} AU and Gaze rows")
    print(f"  → {au_csv}")
    print(f"  → {gaze_csv}")


# ──────────────────────────────────────────────
# MODEL INITIALISATION
# ──────────────────────────────────────────────
device = get_device()
print(f"[INFO] Using device: {device}")

print("[INFO] Loading YOLOv8-pose (detection + tracking + pose)...")
pose_model = YOLO(POSE_MODEL_PATH)
pose_model.to(device)

print("[INFO] Loading YOLO-face model...")
face_det_model = YOLO(FACE_YOLO_MODEL_PATH)
face_det_model.to(device)

print("[INFO] Initializing 6DRepNet...")
gpu_id = 0 if torch.cuda.is_available() else -1
sixd_model = SixDRepNet(gpu_id=gpu_id)

# ──────────────────────────────────────────────
# PREPARE DIRECTORIES
# ──────────────────────────────────────────────
print("[INFO] Cleaning up previous run data...")
if os.path.isdir(FACE_CROPS_DIR):
    shutil.rmtree(FACE_CROPS_DIR, ignore_errors=True)
if os.path.isdir(OPENFACE_OUT_DIR):
    shutil.rmtree(OPENFACE_OUT_DIR, ignore_errors=True)

os.makedirs(FACE_CROPS_DIR, exist_ok=True)
os.makedirs(OPENFACE_OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# OPEN CSV FILES & WRITE HEADERS
# ──────────────────────────────────────────────
body_csv_file      = open(BODY_OUTPUT, "w", newline="", encoding="utf-8")
head_pose_csv_file = open(HEAD_POSE_OUTPUT, "w", newline="", encoding="utf-8")

body_writer      = csv.writer(body_csv_file)
head_pose_writer = csv.writer(head_pose_csv_file)

body_writer.writerow([
    "frame_id", "track_id", "landmark_idx", "x", "y", "visibility",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"
])
head_pose_writer.writerow([
    "frame_id", "track_id", "pitch", "yaw", "roll"
])

# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────
def main():
    start_time = time.time()

    # Verify OpenFace
    if not os.path.isfile(OPENFACE_EXE):
        print(f"[WARN] OpenFace not found at {OPENFACE_EXE} — AU extraction will be skipped")
        openface_available = False
    else:
        openface_available = True

    if not os.path.isfile(INPUT_SOURCE):
        print(f"[ERROR] Video file not found: {INPUT_SOURCE}")
        sys.exit(1)

    cap = cv2.VideoCapture(INPUT_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {INPUT_SOURCE}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"[INFO] Video: {frame_w}x{frame_h}, {total_frames} frames.")
    print(f"[INFO] Frame stride: {FRAME_STRIDE} | Inference size: {INFERENCE_SIZE}px")

    # Start OpenFace background worker
    of_worker = None
    if openface_available:
        of_worker = OpenFaceWorker(FACE_CROPS_DIR, OPENFACE_OUT_DIR)
        of_worker.start()

    frame_id = 0
    processed_frames = 0
    print("[INFO] Starting extraction loop...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n[INFO] End of video stream.")
                break

            print(f"Processing video frame {frame_id}/{total_frames}...", end='\r', flush=True)

            # ── Frame stride: skip frames ──
            if frame_id % FRAME_STRIDE != 0:
                frame_id += 1
                continue

            # ── Step 1: YOLOv8-pose + ByteTrack (detection + tracking + keypoints in ONE call) ──
            results = pose_model.track(
                source=frame,
                tracker="bytetrack.yaml",
                persist=True,
                conf=0.25,
                iou=0.5,
                classes=[0],
                imgsz=INFERENCE_SIZE,
                stream=False,
                verbose=False,
            )

            boxes = results[0].boxes
            keypoints_data = results[0].keypoints

            if boxes is None or len(boxes) == 0:
                frame_id += 1
                continue

            if boxes.id is None:
                frame_id += 1
                continue

            xyxy_list = boxes.xyxy.cpu().numpy().astype(int)
            track_ids = boxes.id.cpu().numpy().astype(int)

            # Get all keypoints (shape: [N, 17, 3])
            all_kpts = None
            if (keypoints_data is not None
                    and keypoints_data.data is not None
                    and len(keypoints_data.data) > 0):
                all_kpts = keypoints_data.data.cpu().numpy()

            # Debug visualisation
            debug_frame = frame.copy()
            cv2.putText(
                debug_frame, f"Frame: {frame_id}/{total_frames}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, (0, 0, 255), 2
            )
            for bbox, track_id in zip(xyxy_list, track_ids):
                x1, y1, x2, y2 = bbox
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    debug_frame, f"ID:{track_id}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2,
                )
            cv2.imshow("Tracking Debug", debug_frame)
            cv2.waitKey(1)

            # ── Process each detected person ──
            for person_idx, (bbox, track_id) in enumerate(zip(xyxy_list, track_ids)):
                x1, y1, x2, y2 = bbox
                tid = int(track_id)

                # ── Step 2: Write body keypoints (already in full-frame coords) ──
                if all_kpts is not None and person_idx < len(all_kpts):
                    kpts = all_kpts[person_idx]  # shape: [17, 3]
                    for lm_idx in range(kpts.shape[0]):
                        body_writer.writerow([
                            frame_id, tid, lm_idx,
                            f"{kpts[lm_idx, 0]:.4f}",
                            f"{kpts[lm_idx, 1]:.4f}",
                            f"{kpts[lm_idx, 2]:.4f}",
                            x1, y1, x2, y2
                        ])
                else:
                    body_writer.writerow([frame_id, tid, None, None, None, None, x1, y1, x2, y2])

                # ── Step 3: Extract body crop for face detection ──
                cx1, cy1, cx2, cy2 = expand_bbox(
                    x1, y1, x2, y2, frame_h, frame_w
                )
                crop_w = cx2 - cx1
                crop_h = cy2 - cy1
                if crop_w < 10 or crop_h < 10:
                    head_pose_writer.writerow([frame_id, tid, None, None, None])
                    continue

                crop_bgr = frame[cy1:cy2, cx1:cx2]

                # ── Step 4: YOLO-face on body crop ──
                try:
                    face_results = face_det_model(crop_bgr, verbose=False)
                    face_boxes = face_results[0].boxes if len(face_results) > 0 else None

                    best_face = None
                    if face_boxes is not None and len(face_boxes) > 0:
                        confidences = face_boxes.conf.cpu().numpy()
                        max_conf_idx = np.argmax(confidences)
                        if confidences[max_conf_idx] > 0.5:
                            best_face = face_boxes.xyxy.cpu().numpy()[max_conf_idx].astype(int)

                    if best_face is None:
                        head_pose_writer.writerow([frame_id, tid, None, None, None])
                        continue

                    fx1, fy1, fx2, fy2 = best_face

                    # ── Step 5: Expand face 25%, save crop ──
                    fw = fx2 - fx1
                    fh = fy2 - fy1
                    pad_w = fw * 0.25
                    pad_h = fh * 0.25

                    cf1 = max(0, int(fx1 - pad_w))
                    cf2 = max(0, int(fy1 - pad_h))
                    cf3 = min(crop_w, int(fx2 + pad_w))
                    cf4 = min(crop_h, int(fy2 + pad_h))

                    face_crop_bgr = crop_bgr[cf2:cf4, cf1:cf3]

                    if face_crop_bgr.shape[0] < 10 or face_crop_bgr.shape[1] < 10:
                        head_pose_writer.writerow([frame_id, tid, None, None, None])
                        continue

                    # Save face crop for OpenFace
                    face_crop_filename = f"frame_{frame_id:06d}_track_{tid}.jpg"
                    face_crop_path = os.path.join(FACE_CROPS_DIR, face_crop_filename)
                    cv2.imwrite(face_crop_path, face_crop_bgr)

                    # Queue for OpenFace background processing
                    if of_worker:
                        of_worker.add_crop(face_crop_filename)

                    # ── Step 6: 6DRepNet → pitch, yaw, roll ──
                    pitch, yaw, roll = sixd_model.predict(face_crop_bgr)
                    p_val = float(np.ravel(pitch)[0])
                    y_val = float(np.ravel(yaw)[0])
                    r_val = float(np.ravel(roll)[0])

                    head_pose_writer.writerow([
                        frame_id, tid,
                        f"{p_val:.4f}", f"{y_val:.4f}", f"{r_val:.4f}"
                    ])

                except Exception as e:
                    print(f"[WARN] Face/HeadPose failed | frame {frame_id}, track {tid}: {e}")
                    head_pose_writer.writerow([frame_id, tid, None, None, None])

            # ── Housekeeping ──
            processed_frames += 1
            del frame
            gc.collect()

            if processed_frames % 100 == 0 and processed_frames > 0:
                body_csv_file.flush()
                head_pose_csv_file.flush()
                print(
                    f"\n[INFO] Processed {processed_frames} frames "
                    f"(frame_id {frame_id}/{total_frames})..."
                )

            frame_id += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C).")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        body_csv_file.close()
        head_pose_csv_file.close()

        print(f"[INFO] Extraction done. {frame_id} total frames, {processed_frames} processed.")
        print(f"  → {BODY_OUTPUT}")
        print(f"  → {HEAD_POSE_OUTPUT}")

        # ── Wait for OpenFace to finish remaining batches ──
        if of_worker:
            print("[INFO] Waiting for OpenFace background worker to finish...")
            of_worker.flush_and_stop()

            # ── Merge all OpenFace outputs into final CSVs ──
            print("[INFO] Merging OpenFace outputs...")
            merge_openface_outputs(OPENFACE_OUT_DIR, AU_OUTPUT, GAZE_OUTPUT)

        print("[INFO] All done!")

        elapsed = time.time() - start_time
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"[INFO] Total execution time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")


if __name__ == "__main__":
    main()
