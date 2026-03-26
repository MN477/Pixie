"""
extract_raw_data_multi.py
=========================
Multi-person landmark extraction using:
  - YOLO11m + ByteTrack  → person detection & tracking
  - YOLOv8n-pose         → 17 COCO body keypoints per person
  - YOLOv11m-face        → face bounding box detection
  - 6DRepNet             → head pose (pitch, yaw, roll)
  - OpenFace 2.2.0       → Action Units (runs in background thread)

Pipeline per frame:
  1. YOLO detect + ByteTrack → bounding boxes with persistent track IDs
  2. For each person: expand bbox 20%, make square, crop
  3. YOLO-pose on body crop  → 17 keypoints → raw_body_multi.csv
  4. YOLO-face on body crop  → face bbox (conf > 0.5)
  5. Expand face bbox 25%, extract face sub-image → save to face_crops/
  6. 6DRepNet on face crop   → pitch, yaw, roll → raw_head_pose_multi.csv
  7. OpenFace processes batches of face crops in a background thread
  8. Always write one row per person per frame (None if failed)

Outputs:
  raw_body_multi.csv, raw_head_pose_multi.csv,
  raw_action_units_multi.csv, face_crops/

Usage:
    python extract_raw_data_multi.py
"""

import csv
import gc
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import queue

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sixdrepnet import SixDRepNet

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
INPUT_SOURCE = "testing_vid/HandRaise_019.mp4"

BODY_OUTPUT      = "raw_body_multi.csv"
HEAD_POSE_OUTPUT = "raw_head_pose_multi.csv"
AU_OUTPUT        = "raw_action_units_multi.csv"
GAZE_OUTPUT      = "raw_gaze_multi.csv"
FACE_CROPS_DIR   = "face_crops"
OPENFACE_OUT_DIR = "openface_output"

# YOLO models
YOLO_MODEL_PATH      = "yolo11m.pt"
POSE_MODEL_PATH      = "yolov8n-pose.pt"
FACE_YOLO_MODEL_PATH = "yolov11m-face.pt"

# OpenFace
OPENFACE_DIR = r"C:\Users\mouss\Documents\OpenFace_2.2.0_win_x86"
OPENFACE_EXE = os.path.join(OPENFACE_DIR, "FaceLandmarkImg.exe")

# Processing settings
EXPAND_RATIO = 0.20
OPENFACE_BATCH_SIZE = 300  # process OpenFace every N face crops

# COCO 17 keypoint names (for reference)
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
def expand_and_square_bbox(x1, y1, x2, y2, frame_h, frame_w, expand=EXPAND_RATIO):
    """Expand a bounding box by `expand` ratio, make it square, and clip to frame."""
    w = x2 - x1
    h = y2 - y1

    pad_w = w * expand / 2
    pad_h = h * expand / 2
    x1 -= pad_w
    y1 -= pad_h
    x2 += pad_w
    y2 += pad_h

    new_w = x2 - x1
    new_h = y2 - y1
    side = max(new_w, new_h)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    x1 = cx - side / 2
    y1 = cy - side / 2
    x2 = cx + side / 2
    y2 = cy + side / 2

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(frame_w, int(x2))
    y2 = min(frame_h, int(y2))

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
        self.pending_crops = []       # list of filenames waiting
        self.batch_count = 0
        self.lock = threading.Lock()
        self.task_queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.total_processed = 0

    def start(self):
        """Start the background worker thread."""
        os.makedirs(self.openface_out_dir, exist_ok=True)
        self.thread.start()
        print("[OpenFace] Background worker started")

    def add_crop(self, filename):
        """Add a face crop filename to the pending list. Triggers batch if full."""
        with self.lock:
            self.pending_crops.append(filename)
            if len(self.pending_crops) >= self.batch_size:
                batch = self.pending_crops.copy()
                self.pending_crops.clear()
                self.batch_count += 1
                batch_id = self.batch_count
                self.task_queue.put(("batch", batch_id, batch))

    def flush_and_stop(self):
        """Process any remaining crops and shut down the worker."""
        with self.lock:
            if self.pending_crops:
                self.batch_count += 1
                batch_id = self.batch_count
                batch = self.pending_crops.copy()
                self.pending_crops.clear()
                self.task_queue.put(("batch", batch_id, batch))

        # Signal the worker to stop
        self.task_queue.put(("stop", None, None))
        self.thread.join(timeout=600)  # wait up to 10 min for last batch
        print(f"[OpenFace] Worker stopped. Total crops processed: {self.total_processed}")

    def _worker_loop(self):
        """Main loop of the background worker thread."""
        while True:
            action, batch_id, data = self.task_queue.get()
            if action == "stop":
                break
            elif action == "batch":
                self._process_batch(batch_id, data)

    def _process_batch(self, batch_id, filenames):
        """Process a batch: move crops to a temp subfolder, run OpenFace, move back."""
        batch_dir = os.path.join(self.face_crops_dir, f"_batch_{batch_id}")
        os.makedirs(batch_dir, exist_ok=True)

        # Move crop files into the batch subfolder
        for fname in filenames:
            src = os.path.join(self.face_crops_dir, fname)
            dst = os.path.join(batch_dir, fname)
            if os.path.exists(src):
                os.rename(src, dst)

        # Run OpenFace on this batch folder
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

        # Move crops back to main folder (so they're still available)
        for fname in filenames:
            src = os.path.join(batch_dir, fname)
            dst = os.path.join(self.face_crops_dir, fname)
            if os.path.exists(src):
                os.rename(src, dst)

        # Clean up empty batch dir
        try:
            os.rmdir(batch_dir)
        except OSError:
            pass


# ──────────────────────────────────────────────
# OPENFACE CSV MERGING
# ──────────────────────────────────────────────
def merge_openface_outputs(openface_out_dir, au_csv, gaze_csv):
    """Parse all OpenFace batch output CSVs and merge into two clean CSVs."""
    au_rows = []
    gaze_rows = []

    # Walk through all batch subdirectories
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
                    
                    # FaceLandmarkImg.exe single-image mode sometimes omits 'success'
                    # Default to 1 (success) because YOLO already guaranteed a face crop.
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
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

print("[INFO] Loading YOLO person detector...")
yolo_model = YOLO(YOLO_MODEL_PATH)
yolo_model.to(device)

print("[INFO] Loading YOLO-pose model...")
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
    "frame_id", "track_id", "landmark_idx", "x", "y", "visibility"
])
head_pose_writer.writerow([
    "frame_id", "track_id", "pitch", "yaw", "roll"
])

# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────
def main():
    start_time = time.time()

    # Verify OpenFace is available
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

    # Start OpenFace background worker
    of_worker = None
    if openface_available:
        of_worker = OpenFaceWorker(FACE_CROPS_DIR, OPENFACE_OUT_DIR)
        of_worker.start()

    frame_id = 0
    saved_frames = 0
    print("[INFO] Starting extraction loop...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video stream.")
                break

            # ── Step 1: YOLO person detection + ByteTrack ──
            results = yolo_model.track(
                source=frame,
                tracker="bytetrack.yaml",
                persist=True,
                conf=0.25,
                iou=0.5,
                classes=[0],
                stream=False,
                verbose=False,
            )

            boxes = results[0].boxes

            if boxes is None or len(boxes) == 0:
                print(f"[DEBUG] No detections at frame {frame_id}")
                frame_id += 1
                continue

            print(f"[DEBUG] Frame {frame_id} | Detections: {len(boxes)}")

            if boxes.id is None:
                print(f"[ERROR] No tracking IDs at frame {frame_id} — tracking NOT working")
                frame_id += 1
                continue

            xyxy_list = boxes.xyxy.cpu().numpy().astype(int)
            track_ids = boxes.id.cpu().numpy().astype(int)
            print(f"[DEBUG] Track IDs: {track_ids}")

            # Debug visualisation
            debug_frame = frame.copy()
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

            frame_has_detections = False
            processed_count = 0

            print(f"[DEBUG] Processing {len(xyxy_list)} detections for frame {frame_id}")

            for bbox, track_id in zip(xyxy_list, track_ids):
                x1, y1, x2, y2 = bbox
                tid = int(track_id)
                print(f"[DEBUG] Processing track ID {tid} with bbox {bbox}")

                # Step 2: Expand, square, clip
                cx1, cy1, cx2, cy2 = expand_and_square_bbox(
                    x1, y1, x2, y2, frame_h, frame_w
                )
                crop_w = cx2 - cx1
                crop_h = cy2 - cy1
                if crop_w < 10 or crop_h < 10:
                    print(f"[DEBUG] Skipping track ID {tid} - crop too small")
                    body_writer.writerow([frame_id, tid, None, None, None, None])
                    head_pose_writer.writerow([frame_id, tid, None, None, None])
                    continue

                crop_bgr = frame[cy1:cy2, cx1:cx2]

                # ── Step 3: YOLO-pose → 17 keypoints ──
                try:
                    pose_results = pose_model(crop_bgr, verbose=False)
                    if (pose_results and len(pose_results) > 0
                            and pose_results[0].keypoints is not None
                            and pose_results[0].keypoints.data is not None
                            and len(pose_results[0].keypoints.data) > 0):

                        frame_has_detections = True
                        processed_count += 1
                        kpts = pose_results[0].keypoints.data[0].cpu().numpy()

                        for lm_idx in range(kpts.shape[0]):
                            global_x = cx1 + kpts[lm_idx, 0]
                            global_y = cy1 + kpts[lm_idx, 1]
                            vis = kpts[lm_idx, 2]
                            body_writer.writerow([
                                frame_id, tid, lm_idx,
                                f"{global_x:.4f}", f"{global_y:.4f}", f"{vis:.4f}",
                            ])
                    else:
                        body_writer.writerow([frame_id, tid, None, None, None, None])
                except Exception as e:
                    print(f"[WARN] Pose failed | frame {frame_id}, track {tid}: {e}")
                    body_writer.writerow([frame_id, tid, None, None, None, None])

                # ── Step 4: YOLO-face → face bbox ──
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
            if frame_has_detections:
                saved_frames += 1

            del frame
            gc.collect()

            if frame_id % 100 == 0 and frame_id > 0:
                body_csv_file.flush()
                head_pose_csv_file.flush()
                print(
                    f"[INFO] Processed {frame_id}/{total_frames} frames "
                    f"({saved_frames} with detections)..."
                )

            frame_id += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C).")

    finally:
        cap.release()

        body_csv_file.close()
        head_pose_csv_file.close()

        print(f"[INFO] Extraction done. {frame_id} total frames, {saved_frames} with detections.")
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
