"""
extract_raw_data_multi.py
=========================
Multi-person landmark extraction — optimized for low latency.

FIX: ID-swap during occlusion
──────────────────────────────
Problem : ByteTrack loses a track during overlap and re-assigns a new ID
          (e.g. person 1 becomes person 4 after occlusion).

Solution (two layers):
  1. Switch tracker from bytetrack.yaml → botsort.yaml.
     BoT-SORT adds an appearance (ReID) embedding so the same person
     is recognized visually after occlusion, not just by position.

  2. IDStabilizer post-hoc guard.
     After every track() call, IDStabilizer checks every "new" ID
     (one that appeared this frame for the first time) against the
     last known bboxes of recently-lost tracks.  If the IoU + centroid
     distance to a lost track is above threshold, the new ID is
     remapped to the original ID.  This catches the cases where
     BoT-SORT still fails (e.g. very long occlusion).

Both layers together make ID swaps essentially disappear for a
2-person classroom scenario.
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
from collections import deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sixdrepnet import SixDRepNet

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
INPUT_SOURCE = "/Users/sarahselmene/Desktop/Pixie/aya2.mov"

BODY_OUTPUT      = "raw_body_multi.csv"
HEAD_POSE_OUTPUT = "raw_head_pose_multi.csv"
AU_OUTPUT        = "raw_action_units_multi.csv"
GAZE_OUTPUT      = "raw_gaze_multi.csv"
FACE_CROPS_DIR   = "face_crops"
OPENFACE_OUT_DIR = "openface_output"

POSE_MODEL_PATH      = "/Users/sarahselmene/Desktop/Pixie/yolo11m-pose.pt"
FACE_YOLO_MODEL_PATH = "/Users/sarahselmene/Desktop/Pixie/yolov8n-pose.pt"

OPENFACE_DIR = "/Users/sarahselmene/OpenFace/build/bin"
OPENFACE_EXE = os.path.join(OPENFACE_DIR, "FaceLandmarkImg")

# ── Performance tuning ──
FRAME_STRIDE        = 1
INFERENCE_SIZE      = 480
EXPAND_RATIO        = 0.20
OPENFACE_BATCH_SIZE = 300

# ── EMA smoothing ──
EMA_ALPHA = 0.6

# ── ID stabilizer thresholds ──
ID_MEMORY_FRAMES = 60    # frames to remember a lost track
ID_IOU_THRESH    = 0.25  # min IoU to consider same person
ID_DIST_THRESH   = 0.25  # max centroid dist (normalized by frame diagonal)

# ── COCO keypoint indices ──
KP_LEFT_WRIST  = 9
KP_RIGHT_WRIST = 10
KP_LEFT_KNEE   = 13
KP_RIGHT_KNEE  = 14
KNEE_INDICES   = {KP_LEFT_KNEE, KP_RIGHT_KNEE}
WRIST_INDICES  = {KP_LEFT_WRIST, KP_RIGHT_WRIST}
KNEE_CONF_MIN  = 0.5

COCO_KEYPOINTS = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle",
]

AU_INTENSITY_COLS = [
    "AU01_r","AU02_r","AU04_r","AU05_r","AU06_r","AU07_r",
    "AU09_r","AU10_r","AU12_r","AU14_r","AU15_r","AU17_r",
    "AU20_r","AU23_r","AU25_r","AU26_r","AU45_r",
]
AU_BINARY_COLS = [
    "AU01_c","AU02_c","AU04_c","AU05_c","AU06_c","AU07_c",
    "AU09_c","AU10_c","AU12_c","AU14_c","AU15_c","AU17_c",
    "AU20_c","AU23_c","AU25_c","AU26_c","AU28_c","AU45_c",
]
GAZE_COLS = [
    "gaze_0_x","gaze_0_y","gaze_0_z",
    "gaze_1_x","gaze_1_y","gaze_1_z",
    "gaze_angle_x","gaze_angle_y",
]

FILENAME_PATTERN = re.compile(r"frame_(\d+)_track_(\d+)")


# ──────────────────────────────────────────────
# ID STABILIZER
# ──────────────────────────────────────────────
class IDStabilizer:
    """
    Post-hoc guard against ID swaps after occlusion.

    Call stabilizer.update(current_tracks) every frame where:
        current_tracks = {raw_tracker_id: [x1, y1, x2, y2]}

    Returns a remap dict {raw_id: stable_id}.
    Use stable_id for all CSV writes and filenames.

    Logic:
    - Every track seen this frame is compared against recently-lost tracks.
    - If IoU >= ID_IOU_THRESH AND centroid distance <= ID_DIST_THRESH,
      the new raw_id is silently mapped to the old stable_id.
    - Lost tracks are forgotten after ID_MEMORY_FRAMES frames.
    """

    def __init__(self, frame_w: int, frame_h: int):
        self.diag = (frame_w**2 + frame_h**2) ** 0.5
        # stable_id → {'bbox': [...], 'lost_frames': int}
        self._lost: dict[int, dict] = {}
        # raw_id → stable_id  (persists across frames for the same raw_id)
        self._remap: dict[int, int] = {}

    @staticmethod
    def _iou(a, b) -> float:
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        iw = max(0, ix2-ix1); ih = max(0, iy2-iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / max(union, 1)

    def _dist(self, a, b) -> float:
        cax, cay = (a[0]+a[2])/2, (a[1]+a[3])/2
        cbx, cby = (b[0]+b[2])/2, (b[1]+b[3])/2
        return ((cax-cbx)**2 + (cay-cby)**2) ** 0.5 / self.diag

    def update(self, current: dict[int, list]) -> dict[int, int]:
        remap: dict[int, int] = {}
        seen_stable: set[int] = set()

        for raw_id, bbox in current.items():
            # Already remapped in a previous frame?
            if raw_id in self._remap:
                stable = self._remap[raw_id]
                remap[raw_id] = stable
                seen_stable.add(stable)
                self._lost.pop(stable, None)
                continue

            # Try to match against lost tracks
            best_stable = raw_id
            best_score  = -1.0
            for s_id, info in self._lost.items():
                iou  = self._iou(bbox, info["bbox"])
                dist = self._dist(bbox, info["bbox"])
                score = iou - dist
                if iou >= ID_IOU_THRESH and dist <= ID_DIST_THRESH and score > best_score:
                    best_score  = score
                    best_stable = s_id

            if best_stable != raw_id:
                self._remap[raw_id] = best_stable
                self._lost.pop(best_stable, None)
                print(f"\n[IDStabilizer] ID recovered: raw={raw_id} → stable={best_stable}")

            remap[raw_id] = best_stable
            seen_stable.add(best_stable)

        # Age lost tracks; evict old ones
        for s_id in list(self._lost.keys()):
            if s_id not in seen_stable:
                self._lost[s_id]["lost_frames"] += 1
                if self._lost[s_id]["lost_frames"] > ID_MEMORY_FRAMES:
                    del self._lost[s_id]
                    self._remap = {k: v for k, v in self._remap.items() if v != s_id}

        # Update last-known bbox for every stable ID seen this frame
        for raw_id, bbox in current.items():
            s_id = remap.get(raw_id, raw_id)
            self._lost[s_id] = {"bbox": list(bbox), "lost_frames": 0}

        return remap


# ──────────────────────────────────────────────
# EMA STATE  (per stable_id, per keypoint)
# ──────────────────────────────────────────────
ema_state: dict[int, dict[int, list[float]]] = {}

def apply_ema(track_id: int, kp_idx: int, x: float, y: float):
    ts = ema_state.setdefault(track_id, {})
    if kp_idx not in ts:
        ts[kp_idx] = [x, y]
    else:
        px, py = ts[kp_idx]
        x = EMA_ALPHA * x + (1.0 - EMA_ALPHA) * px
        y = EMA_ALPHA * y + (1.0 - EMA_ALPHA) * py
        ts[kp_idx] = [x, y]
    return x, y


# ──────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def expand_bbox(x1, y1, x2, y2, frame_h, frame_w, expand=EXPAND_RATIO):
    w, h = x2-x1, y2-y1
    x1 = max(0,       int(x1 - w*expand/2))
    y1 = max(0,       int(y1 - h*expand/2))
    x2 = min(frame_w, int(x2 + w*expand/2))
    y2 = min(frame_h, int(y2 + h*expand/2))
    return x1, y1, x2, y2


# ──────────────────────────────────────────────
# OPENFACE BACKGROUND WORKER
# ──────────────────────────────────────────────
class OpenFaceWorker:
    def __init__(self, face_crops_dir, openface_out_dir, batch_size=OPENFACE_BATCH_SIZE):
        self.face_crops_dir   = face_crops_dir
        self.openface_out_dir = openface_out_dir
        self.batch_size       = batch_size
        self.pending_crops    = []
        self.batch_count      = 0
        self.lock             = threading.Lock()
        self.task_queue       = queue.Queue()
        self.thread           = threading.Thread(target=self._worker_loop, daemon=True)
        self.total_processed  = 0

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
                self.task_queue.put(("batch", self.batch_count, batch))

    def flush_and_stop(self):
        with self.lock:
            if self.pending_crops:
                self.batch_count += 1
                batch = self.pending_crops.copy()
                self.pending_crops.clear()
                self.task_queue.put(("batch", self.batch_count, batch))
        self.task_queue.put(("stop", None, None))
        self.thread.join(timeout=600)
        print(f"[OpenFace] Worker stopped. Total: {self.total_processed}")

    def _worker_loop(self):
        while True:
            action, batch_id, data = self.task_queue.get()
            if action == "stop":
                break
            self._process_batch(batch_id, data)

    def _process_batch(self, batch_id, filenames):
        batch_dir     = os.path.join(self.face_crops_dir, f"_batch_{batch_id}")
        batch_out_dir = os.path.join(self.openface_out_dir, f"batch_{batch_id}")
        os.makedirs(batch_dir, exist_ok=True)
        os.makedirs(batch_out_dir, exist_ok=True)
        for fname in filenames:
            src = os.path.join(self.face_crops_dir, fname)
            if os.path.exists(src):
                os.rename(src, os.path.join(batch_dir, fname))
        cmd = [OPENFACE_EXE, "-fdir", os.path.abspath(batch_dir),
               "-out_dir", os.path.abspath(batch_out_dir),
               "-aus", "-gaze", "-multi_view", "1"]
        print(f"[OpenFace] Processing batch {batch_id} ({len(filenames)} crops)...")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               cwd=OPENFACE_DIR, timeout=300)
            print(f"[OpenFace] Batch {batch_id} {'complete' if r.returncode==0 else 'warning exit '+str(r.returncode)}")
        except subprocess.TimeoutExpired:
            print(f"[OpenFace] Batch {batch_id} timed out!")
        except Exception as e:
            print(f"[OpenFace] Batch {batch_id} error: {e}")
        self.total_processed += len(filenames)
        for fname in filenames:
            src = os.path.join(batch_dir, fname)
            if os.path.exists(src):
                os.rename(src, os.path.join(self.face_crops_dir, fname))
        try:
            os.rmdir(batch_dir)
        except OSError:
            pass


# ──────────────────────────────────────────────
# OPENFACE CSV MERGING
# ──────────────────────────────────────────────
def merge_openface_outputs(openface_out_dir, au_csv, gaze_csv):
    au_rows, gaze_rows = [], []
    for root, _, files in os.walk(openface_out_dir):
        for csv_file in files:
            if not csv_file.endswith(".csv"):
                continue
            match = FILENAME_PATTERN.search(os.path.splitext(csv_file)[0])
            if not match:
                continue
            frame_id = int(match.group(1))
            track_id = int(match.group(2))
            with open(os.path.join(root, csv_file), "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    cleaned = {k.strip(): v.strip() for k, v in row.items()}
                    conf    = float(cleaned.get("confidence", 0))
                    sv      = cleaned.get("success")
                    success = int(sv) if sv is not None else 1
                    common  = {"frame_id": frame_id, "track_id": track_id,
                               "confidence": f"{conf:.4f}", "success": success}
                    out_au, out_gz = dict(common), dict(common)
                    for c in AU_INTENSITY_COLS + AU_BINARY_COLS:
                        out_au[c]  = cleaned.get(c, "") if success else ""
                    for c in GAZE_COLS:
                        out_gz[c]  = cleaned.get(c, "") if success else ""
                    au_rows.append(out_au)
                    gaze_rows.append(out_gz)

    au_rows.sort(key=lambda r: (r["frame_id"], r["track_id"]))
    gaze_rows.sort(key=lambda r: (r["frame_id"], r["track_id"]))

    base_fields = ["frame_id","track_id","confidence","success"]
    with open(au_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=base_fields+AU_INTENSITY_COLS+AU_BINARY_COLS)
        w.writeheader(); w.writerows(au_rows)
    with open(gaze_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=base_fields+GAZE_COLS)
        w.writeheader(); w.writerows(gaze_rows)
    print(f"[OpenFace] Merged {len(au_rows)} rows → {au_csv}, {gaze_csv}")


# ──────────────────────────────────────────────
# MODEL INITIALISATION
# ──────────────────────────────────────────────
device = get_device()
print(f"[INFO] Using device: {device}")

print("[INFO] Loading YOLOv11-pose...")
pose_model = YOLO(POSE_MODEL_PATH)
if device == "mps":
    pose_model.model.to("mps")
else:
    pose_model.to(device)

print("[INFO] Loading YOLO-face model...")
face_det_model = YOLO(FACE_YOLO_MODEL_PATH)
if device == "mps":
    face_det_model.model.to("mps")
else:
    face_det_model.to(device)

print("[INFO] Initializing 6DRepNet...")
# SixDRepNet only accepts gpu_id=-1 (CPU) or a CUDA index.
# On Apple Silicon we always init on CPU, then move to MPS manually.
sixd_model = SixDRepNet(gpu_id=-1)
if device == "mps":
    sixd_model.model.to("mps")
elif device == "cuda":
    sixd_model.model.to("cuda")
else:
    print("[WARN] Running 6DRepNet on CPU — very slow!")

# ──────────────────────────────────────────────
# PREPARE DIRECTORIES
# ──────────────────────────────────────────────
print("[INFO] Cleaning up previous run data...")
for d in [FACE_CROPS_DIR, OPENFACE_OUT_DIR]:
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)

# ──────────────────────────────────────────────
# OPEN CSV FILES & WRITE HEADERS
# ──────────────────────────────────────────────
body_csv_file      = open(BODY_OUTPUT,      "w", newline="", encoding="utf-8")
head_pose_csv_file = open(HEAD_POSE_OUTPUT, "w", newline="", encoding="utf-8")
body_writer        = csv.writer(body_csv_file)
head_pose_writer   = csv.writer(head_pose_csv_file)

body_writer.writerow([
    "frame_id","track_id","landmark_idx",
    "x","y","visibility",
    "bbox_x1","bbox_y1","bbox_x2","bbox_y2",
    "global_desk_y",
])
head_pose_writer.writerow(["frame_id","track_id","pitch","yaw","roll"])


# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────
def main():
    start_time = time.time()

    openface_available = os.path.isfile(OPENFACE_EXE)
    if not openface_available:
        print(f"[WARN] OpenFace not found at {OPENFACE_EXE} — AU extraction skipped")

    if not os.path.isfile(INPUT_SOURCE):
        print(f"[ERROR] Video not found: {INPUT_SOURCE}")
        sys.exit(1)

    cap = cv2.VideoCapture(INPUT_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {INPUT_SOURCE}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"[INFO] Video: {frame_w}x{frame_h}, {total_frames} frames.")
    print(f"[INFO] Tracker: botsort (ReID) + IDStabilizer fallback")

    # ── instantiate ID stabilizer ──
    stabilizer = IDStabilizer(frame_w, frame_h)

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

            print(f"Processing frame {frame_id}/{total_frames}...", end="\r", flush=True)

            if frame_id % FRAME_STRIDE != 0:
                frame_id += 1
                continue

            # ── Step 1: YOLOv11-pose + BoT-SORT ──────────────────────────
            # botsort.yaml uses ReID embeddings → far more stable IDs during
            # occlusion than bytetrack.yaml which is purely position-based.
            results = pose_model.track(
                source=frame,
                tracker="botsort.yaml",   # ← switched from bytetrack.yaml
                persist=True,
                conf=0.25,
                iou=0.5,
                classes=[0],
                imgsz=INFERENCE_SIZE,
                stream=False,
                verbose=False,
                device=device,
            )

            boxes          = results[0].boxes
            keypoints_data = results[0].keypoints

            if boxes is None or len(boxes) == 0 or boxes.id is None:
                frame_id += 1
                continue

            xyxy_list = boxes.xyxy.cpu().numpy().astype(int)
            raw_ids   = boxes.id.cpu().numpy().astype(int)

            all_kpts = None
            if (keypoints_data is not None
                    and keypoints_data.data is not None
                    and len(keypoints_data.data) > 0):
                all_kpts = keypoints_data.data.cpu().numpy()

            # ── Step 2: IDStabilizer remap ────────────────────────────────
            current_tracks = {int(raw_ids[i]): list(xyxy_list[i])
                              for i in range(len(raw_ids))}
            remap = stabilizer.update(current_tracks)
            # remap: {raw_tracker_id → stable_id}

            # ── global_desk_y: median wrist Y across all persons ──────────
            wrist_ys = []
            if all_kpts is not None:
                for p_idx in range(len(all_kpts)):
                    for kp_idx in WRIST_INDICES:
                        kp = all_kpts[p_idx][kp_idx]
                        if float(kp[2]) >= 0.3:
                            wrist_ys.append(float(kp[1]))
            gdsk      = float(np.median(wrist_ys)) if wrist_ys else None
            gdsk_str  = f"{gdsk:.4f}" if gdsk is not None else ""

            # ── Debug overlay ─────────────────────────────────────────────
            debug_frame = frame.copy()
            cv2.putText(debug_frame, f"Frame: {frame_id}/{total_frames}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            for bbox, raw_id in zip(xyxy_list, raw_ids):
                raw_id    = int(raw_id)
                stable_id = remap.get(raw_id, raw_id)
                x1, y1, x2, y2 = bbox
                color = (0,255,0) if stable_id == raw_id else (0,165,255)
                cv2.rectangle(debug_frame, (x1,y1), (x2,y2), color, 2)
                label = f"ID:{stable_id}" + (f"(was {raw_id})" if stable_id != raw_id else "")
                cv2.putText(debug_frame, label,
                            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imshow("Tracking Debug", debug_frame)
            cv2.waitKey(1)

            # ── Process each detected person ──────────────────────────────
            for person_idx, (bbox, raw_id) in enumerate(zip(xyxy_list, raw_ids)):
                x1, y1, x2, y2 = bbox
                raw_id = int(raw_id)
                tid    = remap.get(raw_id, raw_id)   # stable ID for all writes

                # ── Body keypoints (EMA + knee threshold) ─────────────────
                if all_kpts is not None and person_idx < len(all_kpts):
                    kpts = all_kpts[person_idx]
                    for lm_idx in range(kpts.shape[0]):
                        rx   = float(kpts[lm_idx, 0])
                        ry   = float(kpts[lm_idx, 1])
                        conf = float(kpts[lm_idx, 2])

                        # Knee: low confidence → write None, skip EMA
                        if lm_idx in KNEE_INDICES and conf < KNEE_CONF_MIN:
                            body_writer.writerow([
                                frame_id, tid, lm_idx,
                                None, None, f"{conf:.4f}",
                                x1, y1, x2, y2, gdsk_str,
                            ])
                            continue

                        sx, sy = apply_ema(tid, lm_idx, rx, ry)
                        body_writer.writerow([
                            frame_id, tid, lm_idx,
                            f"{sx:.4f}", f"{sy:.4f}", f"{conf:.4f}",
                            x1, y1, x2, y2, gdsk_str,
                        ])
                else:
                    body_writer.writerow([
                        frame_id, tid, None,
                        None, None, None,
                        x1, y1, x2, y2, gdsk_str,
                    ])

                # ── Face crop + head pose ─────────────────────────────────
                cx1, cy1, cx2, cy2 = expand_bbox(x1, y1, x2, y2, frame_h, frame_w)
                cw, ch = cx2-cx1, cy2-cy1
                if cw < 10 or ch < 10:
                    head_pose_writer.writerow([frame_id, tid, None, None, None])
                    continue

                crop_bgr = frame[cy1:cy2, cx1:cx2]

                try:
                    face_results = face_det_model(crop_bgr, verbose=False)
                    face_boxes   = face_results[0].boxes if len(face_results) > 0 else None
                    best_face    = None
                    if face_boxes is not None and len(face_boxes) > 0:
                        confs        = face_boxes.conf.cpu().numpy()
                        best_idx     = np.argmax(confs)
                        if confs[best_idx] > 0.5:
                            best_face = face_boxes.xyxy.cpu().numpy()[best_idx].astype(int)

                    if best_face is None:
                        head_pose_writer.writerow([frame_id, tid, None, None, None])
                        continue

                    fx1, fy1, fx2, fy2 = best_face
                    fw, fh = fx2-fx1, fy2-fy1
                    cf1 = max(0,  int(fx1 - fw*0.25))
                    cf2 = max(0,  int(fy1 - fh*0.25))
                    cf3 = min(cw, int(fx2 + fw*0.25))
                    cf4 = min(ch, int(fy2 + fh*0.25))
                    face_crop = crop_bgr[cf2:cf4, cf1:cf3]

                    if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                        head_pose_writer.writerow([frame_id, tid, None, None, None])
                        continue

                    # Use stable tid in filename
                    fname = f"frame_{frame_id:06d}_track_{tid}.jpg"
                    cv2.imwrite(os.path.join(FACE_CROPS_DIR, fname), face_crop)
                    if of_worker:
                        of_worker.add_crop(fname)

                    pitch, yaw, roll = sixd_model.predict(face_crop)
                    head_pose_writer.writerow([
                        frame_id, tid,
                        f"{float(np.ravel(pitch)[0]):.4f}",
                        f"{float(np.ravel(yaw)[0]):.4f}",
                        f"{float(np.ravel(roll)[0]):.4f}",
                    ])

                except Exception as e:
                    print(f"\n[WARN] Face/HeadPose failed | frame {frame_id}, tid {tid}: {e}")
                    head_pose_writer.writerow([frame_id, tid, None, None, None])

            processed_frames += 1
            del frame
            gc.collect()

            if processed_frames % 100 == 0:
                body_csv_file.flush()
                head_pose_csv_file.flush()
                print(f"\n[INFO] {processed_frames} frames done (frame_id {frame_id}/{total_frames})")

            frame_id += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        body_csv_file.close()
        head_pose_csv_file.close()
        print(f"\n[INFO] Done. {frame_id} total, {processed_frames} processed.")
        if of_worker:
            print("[INFO] Waiting for OpenFace worker...")
            of_worker.flush_and_stop()
            print("[INFO] Merging OpenFace outputs...")
            merge_openface_outputs(OPENFACE_OUT_DIR, AU_OUTPUT, GAZE_OUTPUT)
        elapsed = time.time() - start_time
        h, rem = divmod(elapsed, 3600)
        m, s   = divmod(rem, 60)
        print(f"[INFO] Total time: {int(h):02d}:{int(m):02d}:{s:05.2f}")
        print("[INFO] All done!")


if __name__ == "__main__":
    main()
