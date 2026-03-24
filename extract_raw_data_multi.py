"""
extract_raw_data_multi.py
=========================
Multi-person landmark extraction using:
  - YOLO26s + ByteTrack  → person detection & tracking
  - MediaPipe Tasks API  → Pose, Face, Hand landmarking on per-person crops

Pipeline per frame:
  1. YOLO detect + ByteTrack → bounding boxes with persistent track IDs
  2. For each person: expand bbox 20%, make square, crop
  3. Run MediaPipe landmarkers on the crop
  4. Remap normalised coords back to global frame
  5. Write to CSV with track_id for identity persistence

Outputs:
  raw_body.csv, raw_face.csv, raw_head_pose.csv,
  raw_hands.csv, raw_blendshapes.csv

Usage:
    python extract_raw_data_multi.py
"""

import csv
import gc
import os
import sys
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
INPUT_SOURCE ="testing_vid/2stud.mp4"

BODY_OUTPUT        = "raw_body_multi.csv"
FACE_OUTPUT        = "raw_face_multi.csv"
HEAD_POSE_OUTPUT   = "raw_head_pose_multi.csv"
HANDS_OUTPUT       = "raw_hands_multi.csv"
BLENDSHAPES_OUTPUT = "raw_blendshapes_multi.csv"

# YOLO model
YOLO_MODEL_PATH = "yolo11m.pt"

# MediaPipe task files
FACE_MODEL_PATH = "face_landmarker.task"
HAND_MODEL_PATH = "hand_landmarker.task"
POSE_MODEL_PATH = "pose_landmarker_full.task"

MODEL_URLS = {
    FACE_MODEL_PATH: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    HAND_MODEL_PATH: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    POSE_MODEL_PATH: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
}

EXPAND_RATIO = 0.20  # expand bounding box by 20%

# ──────────────────────────────────────────────
# AUTO-DOWNLOAD MEDIAPIPE MODELS
# ──────────────────────────────────────────────
def ensure_model(path, url):
    if not os.path.exists(path):
        print(f"[INFO] Downloading {path}...")
        urllib.request.urlretrieve(url, path)
        print(f"[INFO] Saved {path} ({os.path.getsize(path) / 1e6:.1f} MB)")

for model_path, model_url in MODEL_URLS.items():
    ensure_model(model_path, model_url)

# ──────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────
def expand_and_square_bbox(x1, y1, x2, y2, frame_h, frame_w, expand=EXPAND_RATIO):
    """Expand a bounding box by `expand` ratio, make it square, and clip to frame."""
    w = x2 - x1
    h = y2 - y1

    # Expand by the ratio
    pad_w = w * expand / 2
    pad_h = h * expand / 2
    x1 -= pad_w
    y1 -= pad_h
    x2 += pad_w
    y2 += pad_h

    # Make square (use the larger dimension)
    new_w = x2 - x1
    new_h = y2 - y1
    side = max(new_w, new_h)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    x1 = cx - side / 2
    y1 = cy - side / 2
    x2 = cx + side / 2
    y2 = cy + side / 2

    # Clip to frame boundaries
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(frame_w, int(x2))
    y2 = min(frame_h, int(y2))

    return x1, y1, x2, y2


def remap_landmark(lm_x, lm_y, crop_x, crop_y, crop_w, crop_h):
    """Convert normalised crop-local coords (0-1) to global pixel coords."""
    gx = crop_x + lm_x * crop_w
    gy = crop_y + lm_y * crop_h
    return gx, gy


# ──────────────────────────────────────────────
# MODEL INITIALISATION
# ──────────────────────────────────────────────
print("[INFO] Loading YOLO26s model...")
yolo_model = YOLO(YOLO_MODEL_PATH)

BaseOptions = python.BaseOptions
VisionRunningMode = vision.RunningMode

# Each crop has exactly one person, so num_* = 1 (hands = 2 for left+right)
print("[INFO] Initializing MediaPipe Pose Landmarker...")
pose_landmarker = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
    )
)

print("[INFO] Initializing MediaPipe Face Landmarker...")
face_landmarker = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
)

print("[INFO] Initializing MediaPipe Hand Landmarker...")
hand_landmarker = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
    )
)

# ──────────────────────────────────────────────
# OPEN CSV FILES & WRITE HEADERS
# ──────────────────────────────────────────────
body_csv_file      = open(BODY_OUTPUT, "w", newline="", encoding="utf-8")
face_csv_file      = open(FACE_OUTPUT, "w", newline="", encoding="utf-8")
head_pose_csv_file = open(HEAD_POSE_OUTPUT, "w", newline="", encoding="utf-8")
hands_csv_file     = open(HANDS_OUTPUT, "w", newline="", encoding="utf-8")
blend_csv_file     = open(BLENDSHAPES_OUTPUT, "w", newline="", encoding="utf-8")

body_writer      = csv.writer(body_csv_file)
face_writer      = csv.writer(face_csv_file)
head_pose_writer = csv.writer(head_pose_csv_file)
hands_writer     = csv.writer(hands_csv_file)
blend_writer     = csv.writer(blend_csv_file)

body_writer.writerow([
    "frame_id", "track_id", "landmark_idx", "x", "y", "z", "visibility", "presence"
])
face_writer.writerow([
    "frame_id", "track_id", "landmark_idx", "x", "y", "z"
])
head_pose_writer.writerow([
    "frame_id", "track_id",
    "m00", "m01", "m02", "m03",
    "m10", "m11", "m12", "m13",
    "m20", "m21", "m22", "m23",
    "m30", "m31", "m32", "m33",
])
hands_writer.writerow([
    "frame_id", "track_id", "hand_label", "landmark_idx", "x", "y", "z"
])
blend_writer.writerow([
    "frame_id", "track_id", "blendshape_name", "score"
])

# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────
def main():
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

    frame_id = 0
    saved_frames = 0
    print("[INFO] Starting extraction loop...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video stream.")
                break

            # ── Step 1: YOLO detection + ByteTrack ────

            results = yolo_model.track(
                source=frame,
                tracker="bytetrack.yaml",
                persist=True,
                conf=0.25,   # lower = detect more people
                iou=0.5,
                classes=[0],  # person only
                stream=False,
                verbose=False,
            )

            boxes = results[0].boxes

            if boxes is None or len(boxes) == 0:
                print(f"[DEBUG] No detections at frame {frame_id}")
                frame_id += 1
                continue

            # DEBUG: check tracking IDs
            print(f"[DEBUG] Frame {frame_id} | Detections: {len(boxes)}")

            if boxes.id is None:
                print(f"[ERROR] No tracking IDs at frame {frame_id} — tracking NOT working")
                frame_id += 1
                continue  # DO NOT fallback to fake IDs anymore


            # Get bounding boxes and track IDs
            xyxy_list = boxes.xyxy.cpu().numpy().astype(int)
            track_ids = boxes.id.cpu().numpy().astype(int)
            print(f"[DEBUG] Track IDs: {track_ids}")
            debug_frame = frame.copy()

            for bbox, track_id in zip(xyxy_list, track_ids):
                x1, y1, x2, y2 = bbox
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    debug_frame,
                    f"ID:{track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Tracking Debug", debug_frame)
            cv2.waitKey(1)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_has_detections = False

            # ── Step 2-6: Process each detected person ─
            print(f"[DEBUG] Processing {len(xyxy_list)} detections for frame {frame_id}")
            processed_count = 0
            
            for bbox, track_id in zip(xyxy_list, track_ids):
                x1, y1, x2, y2 = bbox
                tid = int(track_id)
                print(f"[DEBUG] Processing track ID {tid} with bbox {bbox}")

                # Expand, square, clip
                cx1, cy1, cx2, cy2 = expand_and_square_bbox(
                    x1, y1, x2, y2, frame_h, frame_w
                )
                crop_w = cx2 - cx1
                crop_h = cy2 - cy1
                if crop_w < 10 or crop_h < 10:
                    print(f"[DEBUG] Skipping track ID {tid} - crop too small: {crop_w}x{crop_h}")
                    continue  # skip tiny crops

                print(f"[DEBUG] Crop for track ID {tid}: ({cx1},{cy1}) to ({cx2},{cy2}) size {crop_w}x{crop_h}")
                crop = frame_rgb[cy1:cy2, cx1:cx2]
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop)
                )

                # ── Pose (Body) ───────────────────────
                try:
                    pose_result = pose_landmarker.detect(mp_image)
                    if pose_result and pose_result.pose_landmarks:
                        frame_has_detections = True
                        processed_count += 1
                        print(f"[DEBUG] Pose landmarks found for track ID {tid}")
                        for person_lm in pose_result.pose_landmarks:
                            for lm_idx, lm in enumerate(person_lm):
                                gx, gy = remap_landmark(
                                    lm.x, lm.y, cx1, cy1, crop_w, crop_h
                                )
                                body_writer.writerow([
                                    frame_id, tid, lm_idx,
                                    f"{gx:.8f}", f"{gy:.8f}", f"{lm.z:.8f}",
                                    f"{lm.visibility:.8f}", f"{lm.presence:.8f}",
                                ])
                    else:
                        print(f"[DEBUG] No pose landmarks for track ID {tid}")
                except Exception as e:
                    print(f"[WARN] Pose failed | frame {frame_id}, track {tid}: {e}")

                # ── Face + Head Pose + Blendshapes ────
                try:
                    face_result = face_landmarker.detect(mp_image)
                    if face_result and face_result.face_landmarks:
                        frame_has_detections = True
                        for face_lm in face_result.face_landmarks:
                            for lm_idx, lm in enumerate(face_lm):
                                gx, gy = remap_landmark(
                                    lm.x, lm.y, cx1, cy1, crop_w, crop_h
                                )
                                face_writer.writerow([
                                    frame_id, tid, lm_idx,
                                    f"{gx:.8f}", f"{gy:.8f}", f"{lm.z:.8f}",
                                ])

                        # Head pose (transformation matrix)
                        if face_result.facial_transformation_matrixes:
                            for matrix in face_result.facial_transformation_matrixes:
                                m = np.array(matrix).flatten()
                                head_pose_writer.writerow([
                                    frame_id, tid,
                                    *[f"{v:.8f}" for v in m],
                                ])

                        # Blendshapes
                        if face_result.face_blendshapes:
                            for blend_list in face_result.face_blendshapes:
                                for bs in blend_list:
                                    blend_writer.writerow([
                                        frame_id, tid,
                                        bs.category_name,
                                        f"{bs.score:.8f}",
                                    ])
                except Exception as e:
                    print(f"[WARN] Face failed | frame {frame_id}, track {tid}: {e}")

                # ── Hands ─────────────────────────────
                try:
                    hand_result = hand_landmarker.detect(mp_image)
                    if hand_result and hand_result.hand_landmarks:
                        frame_has_detections = True
                        for hand_idx, hand_lm in enumerate(hand_result.hand_landmarks):
                            hand_label = "unknown"
                            if (hand_result.handedness
                                    and hand_idx < len(hand_result.handedness)):
                                hand_label = (
                                    hand_result.handedness[hand_idx][0]
                                    .category_name.lower()
                                )
                            for lm_idx, lm in enumerate(hand_lm):
                                gx, gy = remap_landmark(
                                    lm.x, lm.y, cx1, cy1, crop_w, crop_h
                                )
                                hands_writer.writerow([
                                    frame_id, tid, hand_label, lm_idx,
                                    f"{gx:.8f}", f"{gy:.8f}", f"{lm.z:.8f}",
                                ])
                except Exception as e:
                    print(f"[WARN] Hand failed | frame {frame_id}, track {tid}: {e}")

            # ── Housekeeping ──────────────────────────
            if frame_has_detections:
                saved_frames += 1

            print(f"[DEBUG] Frame {frame_id} summary: {len(xyxy_list)} detections, {processed_count} processed")
            
            del frame, frame_rgb
            gc.collect()

            if frame_id % 100 == 0 and frame_id > 0:
                body_csv_file.flush()
                face_csv_file.flush()
                head_pose_csv_file.flush()
                hands_csv_file.flush()
                blend_csv_file.flush()
                print(
                    f"[INFO] Processed {frame_id}/{total_frames} frames "
                    f"({saved_frames} with detections)..."
                )

            frame_id += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C).")

    finally:
        cap.release()

        pose_landmarker.close()
        face_landmarker.close()
        hand_landmarker.close()

        body_csv_file.close()
        face_csv_file.close()
        head_pose_csv_file.close()
        hands_csv_file.close()
        blend_csv_file.close()

        print(f"[INFO] Done. {frame_id} total frames, {saved_frames} with detections.")
        print(f"  → {BODY_OUTPUT}")
        print(f"  → {FACE_OUTPUT}")
        print(f"  → {HEAD_POSE_OUTPUT}")
        print(f"  → {HANDS_OUTPUT}")
        print(f"  → {BLENDSHAPES_OUTPUT}")


if __name__ == "__main__":
    main()
