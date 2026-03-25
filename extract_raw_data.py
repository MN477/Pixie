"""
extract_raw_data.py
====================
Extracts raw numerical data from a video file using MediaPipe Tasks API:
  - Pose Landmarker   → 33 body landmarks   → raw_body.csv
  - Face Landmarker   → 468 face landmarks   → raw_face.csv
  - Face Landmarker   → 4x4 head transform   → raw_head_pose.csv
  - Hand Landmarker   → 21 hand landmarks    → raw_hands.csv

No smoothing, no analysis — only raw model outputs are saved.
Frames with zero detections across all models are skipped.
Model .task files are auto-downloaded if missing.

Usage:
    python extract_raw_data.py
"""

import csv
import sys
import gc
import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
INPUT_SOURCE = "test_vid/Sitting_on_Desk_016.mp4"

# Output file paths
BODY_OUTPUT      = "raw_body.csv"
FACE_OUTPUT      = "raw_face.csv"
HEAD_POSE_OUTPUT = "raw_head_pose.csv"
HANDS_OUTPUT     = "raw_hands.csv"
BLENDSHAPES_OUTPUT = "raw_blendshapes.csv"
WORLD_BODY_OUTPUT = "raw_body_world.csv"

# MediaPipe Task model files
FACE_MODEL_PATH = "face_landmarker.task"
HAND_MODEL_PATH = "hand_landmarker.task"
POSE_MODEL_PATH = "pose_landmarker_full.task"

# Download URLs (Google-hosted, stable links)
MODEL_URLS = {
    FACE_MODEL_PATH: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    HAND_MODEL_PATH: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    POSE_MODEL_PATH: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
}

# MediaPipe settings
MP_MAX_FACES = 10
MP_MAX_HANDS = 10
MP_MAX_POSES = 10

# ──────────────────────────────────────────────
# AUTO-DOWNLOAD MODELS
# ──────────────────────────────────────────────
def ensure_model(path, url):
    if not os.path.exists(path):
        print(f"[INFO] Downloading {path}...")
        urllib.request.urlretrieve(url, path)
        print(f"[INFO] Saved {path} ({os.path.getsize(path) / 1e6:.1f} MB)")

for model_path, model_url in MODEL_URLS.items():
    ensure_model(model_path, model_url)

# ──────────────────────────────────────────────
# MODEL INITIALIZATION
# ──────────────────────────────────────────────
BaseOptions = python.BaseOptions
VisionRunningMode = vision.RunningMode

print("[INFO] Initializing MediaPipe Pose Landmarker...")
pose_options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_poses=MP_MAX_POSES,
)
pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

print("[INFO] Initializing MediaPipe Face Landmarker...")
face_options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=MP_MAX_FACES,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
)
face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

print("[INFO] Initializing MediaPipe Hand Landmarker...")
hand_options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=MP_MAX_HANDS,
)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

# ──────────────────────────────────────────────
# OPEN CSV FILES
# ──────────────────────────────────────────────
body_csv_file      = open(BODY_OUTPUT, "w", newline="", encoding="utf-8")
face_csv_file      = open(FACE_OUTPUT, "w", newline="", encoding="utf-8")
head_pose_csv_file = open(HEAD_POSE_OUTPUT, "w", newline="", encoding="utf-8")
hands_csv_file     = open(HANDS_OUTPUT, "w", newline="", encoding="utf-8")
blend_csv_file = open(BLENDSHAPES_OUTPUT, "w", newline="", encoding="utf-8")
world_body_csv_file = open(WORLD_BODY_OUTPUT, "w", newline="", encoding="utf-8")



body_writer      = csv.writer(body_csv_file)
face_writer      = csv.writer(face_csv_file)
head_pose_writer = csv.writer(head_pose_csv_file)
hands_writer     = csv.writer(hands_csv_file)
blend_writer     = csv.writer(blend_csv_file)
world_body_writer = csv.writer(world_body_csv_file)
# Write headers
body_writer.writerow([
    "frame_id", "person_id", "landmark_idx", "x", "y", "z", "visibility", "presence"
])
face_writer.writerow([
    "frame_id", "face_id", "landmark_idx", "x", "y", "z"
])
head_pose_writer.writerow([
    "frame_id", "face_id",
    "m00", "m01", "m02", "m03",
    "m10", "m11", "m12", "m13",
    "m20", "m21", "m22", "m23",
    "m30", "m31", "m32", "m33",
])
hands_writer.writerow([
    "frame_id", "hand_id", "hand_label", "landmark_idx", "x", "y", "z"
])
blend_writer.writerow([
    "frame_id",
    "face_id",
    "blendshape_name",
    "score"
])
world_body_writer.writerow([
    "frame_id",
    "person_id",
    "landmark_idx",
    "x", "y", "z",
    "visibility", "presence"
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
    print(f"[INFO] Video has {total_frames} frames.")

    frame_id = 0
    saved_frames = 0
    print("[INFO] Starting extraction loop...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video stream.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            has_detections = False

            # ── Pose (Body) ────────────────────────────
            try:
                pose_result = pose_landmarker.detect(mp_image)
                if pose_result and pose_result.pose_landmarks:
                    has_detections = True
                    for person_id, person_lm in enumerate(pose_result.pose_landmarks):
                        for lm_idx, lm in enumerate(person_lm):
                            body_writer.writerow([
                                frame_id,
                                person_id,
                                lm_idx,
                                f"{lm.x:.8f}",
                                f"{lm.y:.8f}",
                                f"{lm.z:.8f}",
                                f"{lm.visibility:.8f}",
                                f"{lm.presence:.8f}",
                            ])
                if pose_result and pose_result.pose_world_landmarks:
                    for person_id, person_lm in enumerate(pose_result.pose_world_landmarks):
                        for lm_idx, lm in enumerate(person_lm):
                            world_body_writer.writerow([
                                frame_id,
                                person_id,
                                lm_idx,
                                f"{lm.x:.8f}",
                                f"{lm.y:.8f}",
                                f"{lm.z:.8f}",
                                f"{lm.visibility:.8f}",
                                f"{lm.presence:.8f}",
            ])


            except Exception as e:
                print(f"[WARN] Pose Landmarker failed on frame {frame_id}: {e}")

            # ── Face + Head Pose ───────────────────────
            try:
                face_result = face_landmarker.detect(mp_image)
                if face_result and face_result.face_landmarks:
                    has_detections = True
                    for face_id, face_lm in enumerate(face_result.face_landmarks):
                        for lm_idx, lm in enumerate(face_lm):
                            face_writer.writerow([
                                frame_id,
                                face_id,
                                lm_idx,
                                f"{lm.x:.8f}",
                                f"{lm.y:.8f}",
                                f"{lm.z:.8f}",
                            ])

                    # Head pose from transformation matrices
                    if face_result.facial_transformation_matrixes:
                        for face_id, matrix in enumerate(face_result.facial_transformation_matrixes):
                            m = np.array(matrix).flatten()
                            head_pose_writer.writerow([
                                frame_id,
                                face_id,
                                *[f"{v:.8f}" for v in m],
                            ])
                    
                    if face_result.face_blendshapes:
                        for face_id, blend_list in enumerate(face_result.face_blendshapes):
                            for blendshape in blend_list:
                                blend_writer.writerow([
                                    frame_id,
                                    face_id,
                                    blendshape.category_name,
                                    f"{blendshape.score:.8f}"
            ])
            except Exception as e:
                print(f"[WARN] Face Landmarker failed on frame {frame_id}: {e}")

            # ── Hands ──────────────────────────────────
            try:
                hand_result = hand_landmarker.detect(mp_image)
                if hand_result and hand_result.hand_landmarks:
                    has_detections = True
                    for hand_id, hand_lm in enumerate(hand_result.hand_landmarks):
                        hand_label = "unknown"
                        if hand_result.handedness and hand_id < len(hand_result.handedness):
                            hand_label = hand_result.handedness[hand_id][0].category_name.lower()

                        for lm_idx, lm in enumerate(hand_lm):
                            hands_writer.writerow([
                                frame_id,
                                hand_id,
                                hand_label,
                                lm_idx,
                                f"{lm.x:.8f}",
                                f"{lm.y:.8f}",
                                f"{lm.z:.8f}",
                            ])
            except Exception as e:
                print(f"[WARN] Hand Landmarker failed on frame {frame_id}: {e}")

            # ── Skip frames with zero detections ───────
            if not has_detections:
                del frame, frame_rgb, mp_image
                frame = None
                frame_id += 1
                continue

            saved_frames += 1

            # ── Privacy guard — destroy frame ──────────
            del frame, frame_rgb, mp_image
            frame = None
            gc.collect()

            if frame_id % 100 == 0 and frame_id > 0:
                body_csv_file.flush()
                face_csv_file.flush()
                head_pose_csv_file.flush()
                hands_csv_file.flush()
                print(f"[INFO] Processed {frame_id}/{total_frames} frames ({saved_frames} with detections)...")


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

        print(f"[INFO] Done. {frame_id} total frames, {saved_frames} saved.")
        print(f"  → {BODY_OUTPUT}")
        print(f"  → {FACE_OUTPUT}")
        print(f"  → {HEAD_POSE_OUTPUT}")
        print(f"  → {HANDS_OUTPUT}")

if __name__ == "__main__":
    main()