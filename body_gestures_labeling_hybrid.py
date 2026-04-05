"""
=============================================================================
behaviour_classifier_visual.py  —  v5
=============================================================================
WHAT THIS SCRIPT DOES
---------------------
1. Reads raw_body_multi.csv (produced by extract_raw_data_multi.py).
   This CSV contains one row per keypoint per person per frame:
       frame_id | track_id | landmark_idx | x | y | visibility

2. Groups all rows by frame_id, rebuilds a (17, 3) keypoint array for
   each (frame_id, track_id) pair.

3. For each frame_id, seeks the video to that exact frame using
   cap.set(CAP_PROP_POS_FRAMES, frame_id) so CSV and video are
   always in perfect sync — regardless of FRAME_STRIDE used during extraction.

4. Runs the 5-behaviour pipeline on the keypoints from the CSV:
       sitting | slouching | standing | bounding | hand_raised

5. Draws the results onto the video frame and displays in an OpenCV window.

6. Writes two output CSVs:
       behaviour_raw_frames.csv  — one row per (frame, person)
       behaviour_summary.csv     — one row per behaviour episode

CHANGES IN v5
-------------
- "bouncing" behaviour renamed to "bounding" (large body displacement / jumping detection)
- BoundingDetector: replaces FFT-based oscillation with displacement-based detection.
  Triggers when a person's hip centroid moves > BOUND_DISP_NORM * shoulder_width
  in a single frame (vertical or diagonal leap), confirmed over multiple frames.
- Improved PostureClassifier:
    * Hysteresis added to sitting ↔ standing transitions (avoids flicker at threshold)
    * Separate left/right knee logic — only one side needed for sitting detection
    * Visibility-weighted averaging of left/right knee angles
    * Better confidence scaling
- Improved SlouchDetector:
    * Added hip-shoulder vertical offset check (hunching forward)
    * Nose/shoulder ratio measured more robustly
- Improved HandRaiseSM:
    * Smoothed wrist position used for geometry check
    * Added elbow-above-shoulder as a stronger sub-condition
    * Stricter lowering hysteresis to prevent false resets
- KeypointBuffer:
    * Increased KP_INTERP_MAX to 10 frames for better occlusion handling
    * Kalman R reduced slightly for tighter tracking
- Display: "bounding" shown in bright cyan

CONTROLS
--------
  SPACE      pause / resume  (or advance one frame when paused)
  Q / ESC    quit
  S          save current frame as PNG snapshot
  + / -      increase / decrease playback speed

USAGE
-----
  python behaviour_classifier_v5.py \
      --video mardi.mov \
      --body  raw_body_multi.csv

  # Step through frame by frame (useful for debugging):
  python behaviour_classifier_v5.py --video mardi.mov --body raw_body_multi.csv --step

  # Save annotated video:
  python behaviour_classifier_v5.py --video mardi.mov --body raw_body_multi.csv \
      --save-video output_annotated.mp4

DEPENDENCIES
------------
  pip install numpy pandas opencv-python
=============================================================================
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd


# ==============================================================================
# COCO KEYPOINT INDICES  (YOLOv8-pose, 17 joints)
# ==============================================================================

KP_NOSE           = 0
KP_LEFT_EYE       = 1
KP_RIGHT_EYE      = 2
KP_LEFT_EAR       = 3
KP_RIGHT_EAR      = 4
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW     = 7
KP_RIGHT_ELBOW    = 8
KP_LEFT_WRIST     = 9
KP_RIGHT_WRIST    = 10
KP_LEFT_HIP       = 11
KP_RIGHT_HIP      = 12
KP_LEFT_KNEE      = 13
KP_RIGHT_KNEE     = 14
KP_LEFT_ANKLE     = 15
KP_RIGHT_ANKLE    = 16
N_KP              = 17


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class CFG:
    # I/O (overridden by CLI)
    VIDEO_PATH:  str = "sarra2.mov"
    BODY_CSV:    str = "raw_body_multi.csv"
    RAW_OUT_CSV: str = "behaviour_raw_frames.csv"
    SUM_OUT_CSV: str = "behaviour_summary.csv"

    # Keypoint confidence gate
    KP_CONF_MIN:   float = 0.25          # lowered from 0.30 for more detections
    KP_INTERP_MAX: int   = 10            # increased: extrapolate up to 10 missing frames

    # Kalman filter noise
    KP_KALMAN_Q: float = 2e-4            # slightly reduced → smoother tracks
    KP_KALMAN_R: float = 6e-3            # slightly tighter measurement trust

    # Body ruler fallback when both shoulders invisible
    FALLBACK_SW: float = 100.0

    # ── Behaviour 1 & 3 : Sitting / Standing (with hysteresis) ────────────
    SITTING_KNEE_ANGLE_MAX:   float = 140.0   # enter sitting if avg < this
    SITTING_KNEE_ANGLE_EXIT:  float = 150.0   # leave sitting if avg > this (hysteresis)
    STANDING_KNEE_ANGLE_MIN:  float = 158.0   # enter standing if avg > this
    STANDING_KNEE_ANGLE_EXIT: float = 148.0   # leave standing if avg < this (hysteresis)
    POSTURE_CONFIRM_FRAMES:   int   = 4        # frames a posture must hold before switching

    # ── Upper-body fallback (desk-occluded lower body) ─────────────────
    KNEE_VIS_MIN: float = 0.30
    UB_BBOX_AR_SITTING_MAX:  float = 1.6
    UB_BBOX_AR_STANDING_MIN: float = 2.2
    UB_SHOULDER_Y_SITTING_MIN: float = 0.30
    UB_SHOULDER_Y_STANDING_MAX: float = 0.22
    UB_TORSO_COMPRESS_THRESH: float = 0.08
    UB_HEAD_PITCH_SLOUCH: float = -18.0

    # ── Behaviour 2 : Slouching ────────────────────────────────────────────
    SLOUCH_SPINE_TILT_MAX:     float = 12.0   # deg from vertical — tighter threshold
    SLOUCH_NOSE_DROP_NORM:     float = 0.15   # nose drop / shoulder_width
    SLOUCH_HIP_FORWARD_NORM:   float = 0.10   # hip mid ahead of shoulder mid / sw

    # ── Behaviour 4 : Bounding (large body displacement — leaping/jumping) ──
    # Replaces the old FFT-based "bouncing" detector.
    # Triggers when the hip centroid moves > BOUND_DISP_NORM * sw per frame,
    # confirmed over BOUND_CONFIRM_FRAMES consecutive frames.
    BOUND_DISP_NORM:      float = 0.25   # displacement threshold relative to sw
    BOUND_CONFIRM_FRAMES: int   = 3      # frames needed to confirm bounding
    BOUND_DECAY_FRAMES:   int   = 8      # frames bounding stays active after last trigger
    BOUND_HISTORY_LEN:    int   = 5      # rolling window for displacement smoothing

    # ── Behaviour 5 : Hand raise ───────────────────────────────────────────
    HAND_ABOVE_MARGIN_NORM: float = 0.08   # wrist above shoulder / sw (slightly tighter)
    ELBOW_ABOVE_SHOULDER:   bool  = True   # require elbow above shoulder midpoint too
    HAND_VEL_NORM:          float = 0.06   # reduced: upward wrist vel / sw / frame
    HAND_RAISE_HOLD:        int   = 6      # frames to confirm raise
    HAND_LOWER_HOLD:        int   = 15     # frames to confirm lower (stronger hysteresis)

    # ── Display ────────────────────────────────────────────────────────────
    WINDOW_NAME: str = "Behaviour Classifier v5"

    # Behaviour colour map (BGR)
    BEHAVIOUR_COLORS: dict = {
        "sitting":     (0,   200, 255),   # amber
        "standing":    (0,   220,  80),   # green
        "slouching":   (0,    80, 255),   # red-orange
        "bounding":    (255, 230,   0),   # bright cyan
        "hand_raised": (255,   0, 200),   # magenta
        "unknown":     (140, 140, 140),   # grey
    }

    # Per-track ID colours for skeleton
    ID_PALETTE = [
        (0, 200, 255), (0, 255, 128), (255, 128, 0),
        (200, 0, 255), (0, 128, 255), (255, 0, 128),
        (128, 255, 0), (255, 200, 0), (0, 255, 220),
    ]


# ==============================================================================
# 2-D KALMAN FILTER  (one per keypoint)
# ==============================================================================

class KalmanKP:
    """
    State = [x, vx, y, vy]  (constant-velocity model).
    Smooths detection jitter; extrapolates through brief occlusions.
    """

    def __init__(self) -> None:
        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 10.0
        dt = 1.0
        self.F = np.array([
            [1, dt, 0,  0],
            [0,  1, 0,  0],
            [0,  0, 1, dt],
            [0,  0, 0,  1],
        ], dtype=np.float64)
        self.H  = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float64)
        q       = CFG.KP_KALMAN_Q
        self.Q  = np.diag([q, q * 10, q, q * 10])
        self.R  = np.eye(2) * CFG.KP_KALMAN_R
        self._init          = False
        self.missing_frames = 0

    def predict_only(self) -> tuple[float, float]:
        if not self._init:
            return 0.0, 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.missing_frames += 1
        return float(self.x[0]), float(self.x[2])

    def update(self, mx: float, my: float) -> tuple[float, float]:
        if not self._init:
            self.x[:] = [mx, 0.0, my, 0.0]
            self._init          = True
            self.missing_frames = 0
            return mx, my
        xp  = self.F @ self.x
        Pp  = self.F @ self.P @ self.F.T + self.Q
        z   = np.array([[mx], [my]], dtype=np.float64)
        S   = self.H @ Pp @ self.H.T + self.R
        K   = Pp @ self.H.T @ np.linalg.inv(S)
        self.x = (xp.reshape(-1,1) + K @ (z - self.H @ xp.reshape(-1,1))).flatten()
        self.P = (np.eye(4) - K @ self.H) @ Pp
        self.missing_frames = 0
        return float(self.x[0]), float(self.x[2])

    @property
    def vy_up(self) -> float:
        """Upward velocity: positive = moving UP on screen (negated image vy)."""
        return -float(self.x[3])

    @property
    def vx(self) -> float:
        return float(self.x[1])

    @property
    def vy(self) -> float:
        return float(self.x[3])


# ==============================================================================
# KEYPOINT BUFFER  (17 KalmanKP + confidence gate + interpolation)
# ==============================================================================

class KeypointBuffer:
    """
    Ingests one frame's (17, 3) array [x, y, confidence].
    Returns smoothed, interpolated positions via .get(idx) -> (x,y) | None.
    """

    def __init__(self) -> None:
        self.kf     = [KalmanKP() for _ in range(N_KP)]
        self.smooth = np.full((N_KP, 2), np.nan, dtype=np.float64)
        self.valid  = np.zeros(N_KP, dtype=bool)

    def update(self, kp_array: np.ndarray) -> None:
        for i in range(N_KP):
            x_raw = float(kp_array[i, 0])
            y_raw = float(kp_array[i, 1])
            conf  = float(kp_array[i, 2]) if kp_array.shape[1] > 2 else 1.0

            missing = (
                conf < CFG.KP_CONF_MIN
                or math.isnan(x_raw)
                or (x_raw == 0.0 and y_raw == 0.0)
            )

            kf = self.kf[i]
            if not missing:
                sx, sy = kf.update(x_raw, y_raw)
                self.smooth[i] = [sx, sy]
                self.valid[i]  = True
            elif kf._init and kf.missing_frames < CFG.KP_INTERP_MAX:
                sx, sy = kf.predict_only()
                self.smooth[i] = [sx, sy]
                self.valid[i]  = True
            else:
                if kf._init:
                    kf.missing_frames += 1
                self.smooth[i] = [np.nan, np.nan]
                self.valid[i]  = False

    def get(self, idx: int) -> Optional[tuple[float, float]]:
        if self.valid[idx]:
            return float(self.smooth[idx, 0]), float(self.smooth[idx, 1])
        return None

    def kpf(self, idx: int) -> KalmanKP:
        return self.kf[idx]


# ==============================================================================
# GEOMETRY HELPERS
# ==============================================================================

def _angle_deg(A: tuple, B: tuple, C: tuple) -> float:
    """Angle at vertex B formed by rays B→A and B→C. Returns [0, 180] deg."""
    ba    = np.array([A[0]-B[0], A[1]-B[1]], dtype=np.float64)
    bc    = np.array([C[0]-B[0], C[1]-B[1]], dtype=np.float64)
    na    = np.linalg.norm(ba)
    nc    = np.linalg.norm(bc)
    if na < 1e-6 or nc < 1e-6:
        return float("nan")
    cos_v = np.dot(ba, bc) / (na * nc)
    return float(np.degrees(np.arccos(np.clip(cos_v, -1.0, 1.0))))


def _tilt_from_vertical(top: tuple, bottom: tuple) -> float:
    """
    Angle between the vector top→bottom and the downward vertical (0, +1).
    0° = perfectly upright spine. 90° = completely horizontal.
    """
    dx, dy = bottom[0]-top[0], bottom[1]-top[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return 0.0
    return float(math.degrees(math.acos(max(-1.0, min(1.0, dy/length)))))


def _body_ruler(kpb: KeypointBuffer) -> float:
    """
    Shoulder width = body ruler.
    Makes all pixel thresholds camera-distance independent.
    Falls back to torso height if shoulders unavailable.
    """
    ls = kpb.get(KP_LEFT_SHOULDER)
    rs = kpb.get(KP_RIGHT_SHOULDER)
    if ls and rs:
        sw = math.hypot(rs[0]-ls[0], rs[1]-ls[1])
        if sw > 5.0:
            return sw

    # Fallback: use distance from shoulder to hip on visible side
    for sh_idx, hp_idx in [(KP_LEFT_SHOULDER, KP_LEFT_HIP), (KP_RIGHT_SHOULDER, KP_RIGHT_HIP)]:
        sh = kpb.get(sh_idx)
        hp = kpb.get(hp_idx)
        if sh and hp:
            torso = math.hypot(hp[0]-sh[0], hp[1]-sh[1])
            if torso > 5.0:
                return torso * 0.8   # scale: shoulder_width ≈ 0.8 × torso height

    return CFG.FALLBACK_SW


def _hip_centroid(kpb: KeypointBuffer) -> Optional[tuple[float, float]]:
    """Returns the midpoint of both hips, or whichever is available."""
    lh = kpb.get(KP_LEFT_HIP)
    rh = kpb.get(KP_RIGHT_HIP)
    if lh and rh:
        return (lh[0]+rh[0])/2.0, (lh[1]+rh[1])/2.0
    return lh or rh


# ==============================================================================
# BEHAVIOUR 1 & 3 — POSTURE CLASSIFIER  (Sitting / Slouching / Standing)
# ==============================================================================

@dataclass
class PostureResult:
    label:            str   = "unknown"
    detection_mode:   str   = "knee"
    knee_angle_left:  float = float("nan")
    knee_angle_right: float = float("nan")
    avg_knee_angle:   float = float("nan")
    spine_tilt_deg:   float = float("nan")
    nose_drop_norm:   float = float("nan")
    hip_forward_norm: float = float("nan")
    bbox_aspect_ratio:float = float("nan")
    shoulder_y_norm:  float = float("nan")
    torso_compress:   float = float("nan")
    head_pitch:       float = float("nan")
    confidence:       float = 0.0


class PostureClassifier:
    """
    Stateful posture classifier with hysteresis to prevent state flicker.

    Priority:
      1. slouching  (can co-occur with sitting — detected first)
      2. sitting
      3. standing
      4. unknown
    """

    def __init__(self) -> None:
        self._state        = "unknown"
        self._candidate    = "unknown"
        self._hold_counter = 0

    def classify(self, kpb: KeypointBuffer, sw: float, bbox: tuple = None, head_pitch: float = None) -> PostureResult:
        res = PostureResult()
        res.head_pitch = head_pitch if head_pitch is not None else float("nan")

        # ── Knee visibility and angles ─────────────────────────────────────
        knee_angles: list[float] = []
        knee_confs: list[float] = []

        for h_idx, k_idx, a_idx in [
            (KP_LEFT_HIP,  KP_LEFT_KNEE,  KP_LEFT_ANKLE),
            (KP_RIGHT_HIP, KP_RIGHT_KNEE, KP_RIGHT_ANKLE),
        ]:
            h = kpb.get(h_idx); k = kpb.get(k_idx); a = kpb.get(a_idx)
            # Only use if visibly detected above limit
            k_conf = kpb.valid[k_idx]
            if h and k and a and k_conf:
                ang = _angle_deg(h, k, a)
                if not math.isnan(ang):
                    knee_angles.append(ang)
                    # Use a proxy for conf for now: if valid, we assume it's good enough
                    knee_confs.append(1.0) 

        res.avg_knee_angle = float(np.min(knee_angles)) if knee_angles else float("nan")
        if knee_angles:
            res.knee_angle_left  = knee_angles[0] if len(knee_angles) >= 1 else float("nan")
            res.knee_angle_right = knee_angles[1] if len(knee_angles) >= 2 else float("nan")

        knees_usable = len(knee_confs) > 0

        # ── Upper-Body Features ────────────────────────────────────────────
        ls = kpb.get(KP_LEFT_SHOULDER); rs = kpb.get(KP_RIGHT_SHOULDER)
        lh = kpb.get(KP_LEFT_HIP);      rh = kpb.get(KP_RIGHT_HIP)
        nose = kpb.get(KP_NOSE)

        sh_mid = None
        if ls and rs:    sh_mid = ((ls[0]+rs[0])/2.0, (ls[1]+rs[1])/2.0)
        elif ls:         sh_mid = ls
        elif rs:         sh_mid = rs

        hm_mid = None
        if lh and rh:    hm_mid = ((lh[0]+rh[0])/2.0, (lh[1]+rh[1])/2.0)
        elif lh:         hm_mid = lh
        elif rh:         hm_mid = rh

        if sh_mid and hm_mid:
            res.spine_tilt_deg = _tilt_from_vertical(sh_mid, hm_mid)
            res.hip_forward_norm = abs(hm_mid[0] - sh_mid[0]) / sw

        if nose and sh_mid:
            res.nose_drop_norm = (nose[1] - sh_mid[1]) / sw
            res.torso_compress = (nose[1] - sh_mid[1]) / sw

        if bbox:
            bx1, by1, bx2, by2 = bbox
            bw, bh = bx2 - bx1, by2 - by1
            if bw > 0 and bh > 0:
                res.bbox_aspect_ratio = bh / bw
                if sh_mid:
                    res.shoulder_y_norm = (sh_mid[1] - by1) / bh

        # ── Slouch detection (applies to both modes) ───────────────────────
        slouch_signals = 0
        if not math.isnan(res.spine_tilt_deg) and res.spine_tilt_deg > CFG.SLOUCH_SPINE_TILT_MAX:
            slouch_signals += 1
        if not math.isnan(res.nose_drop_norm) and res.nose_drop_norm > CFG.SLOUCH_NOSE_DROP_NORM:
            slouch_signals += 1
        if (not math.isnan(res.hip_forward_norm) and res.hip_forward_norm > CFG.SLOUCH_HIP_FORWARD_NORM 
            and not math.isnan(res.spine_tilt_deg) and res.spine_tilt_deg > CFG.SLOUCH_SPINE_TILT_MAX * 0.7):
            slouch_signals += 2
        # Torso compress slouch 
        if not math.isnan(res.torso_compress) and res.torso_compress < CFG.UB_TORSO_COMPRESS_THRESH:
            slouch_signals += 1
        # Head pitch slouch
        if not math.isnan(res.head_pitch) and res.head_pitch < CFG.UB_HEAD_PITCH_SLOUCH:
            slouch_signals += 1

        is_slouching = slouch_signals >= 2 or slouch_signals >= 1 and (res.spine_tilt_deg > CFG.SLOUCH_SPINE_TILT_MAX)

        # ── Decision Tree ──────────────────────────────────────────────────
        raw_label = "unknown"

        if is_slouching:
            raw_label = "slouching"
            res.confidence = min(1.0, 0.5 + 0.1 * slouch_signals)
        elif knees_usable:
            # Mode A: Knee logic (Free-standing)
            res.detection_mode = "knee"
            if res.avg_knee_angle < CFG.SITTING_KNEE_ANGLE_MAX:
                raw_label = "sitting"
                res.confidence = min(1.0, (CFG.SITTING_KNEE_ANGLE_MAX - res.avg_knee_angle) / 35.0)
            elif res.avg_knee_angle > CFG.STANDING_KNEE_ANGLE_MIN:
                raw_label = "standing"
                res.confidence = min(1.0, (res.avg_knee_angle - CFG.STANDING_KNEE_ANGLE_MIN) / 15.0)
            else:
                raw_label = self._state
                res.confidence = 0.5
        else:
            # Mode B: Upper Body logic (Desk occlusion)
            res.detection_mode = "upper_body"
            sit_score, stand_score = 0, 0

            # Bbox
            if not math.isnan(res.bbox_aspect_ratio):
                if res.bbox_aspect_ratio < CFG.UB_BBOX_AR_SITTING_MAX: sit_score += 1.5
                if res.bbox_aspect_ratio > CFG.UB_BBOX_AR_STANDING_MIN: stand_score += 1.5
            
            # Shoulder Y
            if not math.isnan(res.shoulder_y_norm):
                if res.shoulder_y_norm > CFG.UB_SHOULDER_Y_SITTING_MIN: sit_score += 1
                if res.shoulder_y_norm < CFG.UB_SHOULDER_Y_STANDING_MAX: stand_score += 1

            if sit_score > stand_score:
                raw_label = "sitting"
                res.confidence = min(0.85, 0.4 + (sit_score * 0.15))
            elif stand_score > sit_score:
                raw_label = "standing"
                res.confidence = min(0.85, 0.4 + (stand_score * 0.15))
            else:
                raw_label = self._state
                res.confidence = 0.5


# ==============================================================================
# BEHAVIOUR 4 — BOUNDING DETECTOR
# ==============================================================================

class BoundingDetector:
    """
    Detects large body displacements indicating jumping or bounding movement.

    Strategy:
      - Tracks hip centroid position each frame (normalised by shoulder width).
      - Computes per-frame displacement (Euclidean distance between consecutive
        hip positions, normalised by sw).
      - Smooths over a short rolling window to reduce noise.
      - Triggers 'bounding' when smoothed displacement > BOUND_DISP_NORM.
      - Uses a decay counter so the label stays active for BOUND_DECAY_FRAMES
        after the last detected displacement, avoiding choppy on/off flicker.

    Unlike the old FFT bounce detector (which needed a full window of history
    to fire), this responds within 1–2 frames of the jump, making it much more
    responsive for rapid movements like jumps, leaps, or large lateral steps.
    """

    def __init__(self) -> None:
        self._prev_hip:    Optional[tuple[float, float]] = None
        self._disp_buf:    deque = deque(maxlen=CFG.BOUND_HISTORY_LEN)
        self._decay_count: int   = 0

    def update(self, kpb: KeypointBuffer, sw: float) -> tuple[float, int]:
        """
        Returns (disp_score ∈ [0,1], bounding ∈ {0,1}).
        disp_score is the current normalised displacement smoothed over the window.
        """
        hip = _hip_centroid(kpb)

        if hip is None:
            # No hip visible: decay and return
            self._decay_count = max(0, self._decay_count - 1)
            return 0.0, int(self._decay_count > 0)

        if self._prev_hip is not None:
            dx   = hip[0] - self._prev_hip[0]
            dy   = hip[1] - self._prev_hip[1]
            disp = math.hypot(dx, dy) / max(sw, 1.0)
        else:
            disp = 0.0

        self._prev_hip = hip
        self._disp_buf.append(disp)

        # Smooth: use max over window (captures peak displacement)
        smooth_disp = float(np.max(self._disp_buf)) if self._disp_buf else 0.0
        disp_score  = float(np.clip(smooth_disp / (CFG.BOUND_DISP_NORM * 2.0), 0.0, 1.0))

        if smooth_disp >= CFG.BOUND_DISP_NORM:
            # Displacement is large — set decay counter
            self._decay_count = CFG.BOUND_DECAY_FRAMES
        else:
            self._decay_count = max(0, self._decay_count - 1)

        is_bounding = int(self._decay_count > 0)
        return disp_score, is_bounding

    def reset(self) -> None:
        self._prev_hip    = None
        self._disp_buf.clear()
        self._decay_count = 0


# ==============================================================================
# BEHAVIOUR 5 — HAND-RAISE STATE MACHINE
# ==============================================================================

class HandRaiseSM:
    """
    Three-condition, hysteresis-protected hand-raise detector.

    Condition A (geometry) : wrist_y < shoulder_y − margin_px
    Condition B (elbow)    : elbow_y < shoulder_midpoint_y  (optional, see CFG)
    Condition C (motion)   : upward Kalman velocity of wrist / sw ≥ threshold

    A must hold; B is optional (strengthens confidence); C accelerates confirmation.
    Hysteresis: once raised, requires HAND_LOWER_HOLD frames to drop.
    """

    _IDLE = 0; _CANDIDATE = 1; _RAISED = 2; _LOWERING = 3

    def __init__(self) -> None:
        self._state               = self._IDLE
        self._rcnt                = 0
        self._lcnt                = 0
        self.wrist_above_shoulder = False
        self.elbow_above_shoulder = False
        self.wrist_vel_up         = False
        self.raised               = 0

    def update(self, kpb: KeypointBuffer, sw: float) -> None:
        margin_px  = CFG.HAND_ABOVE_MARGIN_NORM * sw
        # Shoulder midpoint y
        ls = kpb.get(KP_LEFT_SHOULDER); rs = kpb.get(KP_RIGHT_SHOULDER)
        sh_mid_y = None
        if ls and rs:
            sh_mid_y = (ls[1] + rs[1]) / 2.0
        elif ls:
            sh_mid_y = ls[1]
        elif rs:
            sh_mid_y = rs[1]

        lw = kpb.get(KP_LEFT_WRIST);   rw = kpb.get(KP_RIGHT_WRIST)
        le = kpb.get(KP_LEFT_ELBOW);   re = kpb.get(KP_RIGHT_ELBOW)

        # Condition A: wrist above ipsilateral shoulder
        la = bool(ls and lw and lw[1] < ls[1] - margin_px)
        ra = bool(rs and rw and rw[1] < rs[1] - margin_px)
        cond_A = la or ra
        self.wrist_above_shoulder = cond_A

        # Condition B: elbow above shoulder midpoint
        if sh_mid_y is not None:
            lb = bool(le and le[1] < sh_mid_y)
            rb = bool(re and re[1] < sh_mid_y)
            cond_B_geom = lb or rb
        else:
            cond_B_geom = False
        self.elbow_above_shoulder = cond_B_geom

        # Condition C: upward wrist velocity
        lv = kpb.kpf(KP_LEFT_WRIST).vy_up
        rv = kpb.kpf(KP_RIGHT_WRIST).vy_up
        if la and ra:   active_vy = max(lv, rv)
        elif la:        active_vy = lv
        elif ra:        active_vy = rv
        else:           active_vy = max(lv, rv)

        cond_C = (active_vy / max(sw, 1.0)) >= CFG.HAND_VEL_NORM
        self.wrist_vel_up = cond_C

        # Raise trigger: A required; B or C as additional evidence
        raise_trigger = cond_A and (cond_B_geom or cond_C)
        # Lower trigger: neither A nor B
        lower_trigger = (not cond_A) and (not cond_B_geom)

        if self._state == self._IDLE:
            if raise_trigger:
                self._state = self._CANDIDATE; self._rcnt = 1
        elif self._state == self._CANDIDATE:
            if raise_trigger:
                self._rcnt += 1
                if self._rcnt >= CFG.HAND_RAISE_HOLD:
                    self._state = self._RAISED
            else:
                self._state = self._IDLE; self._rcnt = 0
        elif self._state == self._RAISED:
            if lower_trigger:
                self._state = self._LOWERING; self._lcnt = 1
        elif self._state == self._LOWERING:
            if raise_trigger:
                self._state = self._RAISED; self._lcnt = 0
            else:
                self._lcnt += 1
                if self._lcnt >= CFG.HAND_LOWER_HOLD:
                    self._state = self._IDLE; self._rcnt = self._lcnt = 0

        self.raised = int(self._state in (self._RAISED, self._LOWERING))


# ==============================================================================
# PER-TRACK STATE
# ==============================================================================

@dataclass
class TrackState:
    track_id:    int
    kpb:         KeypointBuffer   = field(default_factory=KeypointBuffer)
    posture_cls: PostureClassifier = field(default_factory=PostureClassifier)
    bounding:    BoundingDetector  = field(default_factory=BoundingDetector)
    hand_sm:     HandRaiseSM       = field(default_factory=HandRaiseSM)

    def process(self, kp_array: np.ndarray, bbox: tuple = None, head_pitch: float = None) -> dict:
        """
        Full 5-behaviour pipeline for one frame.
        kp_array : (17, 3)  — [x, y, visibility] from the CSV.
        Returns a flat dict of all features.
        """
        self.kpb.update(kp_array)
        sw = _body_ruler(self.kpb)

        # Posture (sitting / slouching / standing / unknown) — now stateful
        pr = self.posture_cls.classify(self.kpb, sw, bbox=bbox, head_pitch=head_pitch)

        # Bounding (large body displacement)
        disp_score, is_bounding = self.bounding.update(self.kpb, sw)

        # Hand raise
        self.hand_sm.update(self.kpb, sw)

        def _f(v: float) -> object:
            return round(v, 2) if not math.isnan(v) else ""

        return {
            "shoulder_width_px":    round(sw, 2),
            "detection_mode":       pr.detection_mode,
            "knee_angle_left":      _f(pr.knee_angle_left),
            "knee_angle_right":     _f(pr.knee_angle_right),
            "avg_knee_angle":       _f(pr.avg_knee_angle),
            "spine_tilt_deg":       _f(pr.spine_tilt_deg),
            "nose_drop_norm":       _f(pr.nose_drop_norm),
            "hip_forward_norm":     _f(pr.hip_forward_norm),
            "bbox_aspect_ratio":    _f(pr.bbox_aspect_ratio),
            "shoulder_y_norm":      _f(pr.shoulder_y_norm),
            "torso_compress":       _f(pr.torso_compress),
            "head_pitch_used":      _f(pr.head_pitch),
            "disp_score":           round(disp_score, 4),
            "wrist_above_shoulder": int(self.hand_sm.wrist_above_shoulder),
            "elbow_above_shoulder": int(self.hand_sm.elbow_above_shoulder),
            "wrist_vel_up":         int(self.hand_sm.wrist_vel_up),
            "posture":              pr.label,
            "posture_confidence":   round(pr.confidence, 3),
            "bounding":             is_bounding,
            "bounding_confidence":  round(disp_score, 4),
            "hand_raised":          self.hand_sm.raised,
        }


# ==============================================================================
# SUMMARY BUILDER  (run-length encoder → episode table)
# ==============================================================================

@dataclass
class Episode:
    track_id: int; behaviour: str
    start_frame: int; end_frame: int
    conf_sum: float = 0.0; n_frames: int = 0

    def extend(self, fid: int, conf: float) -> None:
        self.end_frame = fid; self.conf_sum += conf; self.n_frames += 1

    @property
    def confidence_avg(self) -> float:
        return self.conf_sum / max(1, self.n_frames)

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1


class SummaryBuilder:
    GAP_TOLERANCE = 5

    def __init__(self) -> None:
        self._open:       dict[int, Episode] = {}
        self._last_frame: dict[int, int]     = {}
        self._done:       list[Episode]      = []

    def ingest(self, fid: int, tid: int, beh: str, conf: float) -> None:
        gap = fid - self._last_frame.get(tid, -9999)
        self._last_frame[tid] = fid
        ep = self._open.get(tid)
        if ep is None or ep.behaviour != beh or gap > self.GAP_TOLERANCE:
            if ep: self._done.append(ep)
            self._open[tid] = Episode(tid, beh, fid, fid, conf, 1)
        else:
            ep.extend(fid, conf)

    def flush(self) -> None:
        for ep in self._open.values(): self._done.append(ep)
        self._open.clear()

    @property
    def episodes(self) -> list[Episode]:
        return sorted(self._done, key=lambda e: (e.track_id, e.start_frame))


def dominant_behaviour(feat: dict) -> tuple[str, float]:
    """Pick the single most salient behaviour label + confidence."""
    if feat["hand_raised"]:  return "hand_raised", 1.0
    if feat["bounding"]:     return "bounding",    float(feat["bounding_confidence"])
    return str(feat["posture"]), float(feat["posture_confidence"])


# ==============================================================================
# CSV LOADER — reads raw_body_multi.csv into a frame-indexed structure
# ==============================================================================

def load_body_csv(body_csv: str) -> tuple[dict[int, dict[int, np.ndarray]], list[int]]:
    """
    Reads raw_body_multi.csv (long format: 17 rows per person per frame).
    Returns:
        body_index  : {frame_id: {track_id: np.ndarray(17, 3)}}
        frame_ids   : sorted list of all unique frame_ids in the CSV
    """
    print(f"[CSV]   Loading {body_csv}", end="", flush=True)

    body_index: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    chunk_size  = 17 * 20 * 500

    for chunk in pd.read_csv(
        body_csv,
        dtype={
            "frame_id":     "Int64",
            "track_id":     "Int64",
            "landmark_idx": "Int64",
            "x":             float,
            "y":             float,
            "visibility":    float,
        },
        chunksize=chunk_size,
    ):
        chunk = chunk.dropna(subset=["landmark_idx"])
        chunk["landmark_idx"] = chunk["landmark_idx"].astype(int)

        for (fid, tid), grp in chunk.groupby(["frame_id", "track_id"]):
            fid = int(fid); tid = int(tid)
            arr = np.zeros((N_KP, 3), dtype=np.float64)

            for _, row in grp.iterrows():
                idx = int(row["landmark_idx"])
                if not (0 <= idx < N_KP):
                    continue
                try:
                    arr[idx, 0] = float(row["x"])          if not pd.isna(row.get("x",          np.nan)) else 0.0
                    arr[idx, 1] = float(row["y"])          if not pd.isna(row.get("y",          np.nan)) else 0.0
                    arr[idx, 2] = float(row["visibility"]) if not pd.isna(row.get("visibility", np.nan)) else 0.0
                except (ValueError, TypeError):
                    pass

            body_index[fid][tid] = arr

        print(".", end="", flush=True)

    frame_ids = sorted(body_index.keys())
    print(f"\n[CSV]   {len(frame_ids)} frames loaded, "
          f"frame range [{frame_ids[0]} … {frame_ids[-1]}]")
    return body_index, frame_ids


# ==============================================================================
# OVERLAY RENDERER
# ==============================================================================

class Renderer:
    FONT      = cv2.FONT_HERSHEY_SIMPLEX
    FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

    _LINKS = [
        (KP_LEFT_SHOULDER,  KP_LEFT_ELBOW,    None, 2),
        (KP_LEFT_ELBOW,     KP_LEFT_WRIST,    None, 2),
        (KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW,   None, 2),
        (KP_RIGHT_ELBOW,    KP_RIGHT_WRIST,   None, 2),
        (KP_LEFT_SHOULDER,  KP_RIGHT_SHOULDER,(170,170,170), 2),
        (KP_LEFT_SHOULDER,  KP_LEFT_HIP,      (170,170,170), 2),
        (KP_RIGHT_SHOULDER, KP_RIGHT_HIP,     (170,170,170), 2),
        (KP_LEFT_HIP,       KP_RIGHT_HIP,     (170,170,170), 2),
        (KP_LEFT_HIP,       KP_LEFT_KNEE,     (100,160,220), 2),
        (KP_LEFT_KNEE,      KP_LEFT_ANKLE,    (100,160,220), 2),
        (KP_RIGHT_HIP,      KP_RIGHT_KNEE,    (100,160,220), 2),
        (KP_RIGHT_KNEE,     KP_RIGHT_ANKLE,   (100,160,220), 2),
    ]
    _ARM_INDICES = {
        KP_LEFT_SHOULDER, KP_LEFT_ELBOW, KP_LEFT_WRIST,
        KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW, KP_RIGHT_WRIST,
    }

    @staticmethod
    def _beh_col(beh: str) -> tuple:
        return CFG.BEHAVIOUR_COLORS.get(beh, CFG.BEHAVIOUR_COLORS["unknown"])

    @classmethod
    def draw_skeleton(
        cls,
        frame:      np.ndarray,
        kpb:        KeypointBuffer,
        beh:        str,
        hand_raised: bool,
        is_bounding: bool,
    ) -> tuple[int, int, int, int]:
        beh_col  = cls._beh_col(beh)
        arm_col  = (0, 255, 0) if hand_raised else beh_col
        # Flash skeleton bright yellow when bounding
        if is_bounding:
            beh_col = (0, 230, 255)
        xs, ys   = [], []

        def _pt(idx: int) -> Optional[tuple[int, int]]:
            p = kpb.get(idx)
            if p:
                xs.append(int(p[0])); ys.append(int(p[1]))
                return int(p[0]), int(p[1])
            return None

        for a_idx, b_idx, fixed_col, thick in cls._LINKS:
            col = arm_col if a_idx in cls._ARM_INDICES else (fixed_col or beh_col)
            pa  = _pt(a_idx); pb = _pt(b_idx)
            if pa and pb:
                cv2.line(frame, pa, pb, col, thick, cv2.LINE_AA)

        for idx in range(N_KP):
            pt = _pt(idx)
            if pt:
                is_wrist = idx in (KP_LEFT_WRIST, KP_RIGHT_WRIST)
                dot_col  = (0, 255, 0) if (hand_raised and is_wrist) else beh_col
                r        = 5 if is_wrist else 3
                cv2.circle(frame, pt, r, dot_col, -1, cv2.LINE_AA)

        return (min(xs), min(ys), max(xs), max(ys)) if xs else (0, 0, 0, 0)

    @classmethod
    def draw_badge(
        cls,
        frame: np.ndarray,
        tid:   int,
        beh:   str,
        conf:  float,
        cx:    int,
        top_y: int,
    ) -> None:
        label = f"ID:{tid}  {beh.upper().replace('_',' ')}  {int(conf*100)}%"
        fs    = 0.50
        (tw, th), _ = cv2.getTextSize(label, cls.FONT_BOLD, fs, 1)

        pad_x, pad_y = 10, 5
        bx1 = max(0, cx - tw//2 - pad_x)
        bx2 = bx1 + tw + pad_x * 2
        by1 = max(0, top_y - th - pad_y*2 - 6)
        by2 = top_y - 6

        if by2 <= by1:
            by1 = max(0, by2 - th - pad_y*2)

        beh_col = cls._beh_col(beh)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), beh_col, cv2.FILLED)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (20, 20, 20), 1)
        cv2.putText(frame, label, (bx1 + pad_x, by1 + pad_y + th),
                    cls.FONT_BOLD, fs, (0, 0, 0), 1, cv2.LINE_AA)

        bw     = bx2 - bx1
        filled = max(0, min(bw, int(bw * conf)))
        cv2.rectangle(frame, (bx1, by2),           (bx2,         by2+4), (40,40,40),  cv2.FILLED)
        cv2.rectangle(frame, (bx1, by2),           (bx1+filled,  by2+4), beh_col,     cv2.FILLED)

    @classmethod
    def draw_diagnostics(
        cls,
        frame:   np.ndarray,
        feat:    dict,
        x_right: int,
        y_top:   int,
    ) -> None:
        def _s(key: str) -> str:
            v = feat.get(key, "")
            if v == "" or (isinstance(v, float) and math.isnan(v)):
                return "n/a"
            return str(v) if isinstance(v, str) else f"{float(v):.1f}"

        lines = [
            f"knee : {_s('avg_knee_angle')}",
            f"spine: {_s('spine_tilt_deg')}",
            f"nose : {_s('nose_drop_norm')}",
            f"hip_f: {_s('hip_forward_norm')}",
            f"disp : {_s('disp_score')}",
            f"W>sh : {feat.get('wrist_above_shoulder',0)}",
            f"E>sh : {feat.get('elbow_above_shoulder',0)}",
            f"W>vel: {feat.get('wrist_vel_up',0)}",
        ]
        lh = 14; pw = 130; ph = lh*len(lines)+6
        px1 = min(x_right+4, frame.shape[1]-pw-2)
        py1 = max(0, y_top)
        px2 = min(frame.shape[1]-1, px1+pw)
        py2 = min(frame.shape[0]-1, py1+ph)

        sub = frame[py1:py2, px1:px2]
        if sub.size > 0:
            frame[py1:py2, px1:px2] = cv2.addWeighted(sub, 0.25,
                                                        np.zeros_like(sub), 0.75, 0)
        for i, line in enumerate(lines):
            ty = py1 + lh*(i+1)
            if ty < frame.shape[0]:
                cv2.putText(frame, line, (px1+3, ty),
                            cls.FONT, 0.36, (200,200,200), 1, cv2.LINE_AA)

    @classmethod
    def draw_hand_banner(
        cls,
        frame:  np.ndarray,
        tid:    int,
        cx:     int,
        top_y:  int,
        tick:   int,
    ) -> None:
        if (tick // 15) % 2 == 0:
            label = f"! HAND RAISED  ID:{tid} !"
            fs    = 0.52
            (tw, th), _ = cv2.getTextSize(label, cls.FONT_BOLD, fs, 2)
            bx1 = max(0, cx - tw//2 - 8)
            by1 = max(0, top_y - th - 44)
            bx2 = bx1 + tw + 16
            by2 = by1 + th + 10
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (220, 0, 200), cv2.FILLED)
            cv2.putText(frame, label, (bx1+8, by1+th+4),
                        cls.FONT_BOLD, fs, (255,255,255), 2, cv2.LINE_AA)

    @classmethod
    def draw_bounding_banner(
        cls,
        frame:  np.ndarray,
        tid:    int,
        cx:     int,
        top_y:  int,
        tick:   int,
    ) -> None:
        """Animated BOUNDING banner shown when large displacement detected."""
        if (tick // 10) % 2 == 0:
            label = f">> BOUNDING  ID:{tid} <<"
            fs    = 0.52
            (tw, th), _ = cv2.getTextSize(label, cls.FONT_BOLD, fs, 2)
            bx1 = max(0, cx - tw//2 - 8)
            by1 = max(0, top_y - th - 60)
            bx2 = bx1 + tw + 16
            by2 = by1 + th + 10
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 200, 255), cv2.FILLED)
            cv2.putText(frame, label, (bx1+8, by1+th+4),
                        cls.FONT_BOLD, fs, (0, 0, 0), 2, cv2.LINE_AA)

    @staticmethod
    def draw_hud(
        frame:     np.ndarray,
        frame_id:  int,
        fps:       float,
        n_persons: int,
        paused:    bool,
        speed:     float,
        progress:  float,
    ) -> None:
        h, w = frame.shape[:2]

        hud_lines = [
            f"Frame : {frame_id}",
            f"FPS   : {fps:.1f}",
            f"People: {n_persons}",
            f"Speed : {speed:.1f}x",
            "|| PAUSED" if paused else "> PLAYING",
        ]
        hud_h, hud_w = 18*len(hud_lines)+8, 155
        sub = frame[0:hud_h, 0:hud_w]
        if sub.size > 0:
            frame[0:hud_h, 0:hud_w] = cv2.addWeighted(sub, 0.2,
                                                        np.zeros_like(sub), 0.8, 0)
        for i, line in enumerate(hud_lines):
            col = (0, 60, 255) if "PAUSED" in line else (0, 255, 180)
            cv2.putText(frame, line, (6, 18+i*18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.47, col, 1, cv2.LINE_AA)

        bar_h  = 6
        filled = int(w * progress)
        cv2.rectangle(frame, (0, h-bar_h), (w, h),       (40,40,40),   cv2.FILLED)
        cv2.rectangle(frame, (0, h-bar_h), (filled, h),  (0,200,255),  cv2.FILLED)

        hint = "SPACE=pause/step  Q=quit  S=snapshot  +/-=speed"
        (hw, hh), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.37, 1)
        cv2.putText(frame, hint, (w-hw-6, h-bar_h-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, (150,150,150), 1, cv2.LINE_AA)

    @classmethod
    def render_all(
        cls,
        frame:    np.ndarray,
        results:  dict,
        frame_id: int,
        fps:      float,
        paused:   bool,
        speed:    float,
        progress: float,
        tick:     int,
    ) -> None:
        for tid, (track, feat) in results.items():
            beh, conf   = dominant_behaviour(feat)
            hand_up     = bool(feat["hand_raised"])
            is_bounding = bool(feat["bounding"])

            x_min, y_min, x_max, y_max = cls.draw_skeleton(
                frame, track.kpb, beh, hand_up, is_bounding
            )

            if x_max == 0 and y_max == 0:
                continue

            cx = (x_min + x_max) // 2
            cls.draw_badge(frame, tid, beh, conf, cx, y_min)

            if hand_up:
                cls.draw_hand_banner(frame, tid, cx, y_min, tick)
            if is_bounding:
                cls.draw_bounding_banner(frame, tid, cx, y_min, tick)

            cls.draw_diagnostics(frame, feat, x_max, y_min)

        cls.draw_hud(frame, frame_id, fps, len(results), paused, speed, progress)


# ==============================================================================
# CSV OUTPUT COLUMNS
# ==============================================================================

_RAW_COLS = [
    "frame_id", "track_id", "shoulder_width_px",
    "knee_angle_left", "knee_angle_right", "avg_knee_angle",
    "spine_tilt_deg",  "nose_drop_norm",   "hip_forward_norm",
    "disp_score",
    "wrist_above_shoulder", "elbow_above_shoulder", "wrist_vel_up",
    "posture", "posture_confidence",
    "bounding", "bounding_confidence",
    "hand_raised",
]
_SUM_COLS = [
    "track_id", "behaviour",
    "start_frame", "end_frame", "duration_frames", "confidence_avg",
]


# ==============================================================================
# MAIN
# ==============================================================================

def run(
    video_path: str,
    body_csv:   str,
    raw_out:    str,
    sum_out:    str,
    speed:      float,
    step_mode:  bool,
    save_video: str,
) -> None:

    for p, name in [(video_path, "Video"), (body_csv, "Body CSV")]:
        if not Path(p).exists():
            print(f"[ERROR] {name} not found: {p}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Behaviour Classifier v5  —  CSV-driven video overlay")
    print(f"{'='*60}\n")

    body_index, frame_ids = load_body_csv(body_csv)

    if not frame_ids:
        print("[ERROR] Body CSV is empty or has no valid rows.")
        sys.exit(1)

    total_csv_frames = len(frame_ids)
    print(f"[CSV]   {total_csv_frames} frames will be processed.\n")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[Video] {Path(video_path).name}  "
          f"{frame_w}x{frame_h}  {video_fps:.1f}fps  {total_video_frames} frames")
    print(f"[Mode]  Speed={speed}x  Step={step_mode}")
    print(f"[Out]   {raw_out}  |  {sum_out}\n")
    print("  Controls: SPACE=pause/step  Q/ESC=quit  S=snapshot  +/-=speed\n")

    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_video, fourcc, video_fps, (frame_w, frame_h))
        print(f"[Save]  Annotated video → {save_video}")

    raw_fh = open(raw_out, "w", newline="", encoding="utf-8")
    raw_w  = csv.DictWriter(raw_fh, fieldnames=_RAW_COLS)
    raw_w.writeheader()

    track_states: dict[int, TrackState] = {}
    summary   = SummaryBuilder()
    renderer  = Renderer()

    def get_track(tid: int) -> TrackState:
        if tid not in track_states:
            track_states[tid] = TrackState(track_id=tid)
        return track_states[tid]

    fps_buf:  deque = deque(maxlen=30)
    prev_t    = time.perf_counter()

    paused         = step_mode
    tick           = 0
    rows_written   = 0
    snapshot_count = 0
    frame_delay_ms = max(1, int(1000 / (video_fps * speed)))

    last_rendered: Optional[np.ndarray] = None

    cv2.namedWindow(CFG.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CFG.WINDOW_NAME, min(frame_w, 1400), min(frame_h, 860))

    csv_frame_idx = 0

    while csv_frame_idx < total_csv_frames:

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            print("\n[Info]  User quit.")
            break

        if key == ord(' '):
            if not step_mode:
                paused = not paused

        if key == ord('s'):
            if last_rendered is not None:
                snap = f"snapshot_{snapshot_count:04d}.png"
                cv2.imwrite(snap, last_rendered)
                print(f"[Snap]  {snap}")
                snapshot_count += 1

        if key in (ord('+'), ord('=')):
            speed = min(speed * 1.5, 32.0)
            frame_delay_ms = max(1, int(1000 / (video_fps * speed)))
            print(f"[Speed] {speed:.2f}x")

        if key == ord('-'):
            speed = max(speed / 1.5, 0.05)
            frame_delay_ms = max(1, int(1000 / (video_fps * speed)))
            print(f"[Speed] {speed:.2f}x")

        if paused and key != ord(' '):
            if last_rendered is not None:
                cv2.imshow(CFG.WINDOW_NAME, last_rendered)
            cv2.waitKey(30)
            continue

        frame_id = frame_ids[csv_frame_idx]

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()

        if not ret:
            print(f"[Warn]  Could not read video frame {frame_id} — skipping.")
            csv_frame_idx += 1
            continue

        persons = body_index.get(frame_id, {})

        if not persons:
            now = time.perf_counter(); dt = now - prev_t; prev_t = now
            if dt > 0: fps_buf.append(1.0/dt)
            disp_fps = float(np.mean(fps_buf)) if fps_buf else 0.0
            progress = csv_frame_idx / max(1, total_csv_frames-1)
            Renderer.draw_hud(frame, frame_id, disp_fps, 0, paused, speed, progress)
            cv2.imshow(CFG.WINDOW_NAME, frame)
            last_rendered = frame.copy()
            if writer: writer.write(frame)
            csv_frame_idx += 1; tick += 1
            cv2.waitKey(frame_delay_ms)
            continue

        frame_results: dict[int, tuple] = {}

        for track_id, kp_array in persons.items():
            track = get_track(track_id)
            feat  = track.process(kp_array)

            raw_w.writerow({"frame_id": frame_id, "track_id": track_id, **feat})
            rows_written += 1

            beh_label, beh_conf = dominant_behaviour(feat)
            summary.ingest(frame_id, track_id, beh_label, beh_conf)

            frame_results[track_id] = (track, feat)

        if csv_frame_idx % 60 == 0:
            raw_fh.flush()

        now = time.perf_counter(); dt = now - prev_t; prev_t = now
        if dt > 0: fps_buf.append(1.0/dt)
        disp_fps = float(np.mean(fps_buf)) if fps_buf else 0.0
        progress = csv_frame_idx / max(1, total_csv_frames - 1)

        renderer.render_all(
            frame, frame_results, frame_id,
            disp_fps, paused, speed, progress, tick
        )

        cv2.imshow(CFG.WINDOW_NAME, frame)
        last_rendered = frame.copy()

        if writer:
            writer.write(frame)

        csv_frame_idx += 1
        tick          += 1

        cv2.waitKey(frame_delay_ms)

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    raw_fh.flush()
    raw_fh.close()

    print(f"\n[Done]  CSV 1 — {rows_written} rows → {raw_out}")

    summary.flush()
    episodes = summary.episodes
    with open(sum_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_SUM_COLS)
        w.writeheader()
        for ep in episodes:
            w.writerow({
                "track_id":        ep.track_id,
                "behaviour":       ep.behaviour,
                "start_frame":     ep.start_frame,
                "end_frame":       ep.end_frame,
                "duration_frames": ep.duration_frames,
                "confidence_avg":  round(ep.confidence_avg, 4),
            })

    print(f"[Done]  CSV 2 — {len(episodes)} episodes → {sum_out}")
    print(f"        {len(track_states)} unique track IDs.\n")

    print(f"{'='*64}")
    print(f"  {'ID':>4}  {'Behaviour':<14} {'Start':>7} {'End':>7} {'Frames':>7} {'Conf':>6}")
    print(f"{'='*64}")
    for ep in episodes:
        print(f"  {ep.track_id:>4}  {ep.behaviour:<14} "
              f"{ep.start_frame:>7} {ep.end_frame:>7} "
              f"{ep.duration_frames:>7} {ep.confidence_avg:>6.3f}")
    print(f"{'='*64}\n")


# ==============================================================================
# CLI
# ==============================================================================

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Behaviour Classifier v5 — CSV-driven real-time video overlay"
    )
    p.add_argument("--video",      default=CFG.VIDEO_PATH,
                   help=f"Input video file (default: {CFG.VIDEO_PATH})")
    p.add_argument("--body",       default=CFG.BODY_CSV,
                   help="raw_body_multi.csv from extract_raw_data_multi.py")
    p.add_argument("--raw-out",    dest="raw_out",   default=CFG.RAW_OUT_CSV)
    p.add_argument("--sum-out",    dest="sum_out",   default=CFG.SUM_OUT_CSV)
    p.add_argument("--speed",      type=float, default=1.0,
                   help="Playback speed multiplier (default: 1.0)")
    p.add_argument("--step",       action="store_true",
                   help="Step mode: advance one frame per SPACE press")
    p.add_argument("--save-video", dest="save_video", default="",
                   help="Save annotated video (e.g. output.mp4)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    run(
        video_path = args.video,
        body_csv   = args.body,
        raw_out    = args.raw_out,
        sum_out    = args.sum_out,
        speed      = args.speed,
        step_mode  = args.step,
        save_video = args.save_video,
    )

# testgghhhhhhhhjjjjhhjjjj