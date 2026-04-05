"""  
=============================================================================
body_gestures_labeling_hybrid.py  —  v5 (Headless)
=============================================================================
WHAT THIS SCRIPT DOES
---------------------
1. Reads raw_body_multi.csv (produced by extract_raw_data_multi.py).
   This CSV contains one row per keypoint per person per frame:
       frame_id | track_id | landmark_idx | x | y | visibility | bbox_x1..y2

2. Optionally reads labeled_head_pose_multi.csv for head pitch data.

3. Groups all rows by frame_id, rebuilds a (17, 3) keypoint array for
   each (frame_id, track_id) pair.

4. Runs the hybrid 5-behaviour pipeline on the keypoints:
       sitting | slouching | standing | bounding | hand_raised
   Uses knee-angle detection when lower body is visible, and falls
   back to upper-body metrics (bbox aspect ratio, shoulder position,
   torso compression, head pitch) when knees are occluded.

5. Writes two output CSVs:
       behaviour_raw_frames.csv  — one row per (frame, person)
       behaviour_summary.csv     — one row per behaviour episode

USAGE
-----
  python body_gestures_labeling_hybrid.py
  python body_gestures_labeling_hybrid.py --body raw_body_multi.csv
  python body_gestures_labeling_hybrid.py --head-pose labeled_head_pose_multi.csv

DEPENDENCIES
------------
  pip install numpy pandas
=============================================================================
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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
    BODY_CSV:    str = "raw_body_multi.csv"
    RAW_OUT_CSV: str = "behaviour_raw_frames.csv"
    SUM_OUT_CSV: str = "behaviour_summary.csv"

    # Keypoint confidence gate
    KP_CONF_MIN:   float = 0.25
    KP_INTERP_MAX: int   = 10

    # Kalman filter noise
    KP_KALMAN_Q: float = 2e-4
    KP_KALMAN_R: float = 6e-3

    # Body ruler fallback when both shoulders invisible
    FALLBACK_SW: float = 100.0

    # ── Behaviour 1 & 3 : Sitting / Standing (with hysteresis) ────────────
    SITTING_KNEE_ANGLE_MAX:   float = 140.0
    SITTING_KNEE_ANGLE_EXIT:  float = 150.0
    STANDING_KNEE_ANGLE_MIN:  float = 158.0
    STANDING_KNEE_ANGLE_EXIT: float = 148.0
    POSTURE_CONFIRM_FRAMES:   int   = 4

    # ── Upper-body fallback (desk-occluded lower body) ─────────────────
    KNEE_VIS_MIN: float = 0.30
    UB_BBOX_AR_SITTING_MAX:  float = 1.6
    UB_BBOX_AR_STANDING_MIN: float = 2.2
    UB_SHOULDER_Y_SITTING_MIN: float = 0.30
    UB_SHOULDER_Y_STANDING_MAX: float = 0.22
    UB_TORSO_COMPRESS_THRESH: float = 0.08
    UB_HEAD_PITCH_SLOUCH: float = -18.0

    # ── Behaviour 2 : Slouching ────────────────────────────────────────────
    SLOUCH_SPINE_TILT_MAX:     float = 12.0
    SLOUCH_NOSE_DROP_NORM:     float = 0.15
    SLOUCH_HIP_FORWARD_NORM:   float = 0.10

    # ── Behaviour 4 : Bounding (large body displacement) ──────────────────
    BOUND_DISP_NORM:      float = 0.25
    BOUND_CONFIRM_FRAMES: int   = 3
    BOUND_DECAY_FRAMES:   int   = 8
    BOUND_HISTORY_LEN:    int   = 5

    # ── Behaviour 5 : Hand raise ───────────────────────────────────────────
    HAND_ABOVE_MARGIN_NORM: float = 0.08
    ELBOW_ABOVE_SHOULDER:   bool  = True
    HAND_VEL_NORM:          float = 0.06
    HAND_RAISE_HOLD:        int   = 6
    HAND_LOWER_HOLD:        int   = 15


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

        # ── Hysteresis state machine ──────────────────────────────────────
        if raw_label == self._state:
            self._hold_counter = 0
            new_state = self._state
        else:
            if raw_label == self._candidate:
                self._hold_counter += 1
            else:
                self._candidate    = raw_label
                self._hold_counter = 1

            if self._hold_counter >= CFG.POSTURE_CONFIRM_FRAMES:
                self._state        = raw_label
                self._hold_counter = 0
            new_state = self._state

        res.label = new_state
        return res

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

def load_body_csv(body_csv: str) -> tuple[dict[int, dict[int, np.ndarray]], dict[int, dict[int, tuple]], list[int]]:
    """
    Reads raw_body_multi.csv (long format: 17 rows per person per frame).
    Returns:
        body_index  : {frame_id: {track_id: np.ndarray(17, 3)}}
        bbox_index  : {frame_id: {track_id: (x1, y1, x2, y2)}}
        frame_ids   : sorted list of all unique frame_ids in the CSV
    """
    print(f"[CSV]   Loading {body_csv}", end="", flush=True)

    body_index: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    bbox_index: dict[int, dict[int, tuple]] = defaultdict(dict)
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
            "bbox_x1":       float,
            "bbox_y1":       float,
            "bbox_x2":       float,
            "bbox_y2":       float,
        },
        chunksize=chunk_size,
    ):
        chunk = chunk.dropna(subset=["landmark_idx"])
        chunk["landmark_idx"] = chunk["landmark_idx"].astype(int)

        for (fid, tid), grp in chunk.groupby(["frame_id", "track_id"]):
            fid = int(fid); tid = int(tid)
            arr = np.zeros((N_KP, 3), dtype=np.float64)

            # Bbox
            bx1 = grp["bbox_x1"].iloc[0]
            if not pd.isna(bx1):
                bbox_index[fid][tid] = (
                    float(bx1),
                    float(grp["bbox_y1"].iloc[0]),
                    float(grp["bbox_x2"].iloc[0]),
                    float(grp["bbox_y2"].iloc[0])
                )

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
    return body_index, bbox_index, frame_ids



# ==============================================================================
# CSV OUTPUT COLUMNS
# ==============================================================================

_RAW_COLS = [
    "frame_id", "track_id", "shoulder_width_px",
    "detection_mode",
    "knee_angle_left", "knee_angle_right", "avg_knee_angle",
    "spine_tilt_deg",  "nose_drop_norm",   "hip_forward_norm",
    "bbox_aspect_ratio", "shoulder_y_norm", "torso_compress", "head_pitch_used",
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
# HEAD_POSE LOADER
# ==============================================================================


def load_head_pose_csv(head_pose_csv: str) -> dict:
    hp_index = {}
    if not os.path.exists(head_pose_csv):
        return hp_index

    df = pd.read_csv(head_pose_csv)
    if "pitch_smooth" not in df.columns:
        return hp_index
        

    for _, row in df.iterrows():
        try:
            fid = int(row["frame_id"])
            tid = int(row["track_id"])
            if fid not in hp_index:
                hp_index[fid] = {}
            if not math.isnan(row["pitch_smooth"]):
                hp_index[fid][tid] = float(row["pitch_smooth"])
        except ValueError:
            pass
    return hp_index


# ==============================================================================
# MAIN
# ==============================================================================


def run(
    body_csv:   str,
    hp_csv:     str,
    raw_out:    str,
    sum_out:    str,
) -> None:

    if not Path(body_csv).exists():
        print(f"[ERROR] Body CSV not found: {body_csv}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Behaviour Classifier v5  —  Headless Pipeline (Hybrid)")
    print(f"{'='*60}\n")

    body_index, bbox_index, frame_ids = load_body_csv(body_csv)
    head_pose_index = load_head_pose_csv(hp_csv)

    if not frame_ids:
        print("[ERROR] Body CSV is empty or has no valid rows.")
        sys.exit(1)

    total_csv_frames = len(frame_ids)
    print(f"[CSV]   {total_csv_frames} frames will be processed.\n")
    print(f"[Out]   {raw_out}  |  {sum_out}\n")

    raw_fh = open(raw_out, "w", newline="", encoding="utf-8")
    raw_w  = csv.DictWriter(raw_fh, fieldnames=_RAW_COLS)
    raw_w.writeheader()

    track_states: dict = {}
    summary   = SummaryBuilder()

    def get_track(tid: int) -> TrackState:
        if tid not in track_states:
            track_states[tid] = TrackState(track_id=tid)
        return track_states[tid]

    rows_written   = 0

    st_time = time.time()
    for frame_id in frame_ids:
        persons = body_index.get(frame_id, {})
        if not persons:
            continue

        for track_id, kp_array in persons.items():
            track = get_track(track_id)
            bbox = bbox_index.get(frame_id, {}).get(track_id, None)
            pitch = head_pose_index.get(frame_id, {}).get(track_id, None)

            feat  = track.process(kp_array, bbox=bbox, head_pitch=pitch)

            raw_w.writerow({"frame_id": frame_id, "track_id": track_id, **feat})
            rows_written += 1

            beh_label, beh_conf = dominant_behaviour(feat)
            summary.ingest(frame_id, track_id, beh_label, beh_conf)

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
    print(f"Total processing time: {time.time() - st_time:.2f}s")


# ==============================================================================
# CLI
# ==============================================================================



def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Behaviour Classifier v5 — Headless Hybrid Posture Pipeline"
    )
    p.add_argument("--body",       default=CFG.BODY_CSV,
                   help="raw_body_multi.csv from extract_raw_data_multi.py")
    p.add_argument("--head-pose",  default="labeled_head_pose_multi.csv",
                   help="labeled_head_pose_multi.csv from label_head_pose.py")
    p.add_argument("--raw-out",    dest="raw_out",   default=CFG.RAW_OUT_CSV)
    p.add_argument("--sum-out",    dest="sum_out",   default=CFG.SUM_OUT_CSV)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    run(
        body_csv   = args.body,
        hp_csv     = args.head_pose,
        raw_out    = args.raw_out,
        sum_out    = args.sum_out,
    )
