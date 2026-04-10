"""
=============================================================================
behaviour_classifier_visual.py  —  v4
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
       sitting | slouching | standing | bouncing | hand_raised

5. Draws the results onto the video frame and displays in an OpenCV window.

6. Writes two output CSVs:
       behaviour_raw_frames.csv  — one row per (frame, person)
       behaviour_summary.csv     — one row per behaviour episode

ROOT CAUSE OF THE PREVIOUS BUG
--------------------------------
The old code incremented frame_id by 1 every iteration of the video loop,
but raw_body_multi.csv uses the frame counter from extract_raw_data_multi.py
which may skip frames (FRAME_STRIDE > 1) or use a different numbering.
This caused body_index.get(frame_id) to always return {} — empty — so
the behaviour engine received no keypoints and produced nothing.

FIX: This version is 100% CSV-driven. We iterate over frame_id values
taken directly from the CSV (sorted), seek the video to each one with
cap.set(CAP_PROP_POS_FRAMES, frame_id), and process exactly those frames.

CONTROLS
--------
  SPACE      pause / resume  (or advance one frame when paused)
  Q / ESC    quit
  S          save current frame as PNG snapshot
  + / -      increase / decrease playback speed

USAGE
-----
  python behaviour_classifier_visual.py \
      --video mardi.mov \
      --body  raw_body_multi.csv

  # Step through frame by frame (useful for debugging):
  python behaviour_classifier_visual.py --video mardi.mov --body raw_body_multi.csv --step

  # Save annotated video:
  python behaviour_classifier_visual.py --video mardi.mov --body raw_body_multi.csv \
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
    VIDEO_PATH:  str = "aya1.mov"
    BODY_CSV:    str = "raw_body_multi.csv"
    RAW_OUT_CSV: str = "behaviour_raw_frames.csv"
    SUM_OUT_CSV: str = "behaviour_summary.csv"

    # Keypoint confidence gate
    KP_CONF_MIN:   float = 0.30
    KP_INTERP_MAX: int   = 6       # max frames to extrapolate a missing keypoint

    # Kalman filter noise
    KP_KALMAN_Q: float = 3e-4     # process noise (higher = faster response)
    KP_KALMAN_R: float = 8e-3     # measurement noise (higher = smoother)

    # Body ruler fallback when both shoulders invisible
    FALLBACK_SW: float = 100.0    # pixels

    # ══════════════════════════════════════════════════════════════════════
    # CLASSROOM-SPECIFIC DETECTION — no knees required
    # All pixel distances are normalised by shoulder_width (body ruler)
    # so thresholds are camera-distance independent.
    # ══════════════════════════════════════════════════════════════════════

    # ── Desk detection (automatic, per-track) ─────────────────────────────
    # The desk surface Y is estimated as the median of the wrist Y positions
    # over a rolling window.  In a classroom, students rest their wrists/arms
    # on the desk most of the time, so the median is a robust desk estimate.
    DESK_ESTIMATE_WIN:     int   = 60    # frames — rolling window for wrist median
    DESK_MIN_SAMPLES:      int   = 20    # min wrist observations before trusting desk Y

    # ── Behaviour 1 : Sitting ─────────────────────────────────────────────
    SITTING_SHOULDER_ABOVE_DESK_NORM: float = 0.20

    # ── Behaviour 2 : Slouching  (score-based) ────────────────────────────
    # slouch_score = weighted combination of three signals.
    # Weights (should sum to 1.0):
    SLOUCH_W_HEAD_DESK:      float = 0.55   # MOST IMPORTANT: head-to-desk gap
    SLOUCH_W_SPINE_TILT:     float = 0.25   # nose→shoulder forward tilt angle
    SLOUCH_W_FORWARD_SHIFT:  float = 0.20   # horizontal forward head displacement

    # Head-desk gap normalised thresholds:
    #   slouch_score contribution from head_desk is:
    #     0.0  when head_desk_norm >= SLOUCH_HEAD_CLEAR   (clearly upright)
    #     1.0  when head_desk_norm <= SLOUCH_HEAD_FLOOR   (head on/at desk)
    SLOUCH_HEAD_CLEAR:  float = 1.20   # sw above desk → no slouch contribution
    SLOUCH_HEAD_FLOOR:  float = 0.35   # sw above desk → full slouch contribution

    # Spine-tilt normalised thresholds:
    SLOUCH_TILT_CLEAR:  float = 10.0   # deg — no contribution below this
    SLOUCH_TILT_MAX:    float = 45.0   # deg — full contribution above this

    # Forward head shift: (nose_x - shoulder_mid_x) / sw
    # Positive = nose is FORWARD (in front of shoulders, i.e. leaning)
    # Normalised so that FORWARD_SHIFT_MAX = full contribution.
    # Note: sign depends on camera side; we use absolute value.
    SLOUCH_FORWARD_SHIFT_CLEAR: float = 0.10  # norm — no contribution below
    SLOUCH_FORWARD_SHIFT_MAX:   float = 0.45  # norm — full contribution above

    # Hysteresis thresholds on the combined slouch_score
    SLOUCH_SCORE_HIGH: float = 0.40   # score > HIGH → slouching
    SLOUCH_SCORE_LOW:  float = 0.20   # score < LOW  → not slouching
    SLOUCH_SCORE_EMA:  float = 0.30   # EMA smoothing alpha

    # ── Behaviour 3 : Standing (score-based) ─────────────────────────────
    STANDING_SHOULDER_ABOVE_DESK_NORM: float = 1.20

    # Score weights — geometry-first for static standing detection.
    # Velocity weight cut to 0.05 so an already-standing student scores high
    # even with zero upward movement.
    STAND_W_SHOULDER_DESK: float = 0.50   # ↑ primary: shoulder height above desk
    STAND_W_HEAD_DESK:     float = 0.25   # secondary: head height above desk
    STAND_W_VERT_VEL:      float = 0.05   # ↓ minimal: only helps during transition
    STAND_W_BODY_HEIGHT:   float = 0.20   # ↑ torso elongation (static geometry)

    # Vertical velocity normalisation (sw/frame) — kept for the small weight
    STAND_VEL_NORM:        float = 0.08

    # Body elongation thresholds
    STAND_BODY_HT_SIT:   float = 0.80
    STAND_BODY_HT_STAND: float = 1.10

    # Hysteresis — lowered so already-standing students are confirmed fast
    STAND_SCORE_HIGH: float = 0.42   # ↓ easier to cross when static
    STAND_SCORE_LOW:  float = 0.20
    STAND_SCORE_EMA:  float = 0.40   # ↑ faster EMA = responds to static geometry immediately

    # ── Behaviour 4 : Bouncing (multi-signal) ────────────────────────────
    BOUNCE_FFT_WIN:        int   = 32
    BOUNCE_PEAK_THRESH:    float = 0.26   # FFT peakedness threshold (lowered)
    BOUNCE_MIN_AMP_NORM:   float = 0.025  # min amplitude / sw

    # Additional signals:
    BOUNCE_VARIANCE_WIN:   int   = 20     # short window for variance check
    BOUNCE_VARIANCE_MIN:   float = 1e-4   # min normalised variance to be considered
    BOUNCE_ZCR_WIN:        int   = 16     # zero-crossing rate window
    BOUNCE_ZCR_MIN:        float = 0.10   # min zero-crossing rate (oscillations/frame)

    # Score weights (must sum to 1.0):
    BOUNCE_W_FFT:      float = 0.45
    BOUNCE_W_VARIANCE: float = 0.25
    BOUNCE_W_ZCR:      float = 0.20
    BOUNCE_W_AMP:      float = 0.10

    # Hysteresis
    BOUNCE_SCORE_HIGH:     float = 0.35
    BOUNCE_SCORE_LOW:      float = 0.15
    BOUNCE_SCORE_EMA:      float = 0.40   # faster EMA — bouncing starts/stops abruptly
    BOUNCE_CONFIRM_FRAMES: int   = 4

    # ── Behaviour 5 : Hand raise  (score-based, hysteresis) ──────────────────
    # Score = weighted sum of sub-signals; hysteresis on HIGH/LOW thresholds.
    #
    # Sub-signal weights — geometry-first so a motionless raised hand still scores
    # highly.  Velocity weight cut from 0.15 → 0.05; positional weights raised.
    HAND_W_WRIST_VS_SHOULDER: float = 0.38   # ↑ wrist height vs shoulder (primary)
    HAND_W_WRIST_VS_NOSE:     float = 0.30   # ↑ wrist height vs nose (strong geometric signal)
    HAND_W_ELBOW_VS_SHOULDER: float = 0.17   # elbow height vs shoulder
    HAND_W_UPWARD_VEL:        float = 0.05   # ↓ minimal: only helps during raise motion
    HAND_W_ELBOW_ANGLE:       float = 0.10   # elbow bend angle (arm extended up)

    # Normalisation ceilings
    HAND_HEIGHT_NORM:  float = 1.5   # sw units — wrist this far above shoulder = score 1.0
    HAND_VEL_NORM:     float = 0.12  # sw/frame — this upward velocity = score 1.0

    # Hysteresis thresholds — lowered so static raised arm is confirmed faster
    HAND_SCORE_HIGH:   float = 0.38  # ↓ easier to confirm a static raised arm
    HAND_SCORE_LOW:    float = 0.16  # ↓ lower exit threshold for hysteresis

    # EMA — faster so it follows a static geometry change in the first frames
    HAND_SCORE_EMA:    float = 0.45  # ↑ more responsive

    # Confirmation frames — reduced so static raises are caught immediately
    HAND_RAISE_HOLD:   int   = 2    # ↓ frames score > HIGH to confirm (was 5)
    HAND_LOWER_HOLD:   int   = 8    # frames score < LOW  to confirm lower

    # Elbow angle threshold for raised-arm detection
    HAND_ELBOW_RAISE_ANGLE_MIN: float = 100.0  # ↓ slightly relaxed (was 110°)

    # ── Display ────────────────────────────────────────────────────────────
    WINDOW_NAME: str = "Behaviour Classifier v5  —  Classroom Edition"

    # Behaviour colour map (BGR)
    BEHAVIOUR_COLORS: dict = {
        "sitting":     (0,   200, 255),   # amber
        "standing":    (0,   220,  80),   # green
        "slouching":   (0,    80, 255),   # red-orange
        "bouncing":    (255, 140,   0),   # blue
        "hand_raised": (255,   0, 200),   # magenta
        "unknown":     (140, 140, 140),   # grey
    }

    # Desk estimate line drawn on video (BGR)
    DESK_LINE_COLOR: tuple = (0, 255, 220)   # cyan

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
    Smooths detection jitter; extrapolates through brief occlusions via
    predict_only() which advances the filter without a measurement.
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
        """Advance state without measurement — interpolation for missing frames."""
        if not self._init:
            return 0.0, 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.missing_frames += 1
        return float(self.x[0]), float(self.x[2])

    def update(self, mx: float, my: float) -> tuple[float, float]:
        """Feed a valid measurement; returns smoothed (x, y)."""
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
        """kp_array : (N_KP, 3)  columns → [x, y, confidence]"""
        for i in range(N_KP):
            x_raw = float(kp_array[i, 0])
            y_raw = float(kp_array[i, 1])
            conf  = float(kp_array[i, 2]) if kp_array.shape[1] > 2 else 1.0

            # Mark as missing if: low confidence, NaN, or (0,0) sentinel
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
                # Extrapolate: Kalman predicts without new measurement
                sx, sy = kf.predict_only()
                self.smooth[i] = [sx, sy]
                self.valid[i]  = True    # still usable — interpolated
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

def _tilt_from_vertical(top: tuple, bottom: tuple) -> float:
    """
    Angle between vector top→bottom and the downward vertical (0, +1).
    0° = perfectly upright. 90° = horizontal.
    Image coords: y increases downward, so pointing down = (0,+1).
    """
    dx, dy = bottom[0] - top[0], bottom[1] - top[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return 0.0
    return float(math.degrees(math.acos(max(-1.0, min(1.0, dy / length)))))


def _body_ruler(kpb: KeypointBuffer) -> float:
    """
    Shoulder width = body ruler.
    All pixel thresholds are expressed as a fraction of this value,
    making every threshold camera-distance independent.
    """
    ls = kpb.get(KP_LEFT_SHOULDER)
    rs = kpb.get(KP_RIGHT_SHOULDER)
    if ls is None or rs is None:
        return CFG.FALLBACK_SW
    return max(math.hypot(rs[0] - ls[0], rs[1] - ls[1]), 1.0)


def _angle_deg(A: tuple, B: tuple, C: tuple) -> float:
    """
    Angle at vertex B in the triangle A-B-C. Returns [0, 180] degrees.
    Used for elbow angle (shoulder-elbow-wrist).
    """
    ba = np.array([A[0]-B[0], A[1]-B[1]], dtype=np.float64)
    bc = np.array([C[0]-B[0], C[1]-B[1]], dtype=np.float64)
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-6 or nc < 1e-6:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0))))


def _ema(prev: float, new_val: float, alpha: float) -> float:
    """
    Exponential moving average.
    alpha ∈ (0,1]: higher = more responsive, lower = smoother.
    Handles NaN gracefully (returns prev unchanged).
    """
    if math.isnan(prev):
        return new_val
    if math.isnan(new_val):
        return prev
    return alpha * new_val + (1.0 - alpha) * prev


# ==============================================================================
# DESK ESTIMATOR  (per-track, automatic)
# ==============================================================================

class DeskEstimator:
    """
    Estimates the Y pixel coordinate of the desk surface for one student.

    STRATEGY
    ────────
    Students in a classroom rest wrists AND elbows on the desk most of the
    time. We observe both and take the higher Y value each frame (closest
    to the desk surface). The rolling MEDIAN of these observations is the
    desk estimate — robust to outliers from arm-raises and arm-swinging.

    OUTLIER REJECTION
    ─────────────────
    Before appending a new sample we check it is within ±1.5 sw of the
    current estimate. This rejects:
      • Hand-raise frames (wrist suddenly much higher = lower Y)
      • Detection glitches (wrist teleports to a wrong position)

    INITIALISATION
    ──────────────
    For the first DESK_MIN_SAMPLES frames we fall back to a simple running
    maximum (the highest-ever wrist/elbow Y seen so far).  This gives a
    usable estimate fast, before the median window fills up.
    """

    def __init__(self) -> None:
        self._obs_buf:    deque = deque(maxlen=CFG.DESK_ESTIMATE_WIN)
        self.desk_y:      float = float("nan")
        self._init_max:   float = float("nan")   # fast init via running max
        self._n_samples:  int   = 0

    def update(self, kpb: KeypointBuffer, sw: float, hand_raised: bool) -> None:
        """Feed one frame. sw is needed for outlier rejection."""
        if hand_raised:
            return   # wrist Y would be an outlier — skip entirely

        # Candidate anchor points — whichever visible joint is closest to desk
        candidates: list[float] = []
        for idx in (KP_LEFT_WRIST, KP_RIGHT_WRIST,
                    KP_LEFT_ELBOW, KP_RIGHT_ELBOW):
            pt = kpb.get(idx)
            if pt:
                candidates.append(pt[1])

        if not candidates:
            return

        # Take the MAXIMUM Y (lowest on screen = closest to desk surface)
        sample = max(candidates)

        # Outlier rejection once we have a reference
        if not math.isnan(self.desk_y):
            if abs(sample - self.desk_y) > 1.5 * sw:
                return   # too far from current estimate — reject

        # Fast initialisation: track running maximum
        if math.isnan(self._init_max) or sample > self._init_max:
            self._init_max = sample

        self._obs_buf.append(sample)
        self._n_samples += 1

        if self._n_samples >= CFG.DESK_MIN_SAMPLES:
            # Stable estimate: median of rolling window
            self.desk_y = float(np.median(list(self._obs_buf)))
        else:
            # Early estimate: running max (fast to converge)
            self.desk_y = self._init_max

    @property
    def is_ready(self) -> bool:
        return not math.isnan(self.desk_y) and self._n_samples >= 3


# ==============================================================================
# BEHAVIOUR 1, 2, 3 — CLASSROOM POSTURE CLASSIFIER  (score-based)
# ==============================================================================

@dataclass
class PostureResult:
    label:              str   = "unknown"
    # Raw signals (all logged to CSV for analysis)
    spine_tilt_deg:     float = float("nan")
    head_desk_norm:     float = float("nan")   # (desk_y - nose_y) / sw
    shoulder_desk_norm: float = float("nan")   # (desk_y - shoulder_mid_y) / sw
    forward_shift_norm: float = float("nan")   # |nose_x - sh_mid_x| / sw
    body_height_norm:   float = float("nan")   # nose-to-shoulder dist / sw
    # Computed scores [0, 1]
    slouch_score:       float = 0.0
    stand_score:        float = 0.0
    confidence:         float = 0.0


class PostureStateMachine:
    """
    Maintains EMA-smoothed scores for slouching and standing,
    and applies independent hysteresis to each behaviour.

    This avoids binary threshold flipping and detects subtle posture changes.
    """

    _UPRIGHT  = 0
    _SLOUCHING = 1
    _STANDING  = 2

    def __init__(self) -> None:
        # EMA-smoothed scores
        self._slouch_score_ema: float = 0.0
        self._stand_score_ema:  float = 0.0
        # Hysteresis counters
        self._slouch_high_cnt:  int   = 0
        self._slouch_low_cnt:   int   = 0
        self._stand_high_cnt:   int   = 0
        self._stand_low_cnt:    int   = 0
        # Confirmed states
        self._slouching: bool = False
        self._standing:  bool = False
        # History for vertical velocity of nose (for standing detection)
        self._nose_y_hist: deque = deque(maxlen=8)

    def update(
        self,
        kpb:    KeypointBuffer,
        sw:     float,
        desk_y: float,
    ) -> PostureResult:
        res        = PostureResult()
        desk_ready = not math.isnan(desk_y)

        # ── Keypoints ──────────────────────────────────────────────────────
        nose = kpb.get(KP_NOSE)
        ls   = kpb.get(KP_LEFT_SHOULDER)
        rs   = kpb.get(KP_RIGHT_SHOULDER)

        if ls and rs:
            sh_mid: Optional[tuple] = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2)
        elif ls:
            sh_mid = ls
        elif rs:
            sh_mid = rs
        else:
            sh_mid = None

        # ── Signal: forward tilt (nose → shoulder) ─────────────────────────
        # 0° = nose directly above shoulders (upright)
        # Large angle = leaning forward toward desk
        if nose and sh_mid:
            res.spine_tilt_deg = _tilt_from_vertical(nose, sh_mid)

        # ── Signal: shoulder–desk gap ──────────────────────────────────────
        if sh_mid and desk_ready:
            res.shoulder_desk_norm = (desk_y - sh_mid[1]) / sw

        # ── Signal: head–desk gap  ─────────────────────────────────────────
        if nose and desk_ready:
            res.head_desk_norm = (desk_y - nose[1]) / sw

        # ── Signal: forward head shift ─────────────────────────────────────
        # When a student leans forward, the nose X shifts toward/away from
        # the camera projection of the shoulder midpoint.
        # We use absolute value so left/right camera orientation doesn't matter.
        if nose and sh_mid:
            res.forward_shift_norm = abs(nose[0] - sh_mid[0]) / sw

        # ── Signal: body height (nose-to-shoulder distance) ───────────────
        # When standing the torso straightens and the nose rises relative
        # to the shoulder midpoint → this distance increases.
        if nose and sh_mid:
            res.body_height_norm = math.hypot(
                nose[0]-sh_mid[0], nose[1]-sh_mid[1]
            ) / sw

        # ── Vertical velocity of nose (for standing detection) ─────────────
        if nose:
            self._nose_y_hist.append(nose[1] / sw)
        vert_vel_up = 0.0
        if len(self._nose_y_hist) >= 3:
            # Upward velocity (positive = moving up = shrinking y)
            vert_vel_up = max(0.0,
                -(float(self._nose_y_hist[-1]) - float(self._nose_y_hist[-3])) / 2.0
            )

        # ══════════════════════════════════════════════════════════════════
        # SLOUCH SCORE  (weighted sum of 3 sub-signals, each ∈ [0, 1])
        # ══════════════════════════════════════════════════════════════════

        # Sub-signal A: head-desk proximity
        # Maps head_desk_norm from [CLEAR, FLOOR] → [0, 1]
        # Contribution is HIGH when head is close to desk.
        if not math.isnan(res.head_desk_norm):
            hdg = res.head_desk_norm
            s_head = float(np.clip(
                (CFG.SLOUCH_HEAD_CLEAR - hdg)
                / max(CFG.SLOUCH_HEAD_CLEAR - CFG.SLOUCH_HEAD_FLOOR, 0.01),
                0.0, 1.0
            ))
        else:
            s_head = 0.0

        # Sub-signal B: spine tilt
        # Maps tilt from [CLEAR, MAX] → [0, 1]
        if not math.isnan(res.spine_tilt_deg):
            tilt = res.spine_tilt_deg
            s_tilt = float(np.clip(
                (tilt - CFG.SLOUCH_TILT_CLEAR)
                / max(CFG.SLOUCH_TILT_MAX - CFG.SLOUCH_TILT_CLEAR, 0.01),
                0.0, 1.0
            ))
        else:
            s_tilt = 0.0

        # Sub-signal C: forward head shift
        if not math.isnan(res.forward_shift_norm):
            fwd = res.forward_shift_norm
            s_fwd = float(np.clip(
                (fwd - CFG.SLOUCH_FORWARD_SHIFT_CLEAR)
                / max(CFG.SLOUCH_FORWARD_SHIFT_MAX - CFG.SLOUCH_FORWARD_SHIFT_CLEAR, 0.01),
                0.0, 1.0
            ))
        else:
            s_fwd = 0.0

        raw_slouch = (
            CFG.SLOUCH_W_HEAD_DESK     * s_head
            + CFG.SLOUCH_W_SPINE_TILT  * s_tilt
            + CFG.SLOUCH_W_FORWARD_SHIFT * s_fwd
        )
        # EMA smoothing
        self._slouch_score_ema = _ema(
            self._slouch_score_ema, raw_slouch, CFG.SLOUCH_SCORE_EMA
        )
        res.slouch_score = self._slouch_score_ema

        # ══════════════════════════════════════════════════════════════════
        # STAND SCORE  (weighted sum of 4 sub-signals)
        # ══════════════════════════════════════════════════════════════════

        # Sub-signal A: shoulder height above desk
        # Normalised so that a shoulder at exactly STANDING_SHOULDER_ABOVE_DESK_NORM
        # already yields score 0.5 — meaning a motionless standing student reaches
        # STAND_SCORE_HIGH from geometry alone without any velocity contribution.
        if not math.isnan(res.shoulder_desk_norm):
            sdg = res.shoulder_desk_norm
            s_sh = float(np.clip(
                (sdg - CFG.STANDING_SHOULDER_ABOVE_DESK_NORM + 0.40) / 0.80,
                0.0, 1.0
            ))
        else:
            s_sh = 0.0

        # Sub-signal B: head height above desk
        # A standing person's nose is typically 1.7–2.5 sw above desk.
        # Seated: ~1.2–1.6 sw. Map [1.5, 2.5] → [0, 1].
        if not math.isnan(res.head_desk_norm):
            hdg2 = res.head_desk_norm
            s_hd = float(np.clip(
                (hdg2 - 1.50) / 1.00,
                0.0, 1.0
            ))
        else:
            s_hd = 0.0

        # Sub-signal C: upward velocity — minimal weight, only useful during transition
        s_vel = float(np.clip(vert_vel_up / max(CFG.STAND_VEL_NORM, 1e-6), 0.0, 1.0))

        # Sub-signal D: body elongation — static geometry, high weight
        # nose-to-shoulder distance increases when torso straightens upright.
        if not math.isnan(res.body_height_norm):
            bh = res.body_height_norm
            s_bh = float(np.clip(
                (bh - CFG.STAND_BODY_HT_SIT)
                / max(CFG.STAND_BODY_HT_STAND - CFG.STAND_BODY_HT_SIT, 0.01),
                0.0, 1.0
            ))
        else:
            s_bh = 0.0

        raw_stand = (
            CFG.STAND_W_SHOULDER_DESK * s_sh
            + CFG.STAND_W_HEAD_DESK   * s_hd
            + CFG.STAND_W_VERT_VEL    * s_vel
            + CFG.STAND_W_BODY_HEIGHT * s_bh
        )
        self._stand_score_ema = _ema(
            self._stand_score_ema, raw_stand, CFG.STAND_SCORE_EMA
        )
        res.stand_score = self._stand_score_ema

        # ══════════════════════════════════════════════════════════════════
        # HYSTERESIS STATE MACHINES
        # ══════════════════════════════════════════════════════════════════
        # Confirmation counts are intentionally low (1–2 frames) so that a
        # student who is already in a static posture (standing, hand raised)
        # is detected immediately on the first frames the score exceeds HIGH,
        # without waiting for movement dynamics to accumulate.

        # --- Slouch hysteresis ---
        if not self._slouching:
            if self._slouch_score_ema >= CFG.SLOUCH_SCORE_HIGH:
                self._slouch_high_cnt += 1
                if self._slouch_high_cnt >= 2:   # confirm after 2 frames
                    self._slouching = True
                    self._slouch_low_cnt = 0
            else:
                self._slouch_high_cnt = max(0, self._slouch_high_cnt - 1)
        else:
            if self._slouch_score_ema < CFG.SLOUCH_SCORE_LOW:
                self._slouch_low_cnt += 1
                if self._slouch_low_cnt >= 5:    # need 5 frames below LOW to exit
                    self._slouching = False
                    self._slouch_high_cnt = 0
            else:
                self._slouch_low_cnt = max(0, self._slouch_low_cnt - 1)

        # --- Stand hysteresis ---
        if not self._standing:
            if self._stand_score_ema >= CFG.STAND_SCORE_HIGH:
                self._stand_high_cnt += 1
                if self._stand_high_cnt >= 2:    # confirm after 2 frames (was 4)
                    self._standing = True
                    self._stand_low_cnt = 0
            else:
                self._stand_high_cnt = max(0, self._stand_high_cnt - 1)
        else:
            if self._stand_score_ema < CFG.STAND_SCORE_LOW:
                self._stand_low_cnt += 1
                if self._stand_low_cnt >= 6:     # need 6 frames below LOW to exit
                    self._standing = False
                    self._stand_high_cnt = 0
            else:
                self._stand_low_cnt = max(0, self._stand_low_cnt - 1)

        # ══════════════════════════════════════════════════════════════════
        # LABEL ASSIGNMENT  (priority: standing > slouching > sitting > unknown)
        # Standing takes priority: a student who stood up is not "slouching"
        # ══════════════════════════════════════════════════════════════════
        if self._standing and desk_ready:
            res.label      = "standing"
            res.confidence = round(self._stand_score_ema, 3)
        elif self._slouching:
            res.label      = "slouching"
            res.confidence = round(self._slouch_score_ema, 3)
        elif not math.isnan(res.shoulder_desk_norm) and \
                res.shoulder_desk_norm > CFG.SITTING_SHOULDER_ABOVE_DESK_NORM:
            res.label      = "sitting"
            span = max(
                CFG.STANDING_SHOULDER_ABOVE_DESK_NORM - CFG.SITTING_SHOULDER_ABOVE_DESK_NORM,
                0.01
            )
            res.confidence = min(1.0,
                (res.shoulder_desk_norm - CFG.SITTING_SHOULDER_ABOVE_DESK_NORM) / span)
        elif not desk_ready and not math.isnan(res.spine_tilt_deg):
            # Desk not yet estimated — assume sitting in classroom context
            res.label      = "sitting"
            res.confidence = 0.20
        else:
            res.label = "unknown"

        return res


# ==============================================================================
# BEHAVIOUR 4 — BOUNCING DETECTOR  (multi-signal: FFT + variance + ZCR)
# ==============================================================================

class BouncingDetector:
    """
    Detects rhythmic AND irregular bouncing using three complementary signals:

    1. FFT peakedness — good for RHYTHMIC bouncing (regular frequency)
    2. Short-term variance — catches ANY oscillation, regular or not
    3. Zero-crossing rate — counts oscillation frequency independent of amplitude

    All three are weighted into a bounce_score, EMA-smoothed,
    and passed through a hysteresis gate.

    HEAD Y position (nose, then shoulder fallback) is tracked — always
    visible above the desk in a classroom.
    """

    def __init__(self) -> None:
        # Rolling buffer for the full FFT window
        self._head_y_buf: deque = deque(maxlen=CFG.BOUNCE_FFT_WIN * 2)
        # EMA-smoothed bounce score
        self._score_ema:  float = 0.0
        # Hysteresis counters
        self._high_cnt:   int   = 0
        self._low_cnt:    int   = 0
        # Confirmed state
        self._bouncing:   bool  = False

    def _get_head_y(self, kpb: KeypointBuffer, sw: float) -> Optional[float]:
        """Returns normalised head Y. Primary = nose, fallback = shoulder mid."""
        nose = kpb.get(KP_NOSE)
        if nose:
            return nose[1] / sw
        ls = kpb.get(KP_LEFT_SHOULDER); rs = kpb.get(KP_RIGHT_SHOULDER)
        if ls and rs:  return ((ls[1]+rs[1])/2.0) / sw
        if ls:         return ls[1] / sw
        if rs:         return rs[1] / sw
        return None

    def update(self, kpb: KeypointBuffer, sw: float) -> tuple[float, int]:
        """Returns (bounce_score ∈ [0,1], bouncing ∈ {0,1})."""
        head_y = self._get_head_y(kpb, sw)
        if head_y is None:
            self._score_ema = _ema(self._score_ema, 0.0, CFG.BOUNCE_SCORE_EMA)
            return self._score_ema, int(self._bouncing)

        self._head_y_buf.append(head_y)

        if len(self._head_y_buf) < CFG.BOUNCE_FFT_WIN:
            return 0.0, 0

        # ── Detrend the signal (remove slow drift / posture change) ───────
        sig_full = np.array(list(self._head_y_buf), dtype=np.float32)
        sig = sig_full[-CFG.BOUNCE_FFT_WIN:]
        sig_d = sig - sig.mean()

        # ── Amplitude gate — reject signals too small to be bouncing ──────
        amp = float(np.max(np.abs(sig_d)))
        if amp < CFG.BOUNCE_MIN_AMP_NORM:
            raw_score = 0.0
        else:
            # ── Sub-signal 1: FFT peakedness ─────────────────────────────
            # A single dominant frequency = rhythmic bouncing
            spec = np.abs(np.fft.rfft(sig_d))[1:]
            if spec.max() > 0:
                fft_score = float(np.clip(
                    float(spec.max()) / float(spec.sum()), 0.0, 1.0
                ))
                # Only count if peak exceeds threshold (filter white noise)
                fft_score = fft_score if fft_score >= CFG.BOUNCE_PEAK_THRESH else 0.0
            else:
                fft_score = 0.0

            # ── Sub-signal 2: short-term variance ─────────────────────────
            # Measures energy of oscillation regardless of regularity.
            # Normalised so that VARIANCE_MIN → 0, and a strong signal → 1.
            var_sig = sig_full[-CFG.BOUNCE_VARIANCE_WIN:]
            var_sig_d = var_sig - var_sig.mean()
            variance = float(np.var(var_sig_d))
            var_score = float(np.clip(
                (variance - CFG.BOUNCE_VARIANCE_MIN)
                / max(CFG.BOUNCE_VARIANCE_MIN * 20, 1e-8),
                0.0, 1.0
            ))

            # ── Sub-signal 3: zero-crossing rate ──────────────────────────
            # Counts how many times the signal crosses its mean per frame.
            # Regular bouncing produces a consistent ZCR.
            zcr_sig = sig_full[-CFG.BOUNCE_ZCR_WIN:]
            zcr_d   = zcr_sig - zcr_sig.mean()
            n_crossings = int(np.sum(np.diff(np.sign(zcr_d)) != 0))
            zcr = n_crossings / max(len(zcr_d) - 1, 1)
            zcr_score = float(np.clip(
                (zcr - CFG.BOUNCE_ZCR_MIN)
                / max(0.40 - CFG.BOUNCE_ZCR_MIN, 0.01),
                0.0, 1.0
            ))

            # ── Sub-signal 4: normalised amplitude ─────────────────────────
            amp_score = float(np.clip(
                amp / (CFG.BOUNCE_MIN_AMP_NORM * 8),
                0.0, 1.0
            ))

            raw_score = (
                CFG.BOUNCE_W_FFT      * fft_score
                + CFG.BOUNCE_W_VARIANCE * var_score
                + CFG.BOUNCE_W_ZCR      * zcr_score
                + CFG.BOUNCE_W_AMP      * amp_score
            )

        # ── EMA smoothing ─────────────────────────────────────────────────
        self._score_ema = _ema(self._score_ema, raw_score, CFG.BOUNCE_SCORE_EMA)

        # ── Hysteresis ────────────────────────────────────────────────────
        if not self._bouncing:
            if self._score_ema >= CFG.BOUNCE_SCORE_HIGH:
                self._high_cnt += 1
                if self._high_cnt >= CFG.BOUNCE_CONFIRM_FRAMES:
                    self._bouncing = True
                    self._low_cnt  = 0
            else:
                self._high_cnt = max(0, self._high_cnt - 1)
        else:
            if self._score_ema < CFG.BOUNCE_SCORE_LOW:
                self._low_cnt += 1
                if self._low_cnt >= CFG.BOUNCE_CONFIRM_FRAMES * 2:
                    self._bouncing = False
                    self._high_cnt = 0
            else:
                self._low_cnt = max(0, self._low_cnt - 1)

        return round(self._score_ema, 4), int(self._bouncing)


# ==============================================================================
# BEHAVIOUR 5 — HAND RAISE  (score-based, detects partial raises)
# ==============================================================================

class HandRaiseSM:
    """
    Score-based hand-raise detector with partial raise sensitivity.

    PROBLEM WITH BINARY APPROACH:
    In a classroom, a student raising their hand only halfway (wrist near
    shoulder level, not above head) would not trigger the old binary check.

    SOLUTION: compute a continuous hand_score from 5 weighted sub-signals,
    apply EMA smoothing, then use hysteresis on HIGH/LOW thresholds.

    Sub-signals:
    1. wrist_vs_shoulder — normalised vertical gap wrist above shoulder.
       Fires partially when wrist is still below shoulder (partial raise).
    2. wrist_vs_nose     — wrist height relative to nose.
    3. elbow_vs_shoulder — elbow elevation (catches occluded wrists).
    4. upward_velocity   — Kalman vy_up of the best wrist.
    5. elbow_angle       — shoulder-elbow-wrist angle (>110° = raised arm).
    """

    _IDLE = 0; _CANDIDATE = 1; _RAISED = 2; _LOWERING = 3

    def __init__(self) -> None:
        self._state        = self._IDLE
        self._rcnt         = 0
        self._lcnt         = 0
        self._score_ema    = 0.0
        # Exposed diagnostics
        self.hand_score:          float = 0.0
        self.wrist_above_shoulder: bool = False
        self.wrist_vel_up:         bool = False
        self.raised:               int  = 0

    def _sub_wrist_vs_shoulder(self, ls, rs, lw, rw, sw):
        scores = []
        for sh, wr in [(ls, lw), (rs, rw)]:
            if sh and wr:
                gap = (sh[1] - wr[1]) / sw   # positive = wrist above shoulder
                # Partial raise: start contributing when wrist is within 0.5 sw below shoulder
                scores.append(float(np.clip(
                    (gap + 0.5) / (CFG.HAND_HEIGHT_NORM + 0.5), 0.0, 1.0
                )))
        return max(scores) if scores else 0.0

    def _sub_wrist_vs_nose(self, nose, lw, rw, sw):
        if nose is None:
            return 0.0
        scores = []
        for wr in [lw, rw]:
            if wr:
                gap = (nose[1] - wr[1]) / sw   # positive = wrist above nose
                scores.append(float(np.clip((gap + 0.3) / 0.8, 0.0, 1.0)))
        return max(scores) if scores else 0.0

    def _sub_elbow_vs_shoulder(self, ls, rs, le, re, sw):
        scores = []
        for sh, el in [(ls, le), (rs, re)]:
            if sh and el:
                gap = (sh[1] - el[1]) / sw
                scores.append(float(np.clip((gap + 0.3) / 0.8, 0.0, 1.0)))
        return max(scores) if scores else 0.0

    def _sub_upward_velocity(self, kpb, sw):
        lv = kpb.kpf(KP_LEFT_WRIST).vy_up
        rv = kpb.kpf(KP_RIGHT_WRIST).vy_up
        return float(np.clip(max(lv, rv) / (sw * max(CFG.HAND_VEL_NORM, 1e-6)), 0.0, 1.0))

    def _sub_elbow_angle(self, ls, rs, le, re, lw, rw):
        """shoulder-elbow-wrist angle: >110° = arm raised/extended upward."""
        angles = []
        for sh, el, wr in [(ls, le, lw), (rs, re, rw)]:
            if sh and el and wr:
                angles.append(_angle_deg(sh, el, wr))
        if not angles:
            return 0.0
        return float(np.clip(
            (max(angles) - CFG.HAND_ELBOW_RAISE_ANGLE_MIN)
            / (180.0 - CFG.HAND_ELBOW_RAISE_ANGLE_MIN),
            0.0, 1.0
        ))

    def update(self, kpb: KeypointBuffer, sw: float) -> None:
        ls  = kpb.get(KP_LEFT_SHOULDER);  rs  = kpb.get(KP_RIGHT_SHOULDER)
        lw  = kpb.get(KP_LEFT_WRIST);     rw  = kpb.get(KP_RIGHT_WRIST)
        le  = kpb.get(KP_LEFT_ELBOW);     re  = kpb.get(KP_RIGHT_ELBOW)
        nose = kpb.get(KP_NOSE)

        s1 = self._sub_wrist_vs_shoulder(ls, rs, lw, rw, sw)
        s2 = self._sub_wrist_vs_nose(nose, lw, rw, sw)
        s3 = self._sub_elbow_vs_shoulder(ls, rs, le, re, sw)
        s4 = self._sub_upward_velocity(kpb, sw)
        s5 = self._sub_elbow_angle(ls, rs, le, re, lw, rw)

        raw_score = (
            CFG.HAND_W_WRIST_VS_SHOULDER   * s1
            + CFG.HAND_W_WRIST_VS_NOSE     * s2
            + CFG.HAND_W_ELBOW_VS_SHOULDER * s3
            + CFG.HAND_W_UPWARD_VEL        * s4
            + CFG.HAND_W_ELBOW_ANGLE       * s5
        )

        self._score_ema = _ema(self._score_ema, raw_score, CFG.HAND_SCORE_EMA)
        self.hand_score = round(self._score_ema, 3)
        self.wrist_above_shoulder = s1 > 0.5
        self.wrist_vel_up         = s4 > 0.3

        # ── Hysteresis state machine ──────────────────────────────────────
        if self._state == self._IDLE:
            if self._score_ema >= CFG.HAND_SCORE_HIGH:
                self._state = self._CANDIDATE; self._rcnt = 1
        elif self._state == self._CANDIDATE:
            if self._score_ema >= CFG.HAND_SCORE_HIGH:
                self._rcnt += 1
                if self._rcnt >= CFG.HAND_RAISE_HOLD:
                    self._state = self._RAISED
            else:
                self._state = self._IDLE; self._rcnt = 0
        elif self._state == self._RAISED:
            if self._score_ema < CFG.HAND_SCORE_LOW:
                self._state = self._LOWERING; self._lcnt = 1
        elif self._state == self._LOWERING:
            if self._score_ema >= CFG.HAND_SCORE_HIGH:
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
    track_id:  int
    kpb:       KeypointBuffer      = field(default_factory=KeypointBuffer)
    desk:      DeskEstimator       = field(default_factory=DeskEstimator)
    bounce:    BouncingDetector    = field(default_factory=BouncingDetector)
    hand_sm:   HandRaiseSM         = field(default_factory=HandRaiseSM)
    posture_sm: PostureStateMachine = field(default_factory=PostureStateMachine)

    def process(self, kp_array: np.ndarray) -> dict:
        """
        Full 5-behaviour pipeline for one frame.
        kp_array : (17, 3) — [x, y, visibility] from CSV.
        Returns a flat dict for CSV + overlay rendering.
        """
        # 1. Kalman-smooth + interpolate all 17 keypoints
        self.kpb.update(kp_array)

        # 2. Body ruler
        sw = _body_ruler(self.kpb)

        # 3. Hand raise FIRST (desk estimator skips hand-raised frames)
        self.hand_sm.update(self.kpb, sw)

        # 4. Update desk estimate
        self.desk.update(self.kpb, sw=sw, hand_raised=bool(self.hand_sm.raised))

        # 5. Posture via score-based state machine
        pr = self.posture_sm.update(self.kpb, sw, self.desk.desk_y)

        # 6. Bouncing (multi-signal)
        bounce_score, is_bouncing = self.bounce.update(self.kpb, sw)

        def _f(v: float) -> object:
            return round(v, 3) if not math.isnan(v) else ""

        return {
            # Body ruler
            "shoulder_width_px":    round(sw, 2),
            "desk_y_px":            round(self.desk.desk_y, 1)
                                    if self.desk.is_ready else "",
            # Raw posture signals
            "spine_tilt_deg":       _f(pr.spine_tilt_deg),
            "head_desk_norm":       _f(pr.head_desk_norm),
            "shoulder_desk_norm":   _f(pr.shoulder_desk_norm),
            "forward_shift_norm":   _f(pr.forward_shift_norm),
            "body_height_norm":     _f(pr.body_height_norm),
            # Posture scores + label
            "slouch_score":         round(pr.slouch_score, 3),
            "stand_score":          round(pr.stand_score, 3),
            "posture":              pr.label,
            "posture_confidence":   round(pr.confidence, 3),
            # Bouncing
            "bounce_score":         round(bounce_score, 3),
            "bouncing":             is_bouncing,
            # Hand raise
            "hand_score":           self.hand_sm.hand_score,
            "wrist_above_shoulder": int(self.hand_sm.wrist_above_shoulder),
            "wrist_vel_up":         int(self.hand_sm.wrist_vel_up),
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
    if feat["bouncing"]:     return "bouncing",    float(feat.get("bounce_score", 0.0))
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

    Memory: each (17,3) float64 array = 408 bytes.
    10 people × 10 000 frames = ~40 MB — well within normal RAM.

    Uses chunked reading so very large CSVs don't block the process.
    """
    print(f"[CSV]   Loading {body_csv}", end="", flush=True)

    body_index: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    chunk_size  = 17 * 20 * 500   # 500 frames × 20 persons at a time

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
        # Drop rows with no landmark index (invalid / header artifacts)
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
    """All OpenCV drawing logic. Stateless — only reads from track state."""

    FONT      = cv2.FONT_HERSHEY_SIMPLEX
    FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

    # Skeleton links for classroom scene (upper body only — legs hidden behind desk)
    # Format: (kp_a, kp_b, BGR_base_colour, thickness)
    _LINKS = [
        # Arms (colour overridden to arm_col dynamically)
        (KP_LEFT_SHOULDER,  KP_LEFT_ELBOW,    None,          2),
        (KP_LEFT_ELBOW,     KP_LEFT_WRIST,    None,          2),
        (KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW,   None,          2),
        (KP_RIGHT_ELBOW,    KP_RIGHT_WRIST,   None,          2),
        # Torso / spine direction (shoulder bar + shoulder→hip if visible)
        (KP_LEFT_SHOULDER,  KP_RIGHT_SHOULDER,(170, 170, 170), 2),
        (KP_LEFT_SHOULDER,  KP_LEFT_HIP,      (130, 130, 130), 1),
        (KP_RIGHT_SHOULDER, KP_RIGHT_HIP,     (130, 130, 130), 1),
        # Head (nose–eye line for orientation, faint)
        (KP_NOSE,           KP_LEFT_EYE,      (100, 100, 100), 1),
        (KP_NOSE,           KP_RIGHT_EYE,     (100, 100, 100), 1),
        (KP_LEFT_EYE,       KP_LEFT_EAR,      (100, 100, 100), 1),
        (KP_RIGHT_EYE,      KP_RIGHT_EAR,     (100, 100, 100), 1),
    ]
    _ARM_INDICES = {
        KP_LEFT_SHOULDER, KP_LEFT_ELBOW,  KP_LEFT_WRIST,
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
    ) -> tuple[int, int, int, int]:
        """
        Draws the full body skeleton on frame.
        Returns (x_min, y_min, x_max, y_max) bounding box of visible keypoints,
        used to position the badge and diagnostic panel.
        """
        beh_col  = cls._beh_col(beh)
        arm_col  = (0, 255, 0) if hand_raised else beh_col
        xs, ys   = [], []

        def _pt(idx: int) -> Optional[tuple[int, int]]:
            p = kpb.get(idx)
            if p:
                xs.append(int(p[0])); ys.append(int(p[1]))
                return int(p[0]), int(p[1])
            return None

        for a_idx, b_idx, fixed_col, thick in cls._LINKS:
            col = arm_col if a_idx in cls._ARM_INDICES else fixed_col
            pa  = _pt(a_idx); pb = _pt(b_idx)
            if pa and pb:
                cv2.line(frame, pa, pb, col, thick, cv2.LINE_AA)

        # Joint dots
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
        """Coloured pill badge above the person's head: [ ID:2  SITTING  82% ]"""
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

        # Confidence bar under the badge
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
        """Semi-transparent panel showing all new score fields."""

        def _s(key: str, dec: int = 2) -> str:
            v = feat.get(key, "")
            if v == "" or (isinstance(v, float) and math.isnan(v)):
                return "n/a"
            return str(v) if isinstance(v, str) else f"{float(v):.{dec}f}"

        lines = [
            f"desk_y  : {_s('desk_y_px',1)}px",
            f"spine   : {_s('spine_tilt_deg')}°",
            f"hd_desk : {_s('head_desk_norm')}",
            f"sh_desk : {_s('shoulder_desk_norm')}",
            f"fwd_sft : {_s('forward_shift_norm')}",
            f"slouch▶ : {_s('slouch_score')}",
            f"stand▶  : {_s('stand_score')}",
            f"bounce▶ : {_s('bounce_score')}",
            f"hand▶   : {_s('hand_score')}",
            f"W>sh:{feat.get('wrist_above_shoulder',0)}  Wv:{feat.get('wrist_vel_up',0)}",
        ]
        lh = 13; pw = 152; ph = lh * len(lines) + 6
        px1 = min(x_right + 4, frame.shape[1] - pw - 2)
        py1 = max(0, y_top)
        px2 = min(frame.shape[1] - 1, px1 + pw)
        py2 = min(frame.shape[0] - 1, py1 + ph)

        sub = frame[py1:py2, px1:px2]
        if sub.size > 0:
            frame[py1:py2, px1:px2] = cv2.addWeighted(
                sub, 0.20, np.zeros_like(sub), 0.80, 0
            )
        for i, line in enumerate(lines):
            ty = py1 + lh * (i + 1)
            if ty < frame.shape[0]:
                # Highlight lines where score > 0.3 in a warmer colour
                key_map = {
                    6: "slouch_score", 7: "stand_score",
                    8: "bounce_score", 9: "hand_score",
                }
                col = (200, 200, 200)
                if i in key_map:
                    val = feat.get(key_map[i], 0.0)
                    try:
                        if float(val) > 0.30:
                            col = (80, 220, 255)   # warm highlight
                    except (ValueError, TypeError):
                        pass
                cv2.putText(frame, line, (px1 + 3, ty),
                            cls.FONT, 0.35, col, 1, cv2.LINE_AA)

    @staticmethod
    def draw_desk_line(
        frame:    np.ndarray,
        desk_y:   float,
        x_min:    int,
        x_max:    int,
        track_id: int,
    ) -> None:
        """
        Draws a horizontal dashed line at the estimated desk surface Y.
        The line spans from x_min to x_max (the person's skeleton bounding box).
        Labelled with the track ID so multiple students' desks are distinguishable.
        """
        if math.isnan(desk_y):
            return
        y = int(desk_y)
        col = CFG.DESK_LINE_COLOR

        # Dashed line: draw segments of 10px on, 6px off
        seg_on, seg_off = 10, 6
        x = x_min
        first = True
        while x < x_max:
            x_end = min(x + seg_on, x_max)
            cv2.line(frame, (x, y), (x_end, y), col, 1, cv2.LINE_AA)
            x = x_end + seg_off

        # Label
        cv2.putText(frame, f"desk{track_id}", (x_min, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)

    @classmethod
    def draw_hand_banner(
        cls,
        frame:  np.ndarray,
        tid:    int,
        cx:     int,
        top_y:  int,
        tick:   int,
    ) -> None:
        """Blinking HAND RAISED banner (blinks every 15 frames)."""
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

    @staticmethod
    def draw_hud(
        frame:     np.ndarray,
        frame_id:  int,
        fps:       float,
        n_persons: int,
        paused:    bool,
        speed:     float,
        progress:  float,   # 0.0 – 1.0
    ) -> None:
        h, w = frame.shape[:2]

        # Top-left info panel
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

        # Progress bar at the bottom
        bar_h  = 6
        filled = int(w * progress)
        cv2.rectangle(frame, (0, h-bar_h), (w, h),       (40,40,40),   cv2.FILLED)
        cv2.rectangle(frame, (0, h-bar_h), (filled, h),  (0,200,255),  cv2.FILLED)

        # Controls hint
        hint = "SPACE=pause/step  Q=quit  S=snapshot  +/-=speed"
        (hw, hh), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.37, 1)
        cv2.putText(frame, hint, (w-hw-6, h-bar_h-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, (150,150,150), 1, cv2.LINE_AA)

    @classmethod
    def render_all(
        cls,
        frame:    np.ndarray,
        results:  dict,         # {track_id: (TrackState, feat_dict)}
        frame_id: int,
        fps:      float,
        paused:   bool,
        speed:    float,
        progress: float,
        tick:     int,
    ) -> None:
        """Master draw function — renders everything for one frame."""
        for tid, (track, feat) in results.items():
            beh, conf = dominant_behaviour(feat)
            hand_up   = bool(feat["hand_raised"])

            x_min, y_min, x_max, y_max = cls.draw_skeleton(
                frame, track.kpb, beh, hand_up
            )

            if x_max == 0 and y_max == 0:
                continue   # no visible keypoints — skip rendering

            cx = (x_min + x_max) // 2

            # Desk surface line
            cls.draw_desk_line(frame, track.desk.desk_y, x_min, x_max, tid)

            cls.draw_badge(frame, tid, beh, conf, cx, y_min)
            if hand_up:
                cls.draw_hand_banner(frame, tid, cx, y_min, tick)
            cls.draw_diagnostics(frame, feat, x_max, y_min)

        cls.draw_hud(frame, frame_id, fps, len(results), paused, speed, progress)


# ==============================================================================
# CSV OUTPUT COLUMNS
# ==============================================================================

_RAW_COLS = [
    "frame_id", "track_id",
    # Body ruler + desk
    "shoulder_width_px", "desk_y_px",
    # Raw posture signals
    "spine_tilt_deg", "head_desk_norm", "shoulder_desk_norm",
    "forward_shift_norm", "body_height_norm",
    # Posture scores + label
    "slouch_score", "stand_score", "posture", "posture_confidence",
    # Bouncing
    "bounce_score", "bouncing",
    # Hand raise
    "hand_score", "wrist_above_shoulder", "wrist_vel_up", "hand_raised",
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

    # ── Validate ──────────────────────────────────────────────────────────────
    for p, name in [(video_path, "Video"), (body_csv, "Body CSV")]:
        if not Path(p).exists():
            print(f"[ERROR] {name} not found: {p}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Behaviour Classifier v4  —  CSV-driven video overlay")
    print(f"{'='*60}\n")

    # ── Step 1: Load the body CSV completely ──────────────────────────────────
    # This gives us {frame_id: {track_id: (17,3) array}} and the sorted list
    # of frame_ids that were actually processed during extraction.
    body_index, frame_ids = load_body_csv(body_csv)

    if not frame_ids:
        print("[ERROR] Body CSV is empty or has no valid rows.")
        sys.exit(1)

    total_csv_frames = len(frame_ids)
    print(f"[CSV]   {total_csv_frames} frames will be processed.\n")

    # ── Step 2: Open video ────────────────────────────────────────────────────
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

    # ── Step 3: Optional video writer ─────────────────────────────────────────
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_video, fourcc, video_fps, (frame_w, frame_h))
        print(f"[Save]  Annotated video → {save_video}")

    # ── Step 4: Open streaming CSV writers ────────────────────────────────────
    raw_fh = open(raw_out, "w", newline="", encoding="utf-8")
    raw_w  = csv.DictWriter(raw_fh, fieldnames=_RAW_COLS)
    raw_w.writeheader()

    # ── Step 5: Runtime state ─────────────────────────────────────────────────
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

    # Cache of the last rendered frame (shown while paused)
    last_rendered: Optional[np.ndarray] = None

    cv2.namedWindow(CFG.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CFG.WINDOW_NAME, min(frame_w, 1400), min(frame_h, 860))

    # ── Step 6: Main loop — iterate over CSV frame IDs ────────────────────────
    #
    # KEY DESIGN:
    #   We iterate over `frame_ids` (taken from the CSV, not from cap).
    #   For each frame_id, we do:
    #     1. cap.set(CAP_PROP_POS_FRAMES, frame_id)  → seek video to exact frame
    #     2. cap.read()                               → decode that frame
    #     3. body_index[frame_id]                     → get keypoints from CSV
    #     4. run behaviour pipeline on keypoints
    #     5. render onto the decoded frame
    #   This guarantees perfect CSV ↔ video synchronisation regardless of
    #   FRAME_STRIDE or any gap in the CSV frame numbering.
    #
    csv_frame_idx = 0   # index into frame_ids list (for progress bar)

    while csv_frame_idx < total_csv_frames:

        # ── Handle key input ──────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            print("\n[Info]  User quit.")
            break

        if key == ord(' '):
            if not step_mode:
                paused = not paused
            # In step mode OR when paused: SPACE will advance one frame below

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

        # ── Pause handling ────────────────────────────────────────────────
        if paused and key != ord(' '):
            # Display the last rendered frame while paused; don't advance
            if last_rendered is not None:
                cv2.imshow(CFG.WINDOW_NAME, last_rendered)
            cv2.waitKey(30)
            continue

        # ── Get current CSV frame_id ──────────────────────────────────────
        frame_id = frame_ids[csv_frame_idx]

        # ── Seek video to this exact frame ────────────────────────────────
        # cap.set(CAP_PROP_POS_FRAMES, N) positions the reader at frame N.
        # This is what synchronises the video to the CSV perfectly.
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()

        if not ret:
            print(f"[Warn]  Could not read video frame {frame_id} — skipping.")
            csv_frame_idx += 1
            continue

        # ── Get keypoints for this frame from the pre-loaded index ────────
        persons = body_index.get(frame_id, {})

        if not persons:
            # CSV had no people detected in this frame — show the video frame
            # with just the HUD and move on
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

        # ── Run behaviour pipeline on all persons in this frame ───────────
        frame_results: dict[int, tuple] = {}

        for track_id, kp_array in persons.items():
            track = get_track(track_id)
            feat  = track.process(kp_array)

            # Write to raw CSV
            raw_w.writerow({"frame_id": frame_id, "track_id": track_id, **feat})
            rows_written += 1

            # Feed summary
            beh_label, beh_conf = dominant_behaviour(feat)
            summary.ingest(frame_id, track_id, beh_label, beh_conf)

            frame_results[track_id] = (track, feat)

        # Flush CSV periodically
        if csv_frame_idx % 60 == 0:
            raw_fh.flush()

        # ── FPS measurement ───────────────────────────────────────────────
        now = time.perf_counter(); dt = now - prev_t; prev_t = now
        if dt > 0: fps_buf.append(1.0/dt)
        disp_fps = float(np.mean(fps_buf)) if fps_buf else 0.0
        progress = csv_frame_idx / max(1, total_csv_frames - 1)

        # ── Render all overlays onto the frame ────────────────────────────
        renderer.render_all(
            frame, frame_results, frame_id,
            disp_fps, paused, speed, progress, tick
        )

        # ── Display ───────────────────────────────────────────────────────
        cv2.imshow(CFG.WINDOW_NAME, frame)
        last_rendered = frame.copy()

        if writer:
            writer.write(frame)

        csv_frame_idx += 1
        tick          += 1

        # Throttle playback to target speed
        cv2.waitKey(frame_delay_ms)

    # ── Cleanup ───────────────────────────────────────────────────────────────
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
        description="Behaviour Classifier v4 — CSV-driven real-time video overlay"
    )
    p.add_argument("--video",      default=CFG.VIDEO_PATH,
                   help=f"Input video file (default: {CFG.VIDEO_PATH})")
    p.add_argument("--body",       default=CFG.BODY_CSV,
                   help=f"raw_body_multi.csv from extract_raw_data_multi.py")
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
