"""
behavior_classifier.py
======================
Post-traitement de raw_body_multi.csv (généré par extract_raw_data_multi.py).

Calcule pour chaque track_id et chaque frame :
  1. is_hand_raised       — main levée (poignet au-dessus de l'épaule)
  2. posture              — "sitting" | "standing" | "slouching"
  3. agitation_score      — score cumulé d'agitation (fidgeting / TDAH)
  4. is_stimming          — mouvement répétitif oscillatoire des poignets (TSA)

Sorties :
  behavior_per_frame.csv  — indicateurs frame par frame
  behavior_summary.csv    — résumé agrégé par track_id

Usage :
    python behavior_classifier.py
    python behavior_classifier.py --input raw_body_multi.csv --fps 30
"""

import argparse
import os
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# COCO-17 keypoint indices (same as extractor)
# ──────────────────────────────────────────────
KP = {
    "nose":            0,
    "left_eye":        1,
    "right_eye":       2,
    "left_ear":        3,
    "right_ear":       4,
    "left_shoulder":   5,
    "right_shoulder":  6,
    "left_elbow":      7,
    "right_elbow":     8,
    "left_wrist":      9,
    "right_wrist":     10,
    "left_hip":        11,
    "right_hip":       12,
    "left_knee":       13,
    "right_knee":      14,
    "left_ankle":      15,
    "right_ankle":     16,
}

# ──────────────────────────────────────────────
# TUNEABLE THRESHOLDS
# ──────────────────────────────────────────────
VIS_THRESHOLD       = 0.3   # minimum YOLO confidence to consider a keypoint valid

# --- Hand Raising ---
HAND_RAISE_MARGIN   = 0.0   # pixels — wrist must be this much ABOVE shoulder (y_wrist < y_shoulder - margin)

# --- Posture ---
# Sitting  : shoulder-hip dist / total skeleton height in [SIT_LOW, SIT_HIGH]
SIT_RATIO_LOW       = 0.15
SIT_RATIO_HIGH      = 0.45
# Standing : shoulder-ankle vertical span > STAND_RATIO * total height
STAND_RATIO         = 0.55
# Slouching: nose-shoulder vertical dist < SLOUCH_RATIO * shoulder-hip dist
SLOUCH_RATIO        = 0.30

# --- Agitation (Fidgeting) ---
AGITATION_SPEED_THR = 8.0   # px/frame — mean joint speed above this → agitation++
AGITATION_JOINTS    = [     # joints used for agitation speed calculation
    KP["left_wrist"], KP["right_wrist"],
    KP["left_elbow"], KP["right_elbow"],
    KP["left_shoulder"], KP["right_shoulder"],
]

# --- Stimming (Repetitive wrist oscillation) ---
STIM_WINDOW_FRAMES  = 60    # rolling window length (frames) to analyse oscillation
STIM_MIN_PEAKS      = 4     # minimum oscillation peaks in window
STIM_MIN_AMPLITUDE  = 15    # px — minimum peak-to-trough amplitude
STIM_MAX_PERIOD     = 20    # frames — max allowed period between peaks (speed filter)


# ──────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────

def _kp(frame_kps: dict, idx: int):
    """
    Returns (x, y, vis) for keypoint `idx` in a frame's keypoint dict.
    Returns (nan, nan, 0) if missing.
    frame_kps : {landmark_idx -> (x, y, visibility)}
    """
    if idx in frame_kps:
        x, y, v = frame_kps[idx]
        return float(x), float(y), float(v)
    return np.nan, np.nan, 0.0


def _valid(vis: float) -> bool:
    return vis >= VIS_THRESHOLD


def _avg_y(frame_kps, idx_list):
    """Average y of visible keypoints in idx_list. Returns nan if none visible."""
    ys = [frame_kps[i][1] for i in idx_list
          if i in frame_kps and _valid(frame_kps[i][2])]
    return float(np.mean(ys)) if ys else np.nan


def _avg_x(frame_kps, idx_list):
    xs = [frame_kps[i][0] for i in idx_list
          if i in frame_kps and _valid(frame_kps[i][2])]
    return float(np.mean(xs)) if xs else np.nan


# ──────────────────────────────────────────────
# 1. HAND RAISING
# ──────────────────────────────────────────────

def detect_hand_raised(frame_kps: dict) -> bool:
    """
    True if at least one wrist is above (smaller y) its corresponding shoulder
    by more than HAND_RAISE_MARGIN pixels.
    """
    pairs = [
        (KP["left_wrist"],  KP["left_shoulder"]),
        (KP["right_wrist"], KP["right_shoulder"]),
    ]
    for wrist_idx, shoulder_idx in pairs:
        _, wy, wv = _kp(frame_kps, wrist_idx)
        _, sy, sv = _kp(frame_kps, shoulder_idx)
        if _valid(wv) and _valid(sv):
            if wy < sy - HAND_RAISE_MARGIN:   # origin at top-left → smaller y = higher
                return True
    return False


# ──────────────────────────────────────────────
# 2. POSTURE CLASSIFICATION
# ──────────────────────────────────────────────

def classify_posture(frame_kps: dict) -> str:
    """
    Returns one of: 'standing', 'slouching', 'sitting', 'unknown'.

    Geometry (all y-coordinates, origin top-left):
      shoulder_y  = mean(left_shoulder.y, right_shoulder.y)
      hip_y       = mean(left_hip.y, right_hip.y)
      ankle_y     = mean(left_ankle.y, right_ankle.y)
      nose_y      = nose.y

      shoulder_hip_dist  = hip_y - shoulder_y        (positive = hip below shoulder)
      shoulder_ankle_dist = ankle_y - shoulder_y
      total_height        = max visible span (ankle_y - nose_y if available, else shoulder_ankle_dist)
    """
    shoulder_y = _avg_y(frame_kps, [KP["left_shoulder"], KP["right_shoulder"]])
    hip_y      = _avg_y(frame_kps, [KP["left_hip"],      KP["right_hip"]])
    ankle_y    = _avg_y(frame_kps, [KP["left_ankle"],    KP["right_ankle"]])
    nose_x, nose_y, nose_v = _kp(frame_kps, KP["nose"])

    if np.isnan(shoulder_y):
        return "unknown"

    # ── Total skeleton height (reference denominator) ──
    if not np.isnan(ankle_y) and not np.isnan(nose_y) and _valid(nose_v):
        total_height = ankle_y - nose_y
    elif not np.isnan(ankle_y):
        total_height = ankle_y - shoulder_y
    elif not np.isnan(hip_y):
        total_height = hip_y - shoulder_y
    else:
        return "unknown"

    if total_height <= 0:
        return "unknown"

    # ── STANDING check ──
    if not np.isnan(ankle_y):
        shoulder_ankle = ankle_y - shoulder_y
        stand_ratio = shoulder_ankle / total_height
        if stand_ratio > STAND_RATIO:
            return "standing"

    # ── SLOUCHING check ──
    if not np.isnan(hip_y) and not np.isnan(nose_y) and _valid(nose_v):
        shoulder_hip = hip_y - shoulder_y
        nose_shoulder_dist = shoulder_y - nose_y   # positive when nose above shoulder
        if shoulder_hip > 0:
            slouch_ratio = nose_shoulder_dist / shoulder_hip
            if slouch_ratio < SLOUCH_RATIO:
                return "slouching"

    # ── SITTING (default if shoulder-hip ratio is in plausible range) ──
    if not np.isnan(hip_y):
        shoulder_hip = hip_y - shoulder_y
        sit_ratio = shoulder_hip / total_height
        if SIT_RATIO_LOW <= sit_ratio <= SIT_RATIO_HIGH:
            return "sitting"

    return "sitting"   # fallback default


# ──────────────────────────────────────────────
# 3. AGITATION SCORE (Fidgeting / TDAH)
# ──────────────────────────────────────────────

def compute_agitation(
    frame_kps_curr: dict,
    frame_kps_prev: dict,
) -> float:
    """
    Returns mean Euclidean speed (px/frame) of AGITATION_JOINTS between two frames.
    Returns 0.0 if not enough valid keypoints.
    """
    speeds = []
    for idx in AGITATION_JOINTS:
        x1, y1, v1 = _kp(frame_kps_curr, idx)
        x2, y2, v2 = _kp(frame_kps_prev, idx)
        if _valid(v1) and _valid(v2):
            speed = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            speeds.append(speed)
    return float(np.mean(speeds)) if speeds else 0.0


# ──────────────────────────────────────────────
# 4. STIMMING — Repetitive Wrist Oscillation
# ──────────────────────────────────────────────

def detect_stimming(wrist_y_series: np.ndarray) -> bool:
    """
    Analyses the last STIM_WINDOW_FRAMES values of a wrist y-trajectory.
    Returns True if a rapid, repetitive oscillation pattern is detected.

    Method:
      - Find local peaks and troughs in the signal.
      - Count valid oscillations (amplitude > STIM_MIN_AMPLITUDE,
        period < STIM_MAX_PERIOD frames).
    """
    if len(wrist_y_series) < STIM_WINDOW_FRAMES // 2:
        return False

    window = wrist_y_series[-STIM_WINDOW_FRAMES:]
    window = window[~np.isnan(window)]

    if len(window) < 10:
        return False

    # Smooth slightly to reduce noise
    kernel = np.ones(3) / 3
    smoothed = np.convolve(window, kernel, mode="same")

    # Detect peaks (upward excursions) and troughs (downward)
    peaks,  peak_props  = find_peaks(smoothed,  prominence=STIM_MIN_AMPLITUDE / 2)
    troughs, trough_props = find_peaks(-smoothed, prominence=STIM_MIN_AMPLITUDE / 2)

    if len(peaks) < STIM_MIN_PEAKS // 2 or len(troughs) < STIM_MIN_PEAKS // 2:
        return False

    # Merge and sort all extrema
    extrema = sorted(list(peaks) + list(troughs))
    if len(extrema) < STIM_MIN_PEAKS:
        return False

    # Check amplitudes and periods between consecutive extrema
    valid_oscillations = 0
    for i in range(len(extrema) - 1):
        period = extrema[i + 1] - extrema[i]
        amplitude = abs(smoothed[extrema[i + 1]] - smoothed[extrema[i]])
        if amplitude >= STIM_MIN_AMPLITUDE and period <= STIM_MAX_PERIOD:
            valid_oscillations += 1

    return valid_oscillations >= STIM_MIN_PEAKS - 1


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    print(f"[INFO] Loading {path} ...")
    df = pd.read_csv(path, dtype={
        "frame_id":     int,
        "track_id":     int,
        "landmark_idx": float,   # float to handle NaN rows
        "x":            float,
        "y":            float,
        "visibility":   float,
    })
    # Drop rows with missing landmark data
    df = df.dropna(subset=["landmark_idx", "x", "y"])
    df["landmark_idx"] = df["landmark_idx"].astype(int)
    print(f"[INFO] Loaded {len(df):,} rows | "
          f"{df['track_id'].nunique()} tracks | "
          f"{df['frame_id'].nunique()} frames")
    return df


def build_keypoint_dicts(df: pd.DataFrame) -> dict:
    """
    Returns nested dict:
      kp_store[track_id][frame_id] = {landmark_idx: (x, y, visibility)}
    """
    print("[INFO] Building keypoint lookup table ...")
    kp_store = defaultdict(lambda: defaultdict(dict))
    for row in df.itertuples(index=False):
        kp_store[row.track_id][row.frame_id][row.landmark_idx] = (
            row.x, row.y, row.visibility
        )
    return kp_store


def run_classifier(df: pd.DataFrame, fps: float = 30.0):
    kp_store = build_keypoint_dicts(df)

    all_track_ids = sorted(kp_store.keys())
    all_frames    = sorted(df["frame_id"].unique())

    records = []   # per-frame results

    print("[INFO] Running classifiers ...")
    for tid in all_track_ids:
        track_frames = sorted(kp_store[tid].keys())

        # Rolling wrist-y buffers for stimming detection
        wrist_y_left  = []
        wrist_y_right = []

        # Cumulative agitation score
        agitation_score = 0.0

        prev_kps = None

        for fid in track_frames:
            curr_kps = kp_store[tid][fid]

            # ── 1. Hand Raising ──
            hand_raised = detect_hand_raised(curr_kps)

            # ── 2. Posture ──
            posture = classify_posture(curr_kps)

            # ── 3. Agitation ──
            frame_speed = 0.0
            if prev_kps is not None:
                frame_speed = compute_agitation(curr_kps, prev_kps)
                if frame_speed > AGITATION_SPEED_THR:
                    agitation_score += 1.0

            # ── 4. Stimming — update wrist y buffers ──
            _, lwy, lwv = _kp(curr_kps, KP["left_wrist"])
            _, rwy, rwv = _kp(curr_kps, KP["right_wrist"])

            wrist_y_left.append(lwy  if _valid(lwv) else np.nan)
            wrist_y_right.append(rwy if _valid(rwv) else np.nan)

            # Keep buffer bounded
            if len(wrist_y_left)  > STIM_WINDOW_FRAMES * 2:
                wrist_y_left  = wrist_y_left[-STIM_WINDOW_FRAMES:]
                wrist_y_right = wrist_y_right[-STIM_WINDOW_FRAMES:]

            stim_left  = detect_stimming(np.array(wrist_y_left))
            stim_right = detect_stimming(np.array(wrist_y_right))
            is_stimming = stim_left or stim_right

            records.append({
                "frame_id":        fid,
                "track_id":        tid,
                "is_hand_raised":  hand_raised,
                "posture":         posture,
                "frame_speed_px":  round(frame_speed, 3),
                "agitation_score": round(agitation_score, 1),
                "is_stimming":     is_stimming,
            })

            prev_kps = curr_kps

    results_df = pd.DataFrame(records)
    results_df = results_df.sort_values(["track_id", "frame_id"]).reset_index(drop=True)
    return results_df


def build_summary(results_df: pd.DataFrame, fps: float = 30.0) -> pd.DataFrame:
    """Aggregate per-track summary statistics."""
    summaries = []
    for tid, grp in results_df.groupby("track_id"):
        n_frames      = len(grp)
        duration_sec  = n_frames / fps

        hand_raise_pct = grp["is_hand_raised"].mean() * 100
        stim_pct       = grp["is_stimming"].mean() * 100
        posture_counts = grp["posture"].value_counts(normalize=True) * 100

        summaries.append({
            "track_id":              tid,
            "total_frames":          n_frames,
            "duration_sec":          round(duration_sec, 2),
            "hand_raise_pct":        round(hand_raise_pct, 2),
            "posture_sitting_pct":   round(posture_counts.get("sitting",   0), 2),
            "posture_standing_pct":  round(posture_counts.get("standing",  0), 2),
            "posture_slouching_pct": round(posture_counts.get("slouching", 0), 2),
            "posture_unknown_pct":   round(posture_counts.get("unknown",   0), 2),
            "dominant_posture":      grp["posture"].mode()[0],
            "final_agitation_score": round(grp["agitation_score"].iloc[-1], 1),
            "agitation_per_min":     round(grp["agitation_score"].iloc[-1] / max(duration_sec / 60, 1e-6), 2),
            "stimming_pct":          round(stim_pct, 2),
            "mean_frame_speed_px":   round(grp["frame_speed_px"].mean(), 3),
        })

    return pd.DataFrame(summaries).sort_values("track_id").reset_index(drop=True)


def print_summary_table(summary_df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("BEHAVIOR SUMMARY PER CHILD (track_id)")
    print("=" * 70)
    for _, row in summary_df.iterrows():
        print(f"\n  Track ID : {int(row['track_id'])}")
        print(f"  Duration : {row['duration_sec']}s  ({int(row['total_frames'])} frames)")
        print(f"  ✋ Hand Raised    : {row['hand_raise_pct']:.1f}% of frames")
        print(f"  🧍 Posture        : {row['dominant_posture'].upper()} "
              f"(sit={row['posture_sitting_pct']:.0f}% "
              f"stand={row['posture_standing_pct']:.0f}% "
              f"slouch={row['posture_slouching_pct']:.0f}%)")
        print(f"  ⚡ Agitation score: {row['final_agitation_score']} "
              f"({row['agitation_per_min']:.1f}/min)")
        print(f"  🔄 Stimming       : {row['stimming_pct']:.1f}% of frames")
    print("=" * 70 + "\n")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Behavior classifier from raw_body_multi.csv")
    parser.add_argument("--input",  default="raw_body_multi.csv",        help="Path to input CSV")
    parser.add_argument("--out_frames",   default="behavior_per_frame.csv",  help="Per-frame output CSV")
    parser.add_argument("--out_summary",  default="behavior_summary.csv",    help="Summary output CSV")
    parser.add_argument("--fps",    type=float, default=30.0,            help="Video FPS (for duration calc)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        return

    # Load raw data
    df = load_csv(args.input)

    # Run classifiers
    results_df = run_classifier(df, fps=args.fps)

    # Build summary
    summary_df = build_summary(results_df, fps=args.fps)

    # Save outputs
    results_df.to_csv(args.out_frames,  index=False)
    summary_df.to_csv(args.out_summary, index=False)

    print(f"[OK] Per-frame results → {args.out_frames}  ({len(results_df):,} rows)")
    print(f"[OK] Summary           → {args.out_summary} ({len(summary_df)} tracks)")

    print_summary_table(summary_df)


if __name__ == "__main__":
    main()
