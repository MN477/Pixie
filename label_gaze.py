"""
label_gaze.py
=============
Reads raw_gaze_multi.csv (OpenFace eye-tracking output), applies temporal
smoothing (median filter), handles missing frames (interpolation), and maps
continuous gaze angles into discrete directional labels with fuzzy confidence
scores, gaze stability metrics, and eye-head divergence.

Outputs:
  labeled_gaze_multi.csv
  Columns: frame_id, track_id, gaze_angle_x_smooth, gaze_angle_y_smooth,
           gaze_h_label, h_confidence, gaze_v_label, v_confidence,
           gaze_stability, eye_head_divergence,
           is_interpolated, is_missing, openface_reliable
"""

import math
import os
import time

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
INPUT_CSV     = "raw_gaze_multi.csv"
HEAD_POSE_CSV = "labeled_head_pose_multi.csv"
OUTPUT_CSV    = "labeled_gaze_multi.csv"

# Unit conversion — OpenFace gaze is radians, head pose is degrees.
# This constant is used explicitly in the divergence formula so the
# conversion is never hidden inside arithmetic.
DEG_TO_RAD = math.pi / 180.0

# FPS — set manually, no auto-estimation.  Fallback: 30 FPS.
FPS = 30

# Smoothing & Interpolation
MEDIAN_WINDOW    = 5   # Frames to smooth over
INTERPOLATE_LIMIT = 3  # Max consecutive missing frames to interpolate

# OpenFace reliability
OPENFACE_CONFIDENCE_THRESH = 0.85

# Horizontal gaze thresholds (radians)
GAZE_H_CENTER_MAX =  0.15   # |x| <= this → Center
GAZE_H_MARGIN     =  0.15   # margin past threshold for full confidence

# Vertical gaze thresholds (radians)
GAZE_V_LEVEL_MAX  =  0.12   # |y| <= this → Level
GAZE_V_MARGIN     =  0.12   # margin past threshold for full confidence

# Gaze stability
STABILITY_WINDOW  = 15       # frames (~0.5s at 30fps)

# Duration enforcement (seconds) → converted to frames
GAZE_MIN_DURATION = 0.3      # labels must sustain this long

# Confidence scaling margin
CONFIDENCE_MARGIN = 0.15     # radians past threshold for max confidence


# ──────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────

def scale_confidence(value, threshold, margin=CONFIDENCE_MARGIN):
    """0.5 at the threshold boundary, scales linearly to 1.0 at threshold + margin."""
    excess = abs(value) - threshold
    if excess <= 0:
        # Inside center zone — confidence scales from 1.0 at 0 to 0.5 at threshold
        return 1.0 - 0.5 * (abs(value) / max(threshold, 1e-9))
    return min(1.0, 0.5 + 0.5 * (excess / max(margin, 1e-9)))


def enforce_min_duration(series, min_frames):
    """Suppress label runs shorter than min_frames.
    Returns a new Series with short runs replaced by 'Suppressed'."""
    if min_frames <= 1:
        return series.copy()
    result = series.copy()
    groups = (series != series.shift()).cumsum()
    for _, grp in series.groupby(groups):
        if len(grp) < min_frames:
            result.loc[grp.index] = "_suppressed"
    return result


# ──────────────────────────────────────────────
# CLASSIFICATION
# ──────────────────────────────────────────────

def classify_gaze(row):
    """
    Maps smoothed gaze angles to discrete direction labels + confidence.
    Returns: gaze_h_label, h_confidence, gaze_v_label, v_confidence
    """
    if row["is_missing"] or not row["openface_reliable"]:
        return pd.Series(["Missing", 0.0, "Missing", 0.0])

    gx = row["gaze_angle_x_smooth"]
    gy = row["gaze_angle_y_smooth"]

    # ── Horizontal ──
    if abs(gx) <= GAZE_H_CENTER_MAX:
        h_label = "Center"
    elif gx > GAZE_H_CENTER_MAX:
        h_label = "Right"
    else:
        h_label = "Left"
    h_conf = scale_confidence(gx, GAZE_H_CENTER_MAX, GAZE_H_MARGIN)

    # ── Vertical ──
    if abs(gy) <= GAZE_V_LEVEL_MAX:
        v_label = "Level"
    elif gy > GAZE_V_LEVEL_MAX:
        v_label = "Down"
    else:
        v_label = "Up"
    v_conf = scale_confidence(gy, GAZE_V_LEVEL_MAX, GAZE_V_MARGIN)

    return pd.Series([h_label, h_conf, v_label, v_conf])


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────

def main():
    start_time = time.time()

    # ── Validate Input ──
    if not os.path.exists(INPUT_CSV):
        print(f"[ERROR] Input CSV not found: {INPUT_CSV}")
        return

    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    if df.empty:
        print("[WARN] CSV is empty. Exiting.")
        return

    # Check required columns
    required_cols = ["frame_id", "track_id", "confidence", "success",
                     "gaze_angle_x", "gaze_angle_y"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"[ERROR] Missing required columns: {missing_cols}")
        return

    # ── Load head pose reliability ──
    has_head_pose = False
    if os.path.exists(HEAD_POSE_CSV):
        print(f"Loading head pose data from {HEAD_POSE_CSV}...")
        try:
            hp_df = pd.read_csv(HEAD_POSE_CSV, usecols=[
                "frame_id", "track_id", "openface_reliable",
                "yaw_smooth", "pitch_smooth"
            ])
            hp_df = hp_df.rename(columns={"openface_reliable": "pose_reliable"})
            df = pd.merge(df, hp_df, on=["frame_id", "track_id"], how="left")
            has_head_pose = True
        except ValueError as e:
            print(f"[WARN] Could not load head pose columns: {e}")
            df["pose_reliable"] = True
            df["yaw_smooth"] = np.nan
            df["pitch_smooth"] = np.nan
    else:
        print(f"[WARN] Head pose CSV ({HEAD_POSE_CSV}) not found. "
              "Skipping pose reliability and divergence.")
        df["pose_reliable"] = True
        df["yaw_smooth"] = np.nan
        df["pitch_smooth"] = np.nan

    # ── FPS ──
    fps = FPS
    gaze_min_frames = max(1, int(round(GAZE_MIN_DURATION * fps)))
    print(f"Using FPS = {fps}")
    print(f"  Gaze min duration frames: {gaze_min_frames}")

    # ──────────────────────────────────────────────
    # STEP 1: PER-TRACK PREPROCESSING
    # ──────────────────────────────────────────────
    processed_tracks = []
    grouped = df.groupby("track_id")
    print(f"\nProcessing {len(grouped)} track(s)...")

    for track_id, group_df in grouped:
        min_frame = group_df["frame_id"].min()
        max_frame = group_df["frame_id"].max()
        group_df = group_df.set_index("frame_id")

        # Reindex to continuous frame range
        full_index = np.arange(min_frame, max_frame + 1)
        reindexed = group_df.reindex(full_index)
        reindexed.index.name = "frame_id"

        # Mark original missing frames
        reindexed["is_missing_original"] = reindexed["gaze_angle_x"].isna()

        # ── STEP 2: RELIABILITY ──
        # success flag must be 1 AND confidence >= threshold AND head pose reliable
        pose_rel = reindexed["pose_reliable"].fillna(False).astype(bool)
        success_ok = reindexed["success"].fillna(0).astype(int) == 1
        conf_ok = reindexed["confidence"].fillna(0) >= OPENFACE_CONFIDENCE_THRESH

        reindexed["openface_reliable"] = success_ok & conf_ok & pose_rel

        # NaN-out gaze values for unreliable frames (but not originally missing)
        unreliable_mask = ~reindexed["openface_reliable"] & ~reindexed["is_missing_original"]
        reindexed.loc[unreliable_mask, "gaze_angle_x"] = np.nan
        reindexed.loc[unreliable_mask, "gaze_angle_y"] = np.nan

        # Median filter
        reindexed["gaze_angle_x_smooth"] = reindexed["gaze_angle_x"].rolling(
            MEDIAN_WINDOW, center=True, min_periods=1
        ).median()
        reindexed["gaze_angle_y_smooth"] = reindexed["gaze_angle_y"].rolling(
            MEDIAN_WINDOW, center=True, min_periods=1
        ).median()

        # Linear interpolation across small gaps
        reindexed["gaze_angle_x_smooth"] = reindexed["gaze_angle_x_smooth"].interpolate(
            method="linear", limit=INTERPOLATE_LIMIT
        )
        reindexed["gaze_angle_y_smooth"] = reindexed["gaze_angle_y_smooth"].interpolate(
            method="linear", limit=INTERPOLATE_LIMIT
        )

        # Final missing flag
        reindexed["is_missing"] = reindexed["gaze_angle_x_smooth"].isna()
        reindexed["is_interpolated"] = (
            reindexed["is_missing_original"] & ~reindexed["is_missing"]
        )

        # ── Gaze Stability ──
        roll_std_x = reindexed["gaze_angle_x_smooth"].rolling(
            STABILITY_WINDOW, center=True, min_periods=1
        ).std().fillna(0)
        roll_std_y = reindexed["gaze_angle_y_smooth"].rolling(
            STABILITY_WINDOW, center=True, min_periods=1
        ).std().fillna(0)
        reindexed["gaze_stability"] = 1.0 - np.clip(roll_std_x + roll_std_y, 0, 1)

        # ── Eye-Head Divergence ──
        # Convert head pose degrees → radians using the explicit constant
        head_yaw_rad   = reindexed["yaw_smooth"].fillna(0) * DEG_TO_RAD
        head_pitch_rad = reindexed["pitch_smooth"].fillna(0) * DEG_TO_RAD

        gx_s = reindexed["gaze_angle_x_smooth"].fillna(0)
        gy_s = reindexed["gaze_angle_y_smooth"].fillna(0)

        reindexed["eye_head_divergence"] = np.sqrt(
            (gx_s - head_yaw_rad) ** 2 + (gy_s - head_pitch_rad) ** 2
        )
        # Zero out divergence where data is missing or no head pose was loaded
        if not has_head_pose:
            reindexed["eye_head_divergence"] = np.nan

        reindexed["track_id"] = track_id
        reindexed = reindexed.reset_index()
        processed_tracks.append(reindexed)

    final_df = pd.concat(processed_tracks, ignore_index=True)
    final_df = final_df.sort_values(by=["frame_id", "track_id"]).reset_index(drop=True)

    # ──────────────────────────────────────────────
    # STEP 3: CLASSIFICATION
    # ──────────────────────────────────────────────
    print("Applying gaze classification thresholds...")
    final_df[["gaze_h_label", "h_confidence", "gaze_v_label", "v_confidence"]] = (
        final_df.apply(classify_gaze, axis=1)
    )

    # ──────────────────────────────────────────────
    # STEP 4: DURATION ENFORCEMENT
    # ──────────────────────────────────────────────
    print("Enforcing minimum-duration thresholds per track...")
    for track_id in final_df["track_id"].unique():
        mask = final_df["track_id"] == track_id

        # Horizontal
        original_h = final_df.loc[mask, "gaze_h_label"].copy()
        filtered_h = enforce_min_duration(original_h, gaze_min_frames)
        suppressed_h = filtered_h == "_suppressed"
        # Revert suppressed labels to "Center" (neutral) and zero confidence
        final_df.loc[mask & suppressed_h.values, "gaze_h_label"] = "Center"
        final_df.loc[mask & suppressed_h.values, "h_confidence"] = 0.0

        # Vertical
        original_v = final_df.loc[mask, "gaze_v_label"].copy()
        filtered_v = enforce_min_duration(original_v, gaze_min_frames)
        suppressed_v = filtered_v == "_suppressed"
        final_df.loc[mask & suppressed_v.values, "gaze_v_label"] = "Level"
        final_df.loc[mask & suppressed_v.values, "v_confidence"] = 0.0

    # ──────────────────────────────────────────────
    # STEP 5: FORMAT & SAVE
    # ──────────────────────────────────────────────

    # Round floats
    for col in ["gaze_angle_x_smooth", "gaze_angle_y_smooth",
                "h_confidence", "v_confidence",
                "gaze_stability", "eye_head_divergence"]:
        if col in final_df.columns:
            final_df[col] = final_df[col].round(4)

    out_cols = [
        "frame_id", "track_id",
        "gaze_angle_x_smooth", "gaze_angle_y_smooth",
        "gaze_h_label", "h_confidence",
        "gaze_v_label", "v_confidence",
        "gaze_stability", "eye_head_divergence",
        "is_interpolated", "is_missing", "openface_reliable",
    ]
    output_df = final_df[out_cols].copy()
    output_df.to_csv(OUTPUT_CSV, index=False)

    # ──────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────
    reliable_df = output_df[output_df["openface_reliable"]]
    total    = len(output_df)
    reliable = len(reliable_df)
    missing  = output_df["is_missing"].sum()

    print("\n" + "─" * 50)
    print("GAZE LABEL DISTRIBUTION SUMMARY")
    print("─" * 50)
    print(f"  Total frames:    {total}")
    print(f"  Reliable frames: {reliable} ({100*reliable/max(1,total):.1f}%)")
    print(f"  Missing frames:  {int(missing)}")

    print("\n  Horizontal Gaze:")
    for label in ["Center", "Right", "Left"]:
        count = (reliable_df["gaze_h_label"] == label).sum()
        pct = 100 * count / max(1, reliable)
        print(f"    {label:10s}  {int(count):5d} frames  ({pct:5.1f}%)")

    print("\n  Vertical Gaze:")
    for label in ["Level", "Down", "Up"]:
        count = (reliable_df["gaze_v_label"] == label).sum()
        pct = 100 * count / max(1, reliable)
        print(f"    {label:10s}  {int(count):5d} frames  ({pct:5.1f}%)")

    stab_mean = reliable_df["gaze_stability"].mean()
    stab_std  = reliable_df["gaze_stability"].std()
    print(f"\n  Stability (mean±std):        {stab_mean:.3f} ± {stab_std:.3f}")

    if has_head_pose:
        div_mean = reliable_df["eye_head_divergence"].mean()
        div_std  = reliable_df["eye_head_divergence"].std()
        print(f"  Eye-Head Divergence (mean±std): {div_mean:.3f} ± {div_std:.3f} rad")
    else:
        print("  Eye-Head Divergence: N/A (no head pose data)")

    # Per-track breakdown
    print("\n  Per-Track Breakdown:")
    for tid in sorted(output_df["track_id"].unique()):
        t = reliable_df[reliable_df["track_id"] == tid]
        h_center = (t["gaze_h_label"] == "Center").sum()
        v_level  = (t["gaze_v_label"] == "Level").sum()
        v_down   = (t["gaze_v_label"] == "Down").sum()
        stab     = t["gaze_stability"].mean() if len(t) > 0 else 0.0
        print(f"    Track {tid}: h_center={h_center}, v_level={v_level}, "
              f"v_down={v_down}, stab_mean={stab:.3f}")

    print("─" * 50)
    print(f"\nData saved to {OUTPUT_CSV}")

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds\n")


if __name__ == "__main__":
    main()
