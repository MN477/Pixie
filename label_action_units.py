"""
label_action_units.py
=====================
Reads raw_action_units_multi.csv (OpenFace output), applies temporal smoothing
(median filter), detects compound behavioral expressions, computes a continuous
expressiveness score, and outputs labeled CSVs for downstream behavioral analysis.

Detected Expressions:
  Genuine_Smile          AU06 + AU12               Engagement / positive affect
  Fatigue_Indicator      AU45 rolling + AU05 inv + AU15 (+AU01)  Drowsiness
  Yawning                AU25 + AU26 + AU27         Strong fatigue confirmation
  Talking_Flag           AU25 + AU26 sustained      Speech contamination filter
  Expressiveness_Score   10 AUs normalized           Withdrawal / depression indicator
  Collective_Smile_Event 3+ tracks smiling           Social engagement

Outputs:
  labeled_action_units_multi.csv
  collective_events.csv
"""

import pandas as pd
import numpy as np
import os
import time

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

# Input / Output
INPUT_CSV  = "raw_action_units_multi.csv"
HEAD_POSE_CSV = "labeled_head_pose_multi.csv"
OUTPUT_CSV = "labeled_action_units_multi.csv"
EVENTS_CSV = "collective_events.csv"

# FPS — set manually, no auto-estimation.  Fallback: 30 FPS.
FPS = 30

# OpenFace confidence threshold
OPENFACE_CONFIDENCE_THRESH = 0.85

# Median smoothing window (frames)
MEDIAN_WINDOW = 5

# Duration thresholds (seconds) — converted to frames using FPS
SMILE_MIN_DURATION       = 0.5   # seconds
FATIGUE_MIN_DURATION     = 2.0   # seconds
YAWNING_MIN_DURATION     = 0.2   # seconds (lowered from 0.3)
TALKING_MIN_DURATION     = 0.5   # seconds

# Collective smile event window (seconds)
COLLECTIVE_WINDOW_SEC = 5.0
COLLECTIVE_MIN_TRACKS = 3

# Confidence scaling margin (AU intensity units above threshold for max confidence)
CONFIDENCE_MARGIN = 1.5

# ── Genuine Smile ──
SMILE_AU06_THRESH = 1.0
SMILE_AU12_THRESH = 1.0

# ── Fatigue ──
FATIGUE_AU45_ROLLING_THRESH = 0.3   # rolling mean of AU45_r
FATIGUE_AU05_UPPER_THRESH   = 0.5   # upper lid NOT raised (< this)
FATIGUE_AU15_THRESH         = 0.5   # lip corner depressor
FATIGUE_AU01_THRESH         = 1.0   # optional booster

# ── Yawning ──
YAWNING_AU25_THRESH = 1.5    # high intensity jaw drop onset
YAWNING_AU26_THRESH = 1.5    # high intensity jaw drop
YAWNING_AU27_THRESH = 1.0    # mouth stretch — high intensity

# ── Talking Flag ──
TALKING_AU25_THRESH = 0.5    # lips part
TALKING_AU26_THRESH = 0.5    # jaw drop

# ── Expressiveness Score ──
# AU25 removed — it activates during speech and inflates the metric
EXPRESSIVENESS_AUS = [
    "AU06_r", "AU12_r", "AU04_r", "AU07_r", "AU09_r",
    "AU10_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r"
]
EXPRESSIVENESS_ACTIVITY_THRESH = 1.0   # AU_r must exceed this to count


# ──────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────

def scale_confidence(value, threshold, margin=CONFIDENCE_MARGIN):
    """0.5 at the threshold boundary, scales linearly to 1.0 at threshold + margin."""
    if value <= threshold:
        return 0.0
    excess = value - threshold
    return min(1.0, 0.5 + 0.5 * (excess / margin))


def inverse_scale_confidence(value, threshold, margin=CONFIDENCE_MARGIN):
    """Confidence increases as value DECREASES below threshold.
    1.0 when value == 0, 0.5 at threshold, 0.0 above threshold."""
    if value >= threshold:
        return 0.0
    deficit = threshold - value
    return min(1.0, 0.5 + 0.5 * (deficit / margin))


def enforce_min_duration(series, min_frames):
    """Suppress True runs shorter than min_frames.
    Returns a new boolean Series with only sustained activations kept."""
    if min_frames <= 1:
        return series.copy()
    result = series.copy()
    # Identify consecutive runs
    groups = (series != series.shift()).cumsum()
    for _, grp in series.groupby(groups):
        if grp.iloc[0] and len(grp) < min_frames:
            result.loc[grp.index] = False
    return result


def get_smoothed_col(col_name):
    """Return the expected smoothed column name for a raw AU column."""
    return col_name.replace("_r", "_r_smooth")


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

    # Load head pose reliability if available
    if os.path.exists(HEAD_POSE_CSV):
        print(f"Loading head pose data from {HEAD_POSE_CSV} for reliability checking...")
        try:
            hp_df = pd.read_csv(HEAD_POSE_CSV, usecols=["frame_id", "track_id", "openface_reliable"])
            hp_df = hp_df.rename(columns={"openface_reliable": "pose_reliable"})
            df = pd.merge(df, hp_df, on=["frame_id", "track_id"], how="left")
        except ValueError:
            print(f"[WARN] openface_reliable column missing in {HEAD_POSE_CSV}. Skipping pose reliability check.")
            df["pose_reliable"] = True
    else:
        print(f"[WARN] Head pose CSV ({HEAD_POSE_CSV}) not found. Skipping pose reliability check.")
        df["pose_reliable"] = True

    # Check required columns exist
    required_au_cols = [
        "AU01_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r",
        "AU10_r", "AU12_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r",
        "AU25_r", "AU26_r", "AU27_r", "AU45_r"
    ]
    # AU27_r may not exist in all OpenFace builds — handle gracefully
    au27_available = "AU27_r" in df.columns
    if not au27_available:
        print("[WARN] AU27_r not found in data — Yawning detection will use AU25+AU26 only.")
        required_au_cols = [c for c in required_au_cols if c != "AU27_r"]

    missing_cols = [c for c in required_au_cols if c not in df.columns]
    if missing_cols:
        print(f"[ERROR] Missing required AU columns: {missing_cols}")
        return

    if "confidence" not in df.columns:
        print("[ERROR] Missing 'confidence' column.")
        return

    # ── FPS ──
    fps = FPS
    print(f"Using FPS = {fps}")

    # Convert duration thresholds to frame counts
    smile_min_frames     = max(1, int(round(SMILE_MIN_DURATION * fps)))
    fatigue_min_frames   = max(1, int(round(FATIGUE_MIN_DURATION * fps)))
    yawning_min_frames   = max(1, int(round(YAWNING_MIN_DURATION * fps)))
    talking_min_frames   = max(1, int(round(TALKING_MIN_DURATION * fps)))

    print(f"  Smile min frames:     {smile_min_frames}")
    print(f"  Fatigue min frames:   {fatigue_min_frames}")
    print(f"  Yawning min frames:   {yawning_min_frames}")
    print(f"  Talking min frames:   {talking_min_frames}")

    # Identify all _r columns to smooth
    au_r_cols = [c for c in df.columns if c.endswith("_r")]

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
        reindexed["is_missing_original"] = reindexed["confidence"].isna()

        # Mark unreliable frames (confidence below threshold AND head pose reliable)
        pose_rel = reindexed["pose_reliable"].fillna(False).astype(bool) if "pose_reliable" in reindexed.columns else True
        reindexed["openface_reliable"] = (
            (reindexed["confidence"] >= OPENFACE_CONFIDENCE_THRESH) & pose_rel
        )

        # NaN-out AU intensity values for unreliable frames
        for col in au_r_cols:
            reindexed.loc[~reindexed["openface_reliable"] & ~reindexed["is_missing_original"], col] = np.nan

        # Median filter on all _r columns
        smooth_cols = {}
        for col in au_r_cols:
            smooth_name = get_smoothed_col(col)
            reindexed[smooth_name] = reindexed[col].rolling(
                MEDIAN_WINDOW, center=True, min_periods=1
            ).median()
            smooth_cols[col] = smooth_name

        # Final missing flag: frame has no usable data after smoothing
        reindexed["is_missing"] = reindexed[get_smoothed_col("AU06_r")].isna()

        reindexed["track_id"] = track_id
        reindexed = reindexed.reset_index()
        processed_tracks.append(reindexed)

    final_df = pd.concat(processed_tracks, ignore_index=True)
    final_df = final_df.sort_values(by=["frame_id", "track_id"]).reset_index(drop=True)

    # ──────────────────────────────────────────────
    # STEP 2: PER-FRAME EXPRESSION LABELING
    # ──────────────────────────────────────────────
    print("Applying per-frame expression rules...")

    n = len(final_df)

    # Pre-extract smoothed columns as arrays for vectorized operations
    au06 = final_df[get_smoothed_col("AU06_r")].values
    au12 = final_df[get_smoothed_col("AU12_r")].values
    au25 = final_df[get_smoothed_col("AU25_r")].values
    au26 = final_df[get_smoothed_col("AU26_r")].values
    au45 = final_df[get_smoothed_col("AU45_r")].values
    au05 = final_df[get_smoothed_col("AU05_r")].values
    au15 = final_df[get_smoothed_col("AU15_r")].values
    au01 = final_df[get_smoothed_col("AU01_r")].values

    if au27_available:
        au27 = final_df[get_smoothed_col("AU27_r")].values
    else:
        au27 = np.zeros(n)

    reliable = final_df["openface_reliable"].values
    missing  = final_df["is_missing"].values

    # ── 2a. Genuine Smile ──
    smile_raw = np.zeros(n, dtype=bool)
    smile_conf = np.zeros(n, dtype=float)
    mask = reliable & ~missing
    smile_hit = mask & (au06 > SMILE_AU06_THRESH) & (au12 > SMILE_AU12_THRESH)
    smile_raw[smile_hit] = True
    # Confidence: average of AU06 and AU12 scaled
    for i in np.where(smile_hit)[0]:
        c06 = scale_confidence(au06[i], SMILE_AU06_THRESH)
        c12 = scale_confidence(au12[i], SMILE_AU12_THRESH)
        smile_conf[i] = 0.5 * c06 + 0.5 * c12



    # ── 2c. Fatigue Indicator ──
    fatigue_raw = np.zeros(n, dtype=bool)
    fatigue_conf = np.zeros(n, dtype=float)

    # Compute rolling mean of AU45 per track for fatigue window
    final_df["_au45_rolling"] = (
        final_df.groupby("track_id")[get_smoothed_col("AU45_r")]
        .transform(lambda x: x.rolling(fatigue_min_frames, min_periods=1, center=True).mean())
    )
    au45_roll = final_df["_au45_rolling"].values

    fatigue_hit = (
        mask
        & (au45_roll > FATIGUE_AU45_ROLLING_THRESH)
        & (au05 < FATIGUE_AU05_UPPER_THRESH)
        & (au15 > FATIGUE_AU15_THRESH)
    )
    fatigue_raw[fatigue_hit] = True
    for i in np.where(fatigue_hit)[0]:
        c45 = scale_confidence(au45_roll[i], FATIGUE_AU45_ROLLING_THRESH)
        c15 = scale_confidence(au15[i], FATIGUE_AU15_THRESH)
        c05 = inverse_scale_confidence(au05[i], FATIGUE_AU05_UPPER_THRESH)
        c01 = scale_confidence(au01[i], FATIGUE_AU01_THRESH) if au01[i] > FATIGUE_AU01_THRESH else 0.0
        if c01 > 0:
            fatigue_conf[i] = 0.30 * c45 + 0.25 * c15 + 0.20 * c05 + 0.25 * c01
        else:
            fatigue_conf[i] = 0.40 * c45 + 0.35 * c15 + 0.25 * c05

    # ── 2d. Yawning ──
    yawning_raw = np.zeros(n, dtype=bool)
    yawning_conf = np.zeros(n, dtype=float)
    if au27_available:
        yawning_hit = (
            mask
            & (au25 > YAWNING_AU25_THRESH)
            & (au26 > YAWNING_AU26_THRESH)
            & (au27 > YAWNING_AU27_THRESH)
        )
    else:
        # Fallback: use AU25 + AU26 at slightly higher thresholds if AU27 is missing
        yawning_hit = (
            mask
            & (au25 > YAWNING_AU25_THRESH * 1.1)
            & (au26 > YAWNING_AU26_THRESH * 1.1)
        )
    yawning_raw[yawning_hit] = True
    for i in np.where(yawning_hit)[0]:
        c25 = scale_confidence(au25[i], YAWNING_AU25_THRESH)
        c26 = scale_confidence(au26[i], YAWNING_AU26_THRESH)
        if au27_available:
            c27 = scale_confidence(au27[i], YAWNING_AU27_THRESH)
            yawning_conf[i] = 0.30 * c25 + 0.30 * c26 + 0.40 * c27
        else:
            yawning_conf[i] = 0.50 * c25 + 0.50 * c26

    # ── 2e. Talking Flag ──
    talking_raw = np.zeros(n, dtype=bool)
    talking_conf = np.zeros(n, dtype=float)
    talking_hit = (
        mask
        & (au25 > TALKING_AU25_THRESH)
        & (au26 > TALKING_AU26_THRESH)
    )
    talking_raw[talking_hit] = True
    for i in np.where(talking_hit)[0]:
        c25 = scale_confidence(au25[i], TALKING_AU25_THRESH)
        c26 = scale_confidence(au26[i], TALKING_AU26_THRESH)
        talking_conf[i] = 0.50 * c25 + 0.50 * c26



    # ── 2g. Expressiveness Score (per-frame, no duration filter) ──
    expressiveness = np.zeros(n, dtype=float)
    for col_name in EXPRESSIVENESS_AUS:
        smooth_name = get_smoothed_col(col_name)
        if smooth_name in final_df.columns:
            vals = final_df[smooth_name].values
            expressiveness += (vals > EXPRESSIVENESS_ACTIVITY_THRESH).astype(float)
    expressiveness = expressiveness / len(EXPRESSIVENESS_AUS)
    # Zero out missing/unreliable
    expressiveness[~mask] = 0.0

    # ──────────────────────────────────────────────
    # STEP 3: ENFORCE MINIMUM DURATION PER TRACK
    # ──────────────────────────────────────────────
    print("Enforcing minimum-duration thresholds per track...")

    final_df["genuine_smile"]        = smile_raw
    final_df["smile_confidence"]     = smile_conf
    final_df["fatigue_indicator"]    = fatigue_raw
    final_df["fatigue_confidence"]   = fatigue_conf
    final_df["yawning"]              = yawning_raw
    final_df["yawning_confidence"]   = yawning_conf
    final_df["talking_flag"]         = talking_raw
    final_df["talking_confidence"]   = talking_conf
    final_df["expressiveness_score"] = expressiveness

    # Duration enforcement must be done per-track
    label_duration_map = {
        "genuine_smile":        (smile_min_frames,     "smile_confidence"),
        "fatigue_indicator":    (fatigue_min_frames,    "fatigue_confidence"),
        "yawning":              (yawning_min_frames,    "yawning_confidence"),
        "talking_flag":         (talking_min_frames,    "talking_confidence"),
    }

    for track_id in final_df["track_id"].unique():
        track_mask = final_df["track_id"] == track_id
        for label_col, (min_frames, conf_col) in label_duration_map.items():
            original = final_df.loc[track_mask, label_col].copy()
            filtered = enforce_min_duration(original, min_frames)
            # Zero out confidence where label was suppressed
            suppressed = original & ~filtered
            final_df.loc[track_mask, label_col] = filtered
            final_df.loc[suppressed.index[suppressed], conf_col] = 0.0

    # ──────────────────────────────────────────────
    # STEP 4: COLLECTIVE SMILE EVENTS
    # ──────────────────────────────────────────────
    print("Detecting collective smile events...")

    collective_window_frames = max(1, int(round(COLLECTIVE_WINDOW_SEC * fps)))
    min_frame_global = final_df["frame_id"].min()
    max_frame_global = final_df["frame_id"].max()

    collective_events = []
    for win_start in range(int(min_frame_global), int(max_frame_global) + 1, collective_window_frames):
        win_end = win_start + collective_window_frames - 1
        window_data = final_df[
            (final_df["frame_id"] >= win_start)
            & (final_df["frame_id"] <= win_end)
            & (final_df["genuine_smile"] == True)
        ]
        smiling_tracks = window_data["track_id"].unique()
        if len(smiling_tracks) >= COLLECTIVE_MIN_TRACKS:
            collective_events.append({
                "window_start_frame": win_start,
                "window_end_frame": win_end,
                "track_ids_smiling": ",".join(str(t) for t in sorted(smiling_tracks)),
                "num_tracks_smiling": len(smiling_tracks),
                "event_type": "collective_smile"
            })

    events_df = pd.DataFrame(collective_events, columns=[
        "window_start_frame", "window_end_frame",
        "track_ids_smiling", "num_tracks_smiling", "event_type"
    ])
    events_df.to_csv(EVENTS_CSV, index=False)
    print(f"  → {len(events_df)} collective event(s) saved to {EVENTS_CSV}")

    # ──────────────────────────────────────────────
    # STEP 5: FORMAT & SAVE OUTPUT
    # ──────────────────────────────────────────────

    # Round confidence columns
    for col in ["smile_confidence", "fatigue_confidence",
                "yawning_confidence", "talking_confidence",
                "expressiveness_score"]:
        final_df[col] = final_df[col].round(4)

    out_cols = [
        "frame_id", "track_id",
        "genuine_smile",        "smile_confidence",
        "fatigue_indicator",    "fatigue_confidence",
        "yawning",              "yawning_confidence",
        "talking_flag",         "talking_confidence",
        "expressiveness_score",
        "openface_reliable", "is_missing"
    ]
    output_df = final_df[out_cols].copy()
    output_df.to_csv(OUTPUT_CSV, index=False)

    # ──────────────────────────────────────────────
    # STEP 6: SUMMARY STATISTICS
    # ──────────────────────────────────────────────
    reliable_count = output_df["openface_reliable"].sum()
    total_count = len(output_df)

    print("\n" + "─" * 50)
    print("LABEL DISTRIBUTION SUMMARY")
    print("─" * 50)
    print(f"  Total frames:    {total_count}")
    print(f"  Reliable frames: {reliable_count} ({100*reliable_count/max(1,total_count):.1f}%)")
    print(f"  Missing frames:  {output_df['is_missing'].sum()}")
    print()

    label_cols = {
        "Genuine Smile":        "genuine_smile",
        "Fatigue Indicator":    "fatigue_indicator",
        "Yawning":              "yawning",
        "Talking Flag":         "talking_flag",
    }

    for label_name, col in label_cols.items():
        count = output_df[col].sum()
        pct = 100 * count / max(1, reliable_count)
        print(f"  {label_name:25s}  {int(count):5d} frames  ({pct:5.1f}% of reliable)")

    expr_mean = output_df.loc[output_df["openface_reliable"], "expressiveness_score"].mean()
    expr_std  = output_df.loc[output_df["openface_reliable"], "expressiveness_score"].std()
    print(f"\n  {'Expressiveness (mean±std)':25s}  {expr_mean:.3f} ± {expr_std:.3f}")

    # Per-track breakdown
    print("\n  Per-Track Breakdown:")
    for track_id in sorted(output_df["track_id"].unique()):
        t = output_df[output_df["track_id"] == track_id]
        t_rel = t[t["openface_reliable"]]
        parts = []
        for label_name, col in label_cols.items():
            parts.append(f"{label_name.split()[0].lower()}={int(t[col].sum())}")
        expr_m = t_rel["expressiveness_score"].mean() if len(t_rel) > 0 else 0.0
        parts.append(f"expr_mean={expr_m:.3f}")
        print(f"    Track {track_id}: {', '.join(parts)}")

    print(f"\n  Collective Smile Events: {len(events_df)} window(s) flagged")
    print("─" * 50)

    print(f"\nData saved to {OUTPUT_CSV}")

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds\n")


if __name__ == "__main__":
    main()
