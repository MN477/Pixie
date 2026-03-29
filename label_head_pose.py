"""
label_head_pose.py
==================
Reads the raw_head_pose_multi.csv output from 6DRepNet, applies temporal smoothing
(median filter), handles missing frames (interpolation), and maps the continuous
Euler angles into discrete compound labels and tilt labels, along with a fuzzy
confidence score for downstream behavioral analysis.

Outputs:
  labeled_head_pose_multi.csv
  Columns: frame_id, track_id, pitch_smooth, yaw_smooth, roll_smooth,
           pose_label, tilt_label, confidence, is_interpolated, is_missing
"""

import pandas as pd
import numpy as np
import os
import time

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
INPUT_CSV = "raw_head_pose_multi.csv"
OUTPUT_CSV = "labeled_head_pose_multi.csv"

# Smoothing & Interpolation
MEDIAN_WINDOW = 5       # Frames to smooth over (e.g., 5 frames = ~0.3s at 15fps) !!!!!!hard coded 
INTERPOLATE_LIMIT = 3   # Max consecutive missing frames to "guess" (interpolate) !!!!!!hard coded

# Hard Thresholds (Degrees)
PITCH_DOWN_THRESH =  15.0  # Pitch > this = Looking Down   !!!!!!hard coded
PITCH_UP_THRESH   = -15.0  # Pitch < this = Looking Up     !!!!!!hard coded

YAW_LEFT_THRESH   =  20.0  # Yaw > this = Turned Left      !!!!!!hard coded
YAW_RIGHT_THRESH  = -20.0  # Yaw < this = Turned Right      !!!!!!hard coded

ROLL_LEFT_THRESH  =  15.0  # Roll > this = Tilt-Left   !!!!!!hard coded
ROLL_RIGHT_THRESH = -15.0  # Roll < this = Tilt-Rightb   !!!!!!hard coded

# Margin (Degrees) past threshold required to reach maximum confidence (1.0)
MARGIN = 10.0           # !!!!!!hard coded


# ──────────────────────────────────────────────
# PROCESSING PIPELINE
# ──────────────────────────────────────────────
def classify_pose(row):
    """
    Maps continuous smoothed angles to discrete compound labels (pose_label),
    tilt labels (tilt_label), and calculates a fuzzy confidence score per frame.
    """
    if row["is_missing"]:
        return pd.Series(["Missing", "Missing", 0.0])
    
    pitch = row["pitch_smooth"]
    yaw = row["yaw_smooth"]
    roll = row["roll_smooth"]

    # 1. Pitch Classification & Confidence
    p_label = ""
    p_conf = 1.0
    if pitch > PITCH_DOWN_THRESH:
        p_label = "Up"
        p_conf = min(1.0, 0.5 + 0.5 * (pitch - PITCH_DOWN_THRESH) / MARGIN)
    elif pitch < PITCH_UP_THRESH:
        p_label = "Down"
        p_conf = min(1.0, 0.5 + 0.5 * (PITCH_UP_THRESH - pitch) / MARGIN)
    else:
        # Center pitch logic (scales from 1.0 at 0 degrees, down to 0.5 at threshold)
        if pitch > 0:
            p_conf = 1.0 - 0.5 * (pitch / PITCH_DOWN_THRESH)
        else:
            p_conf = 1.0 - 0.5 * (pitch / PITCH_UP_THRESH)

    # 2. Yaw Classification & Confidence
    y_label = ""
    y_conf = 1.0
    if yaw > YAW_LEFT_THRESH:
        y_label = "Right"
        y_conf = min(1.0, 0.5 + 0.5 * (yaw - YAW_LEFT_THRESH) / MARGIN)
    elif yaw < YAW_RIGHT_THRESH:
        y_label = "Left"
        y_conf = min(1.0, 0.5 + 0.5 * (YAW_RIGHT_THRESH - yaw) / MARGIN)
    else:
        # Center yaw logic
        if yaw > 0:
            y_conf = 1.0 - 0.5 * (yaw / YAW_LEFT_THRESH)
        else:
            y_conf = 1.0 - 0.5 * (yaw / YAW_RIGHT_THRESH)

    # Combine into Compound Pose Label
    if p_label and y_label:
        pose_label = f"{p_label}-{y_label}"
    elif p_label:
        pose_label = p_label
    elif y_label:
        pose_label = y_label
    else:
        pose_label = "Center"

    # 3. Roll (Tilt) Classification & Confidence
    t_label = "No-Tilt"
    t_conf = 1.0
    if roll > ROLL_LEFT_THRESH:
        t_label = "Tilt-Left"
        t_conf = min(1.0, 0.5 + 0.5 * (roll - ROLL_LEFT_THRESH) / MARGIN)
    elif roll < ROLL_RIGHT_THRESH:
        t_label = "Tilt-Right"
        t_conf = min(1.0, 0.5 + 0.5 * (ROLL_RIGHT_THRESH - roll) / MARGIN)
    else:
        if roll > 0:
            t_conf = 1.0 - 0.5 * (roll / ROLL_LEFT_THRESH)
        else:
            t_conf = 1.0 - 0.5 * (roll / ROLL_RIGHT_THRESH)

    # 4. Overall Confidence (Average of the three axes, weighted)
    avg_conf= (y_conf * 0.4) + (p_conf * 0.4) + (t_conf * 0.2)

    return pd.Series([pose_label, t_label, avg_conf])


def main():
    start_time = time.time()

    if not os.path.exists(INPUT_CSV):
        print(f"[ERROR] Input CSV not found: {INPUT_CSV}")
        return
    
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    if df.empty:
        print("[WARN] The CSV is empty. Exiting.")
        return

    processed_tracks = []
    grouped = df.groupby("track_id")
    
    print("Processing individual tracks (smoothing & interpolating)...")
    for track_id, group_df in grouped:
        # Setup continuous frame index
        min_frame = group_df["frame_id"].min()
        max_frame = group_df["frame_id"].max()
        group_df = group_df.set_index("frame_id")
        
        full_index = np.arange(min_frame, max_frame + 1)
        reindexed_df = group_df.reindex(full_index)
        
        # Determine tracking gaps
        reindexed_df["is_missing_original"] = reindexed_df["pitch"].isna()
        
        # 1. Median Filter BEFORE interpolation (ignores NaNs)
        # min_periods=1 allows smoothing of edges and ignores NaNs in the window
        reindexed_df["pitch_smooth"] = reindexed_df["pitch"].rolling(MEDIAN_WINDOW, center=True, min_periods=1).median()
        reindexed_df["yaw_smooth"]   = reindexed_df["yaw"].rolling(MEDIAN_WINDOW, center=True, min_periods=1).median()
        reindexed_df["roll_smooth"]  = reindexed_df["roll"].rolling(MEDIAN_WINDOW, center=True, min_periods=1).median()
        
        # 2. Interpolate the smoothed data across small gaps
        reindexed_df["pitch_smooth"] = reindexed_df["pitch_smooth"].interpolate(method="linear", limit=INTERPOLATE_LIMIT)
        reindexed_df["yaw_smooth"]   = reindexed_df["yaw_smooth"].interpolate(method="linear", limit=INTERPOLATE_LIMIT)
        reindexed_df["roll_smooth"]  = reindexed_df["roll_smooth"].interpolate(method="linear", limit=INTERPOLATE_LIMIT)
        
        # Flagging
        reindexed_df["is_missing"] = reindexed_df["pitch_smooth"].isna()
        
        # It's interpolated if it was originally missing but the smooth column now has data
        reindexed_df["is_interpolated"] = reindexed_df["is_missing_original"] & ~reindexed_df["is_missing"]
        
        # Clean up
        reindexed_df = reindexed_df.reset_index().rename(columns={"index": "frame_id"})
        reindexed_df["track_id"] = track_id
        
        processed_tracks.append(reindexed_df)

    final_df = pd.concat(processed_tracks, ignore_index=True)
    final_df = final_df.sort_values(by=["frame_id", "track_id"]).reset_index(drop=True)
    
    # 3. Apply Classification and Confidence Scoring
    print("Applying discrete classification thresholds ...")
    final_df[["pose_label", "tilt_label", "confidence"]] = final_df.apply(classify_pose, axis=1)
    
    # Format floating point numbers
    for col in ["pitch_smooth", "yaw_smooth", "roll_smooth"]:
        final_df[col] = final_df[col].round(4)
    final_df["confidence"] = final_df["confidence"].round(4)
    
    # Select and order output columns
    out_cols = [
        "frame_id", "track_id", 
        "pitch_smooth", "yaw_smooth", "roll_smooth", 
        "pose_label", "tilt_label", "confidence", 
        "is_interpolated", "is_missing"
    ]
    output_df = final_df[out_cols]
    
    # Save to disk
    output_df.to_csv(OUTPUT_CSV, index=False)
    
    print("-" * 40)
    print(f"Data saved to {OUTPUT_CSV}")
    print(f"Total rows: {len(output_df)}")
    print("\nPose Label Distribution:")
    print(output_df["pose_label"].value_counts().to_string())
    print("\nTilt Label Distribution:")
    print(output_df["tilt_label"].value_counts().to_string())
    print("-" * 40)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f" Total execution time: {execution_time:.2f} seconds\n")

if __name__ == "__main__":
    main()
