"""
run_openface_batch.py
=====================
Batch-processes saved face crops through OpenFace 2.2.0 to extract
Action Unit (AU) intensities and binary classifications.

Pipeline:
  1. Runs OpenFace FaceLandmarkImg on the face_crops/ folder
  2. Parses the OpenFace output CSV
  3. Extracts frame_id and track_id from filenames (frame_000123_track_1.jpg)
  4. Produces a clean raw_action_units_multi.csv with frame_id, track_id, and all AUs

Usage:
    python run_openface_batch.py
"""

import csv
import os
import re
import shutil
import subprocess
import sys

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
OPENFACE_DIR = r"C:\Users\mouss\Documents\OpenFace_2.2.0_win_x86"
OPENFACE_EXE = os.path.join(OPENFACE_DIR, "FaceLandmarkImg.exe")

FACE_CROPS_DIR = "face_crops"
OPENFACE_OUTPUT_DIR = "openface_output"
FINAL_OUTPUT = "raw_action_units_multi.csv"

# Regex to extract frame_id and track_id from filenames like frame_000123_track_1.jpg
FILENAME_PATTERN = re.compile(r"frame_(\d+)_track_(\d+)")

# AU columns output by OpenFace (intensities _r and classifications _c)
AU_INTENSITY_COLS = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
    "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
    "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
]
AU_BINARY_COLS = [
    "AU01_c", "AU02_c", "AU04_c", "AU05_c", "AU06_c", "AU07_c",
    "AU09_c", "AU10_c", "AU12_c", "AU14_c", "AU15_c", "AU17_c",
    "AU20_c", "AU23_c", "AU25_c", "AU26_c", "AU28_c", "AU45_c",
]


def run_openface():
    """Run OpenFace FaceLandmarkImg on the face_crops folder."""

    if not os.path.isfile(OPENFACE_EXE):
        print(f"[ERROR] OpenFace executable not found: {OPENFACE_EXE}")
        sys.exit(1)

    if not os.path.isdir(FACE_CROPS_DIR):
        print(f"[ERROR] Face crops directory not found: {FACE_CROPS_DIR}")
        sys.exit(1)

    crop_files = [f for f in os.listdir(FACE_CROPS_DIR) if f.endswith((".jpg", ".png"))]
    if not crop_files:
        print(f"[ERROR] No face crop images found in {FACE_CROPS_DIR}")
        sys.exit(1)

    print(f"[INFO] Found {len(crop_files)} face crops to process")

    os.makedirs(OPENFACE_OUTPUT_DIR, exist_ok=True)

    # Run OpenFace on the directory of face crops
    cmd = [
        OPENFACE_EXE,
        "-fdir", os.path.abspath(FACE_CROPS_DIR),
        "-out_dir", os.path.abspath(OPENFACE_OUTPUT_DIR),
        "-aus",       # output Action Units
        "-multi_view", "1",  # single face per image
    ]

    print(f"[INFO] Running OpenFace: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=OPENFACE_DIR)

    if result.returncode != 0:
        print(f"[WARN] OpenFace exited with code {result.returncode}")
        if result.stderr:
            print(f"[WARN] stderr: {result.stderr[:500]}")
    else:
        print("[INFO] OpenFace completed successfully")


def parse_and_merge():
    """Parse OpenFace output CSVs and produce the final merged CSV."""

    if not os.path.isdir(OPENFACE_OUTPUT_DIR):
        print(f"[ERROR] OpenFace output directory not found: {OPENFACE_OUTPUT_DIR}")
        sys.exit(1)

    # OpenFace creates one CSV per input image
    of_csvs = [f for f in os.listdir(OPENFACE_OUTPUT_DIR) if f.endswith(".csv")]
    if not of_csvs:
        print(f"[ERROR] No OpenFace output CSVs found in {OPENFACE_OUTPUT_DIR}")
        sys.exit(1)

    print(f"[INFO] Found {len(of_csvs)} OpenFace output CSVs")

    # Collect all rows
    all_rows = []

    for csv_file in of_csvs:
        csv_path = os.path.join(OPENFACE_OUTPUT_DIR, csv_file)

        # Extract frame_id and track_id from the filename
        basename = os.path.splitext(csv_file)[0]
        match = FILENAME_PATTERN.search(basename)
        if not match:
            print(f"[WARN] Cannot parse frame/track from: {csv_file}, skipping")
            continue

        frame_id = int(match.group(1))
        track_id = int(match.group(2))

        # Read the OpenFace CSV
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # OpenFace columns have leading spaces — strip them
                cleaned = {k.strip(): v.strip() for k, v in row.items()}

                # Check if face was detected successfully
                confidence = float(cleaned.get("confidence", 0))
                success = int(cleaned.get("success", 0))

                out_row = {
                    "frame_id": frame_id,
                    "track_id": track_id,
                    "confidence": f"{confidence:.4f}",
                    "success": success,
                }

                if success:
                    for au_col in AU_INTENSITY_COLS + AU_BINARY_COLS:
                        out_row[au_col] = cleaned.get(au_col, "")
                else:
                    # Failed — fill AUs with None
                    for au_col in AU_INTENSITY_COLS + AU_BINARY_COLS:
                        out_row[au_col] = ""

                all_rows.append(out_row)

    # Sort by frame_id, then track_id
    all_rows.sort(key=lambda r: (r["frame_id"], r["track_id"]))

    # Write final CSV
    fieldnames = ["frame_id", "track_id", "confidence", "success"] + AU_INTENSITY_COLS + AU_BINARY_COLS

    with open(FINAL_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[INFO] Written {len(all_rows)} rows to {FINAL_OUTPUT}")


def main():
    print("=" * 60)
    print("OpenFace Batch Processing — Action Unit Extraction")
    print("=" * 60)

    try:
        # Step 1: Run OpenFace on face crops
        run_openface()

        # Step 2: Parse and merge into final CSV
        parse_and_merge()

        print(f"\n[DONE] Final output: {FINAL_OUTPUT}")
        print("  Columns: frame_id, track_id, confidence, success, AU01_r..AU45_r, AU01_c..AU45_c")
        print("  Join with raw_body_multi.csv / raw_head_pose_multi.csv by (frame_id, track_id)")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C).")

    finally:
        print("[INFO] Cleaning up temporary files...")
        if os.path.isdir(FACE_CROPS_DIR):
            shutil.rmtree(FACE_CROPS_DIR, ignore_errors=True)
            print(f"  ✓ Deleted {FACE_CROPS_DIR}/")
        if os.path.isdir(OPENFACE_OUTPUT_DIR):
            shutil.rmtree(OPENFACE_OUTPUT_DIR, ignore_errors=True)
            print(f"  ✓ Deleted {OPENFACE_OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
