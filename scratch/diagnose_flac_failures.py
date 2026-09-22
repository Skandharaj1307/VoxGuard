"""
scratch/diagnose_flac_failures.py

Diagnostic and repair script for the 136 failing FLAC audio files in VoxGuard:
1. Identifies the exact 136 failing subset_filepath values by comparing
   data/subset_combined_metadata.csv against data/features_metadata.csv.
2. Directly tests each file with soundfile.read(), capturing the raw exception message.
3. Directly tests each file with librosa.load(..., sr=None), capturing the exception message.
4. Inspects byte size and verifies the 4-byte 'fLaC' header signature.
5. Runs FFmpeg integrity decoding test (ffmpeg -v error -i <file> -f null -).
6. Outputs a comprehensive diagnostic report to scratch/flac_diagnosis_report.csv.
7. Analyzes clustering patterns (by split, label, attack_id, speaker_id, filename ID range).
8. Re-encodes recoverable files into data/subset_raw_repaired/ and verifies soundfile.read() on them.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

# Audio library imports
import soundfile as sf
import librosa

try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_EXE = shutil.which("ffmpeg")

if not FFMPEG_EXE:
    raise RuntimeError("FFmpeg executable not found!")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBSET_CSV = PROJECT_ROOT / "data" / "subset_combined_metadata.csv"
FEATURES_CSV = PROJECT_ROOT / "data" / "features_metadata.csv"
REPORT_CSV = PROJECT_ROOT / "scratch" / "flac_diagnosis_report.csv"
REPAIRED_DIR = PROJECT_ROOT / "data" / "subset_raw_repaired"

print("=" * 75)
print("VoxGuard Module 4: FLAC Diagnostic & Recovery Pipeline")
print("=" * 75)

# 1. Load metadata to find the 136 failing files
df_subset = pd.read_csv(SUBSET_CSV)
df_features = pd.read_csv(FEATURES_CSV)

# Group features by subset_filepath
successful_subset_fps = set(df_features["subset_filepath"].dropna().unique())

# Determine failing rows
failing_mask = ~df_subset["subset_filepath"].isin(successful_subset_fps)
df_failing = df_subset[failing_mask].copy()

print(f"Total rows in subset_combined_metadata: {len(df_subset)}")
print(f"Total unique files in features_metadata : {len(successful_subset_fps)}")
print(f"Identified failing subset files         : {len(df_failing)}")

if len(df_failing) != 136:
    print(f"NOTE: Expected 136 failing files, found {len(df_failing)}.")

# 2. Diagnose each file
records = []
recoverable_files = []

for idx, row in df_failing.iterrows():
    raw_fp_str = row["subset_filepath"]
    p = PROJECT_ROOT / raw_fp_str
    filename = p.name
    split = row.get("split", "unknown")
    label = row.get("label", "unknown")
    attack_id = row.get("attack_id", "unknown")
    attack_category = row.get("attack_category", "unknown")
    speaker_id = row.get("speaker_id", "unknown")

    # Check file existence and size
    if not p.is_file():
        file_size = 0
        has_flac_header = False
        sf_err = "File not found"
        librosa_err = "File not found"
        ffmpeg_err = "File not found"
    else:
        file_size = p.stat().st_size

        # Check fLaC magic header (bytes: 0x66, 0x4C, 0x61, 0x43 -> b'fLaC')
        try:
            with open(p, "rb") as f:
                header = f.read(4)
            has_flac_header = (header == b"fLaC")
        except Exception as e:
            has_flac_header = False

        # Test soundfile.read()
        try:
            audio_sf, sr_sf = sf.read(str(p))
            sf_err = "OK"
        except Exception as e:
            sf_err = f"{type(e).__name__}: {str(e).strip()}"

        # Test librosa.load(..., sr=None)
        try:
            audio_lib, sr_lib = librosa.load(str(p), sr=None, mono=False)
            librosa_err = "OK"
        except Exception as e:
            librosa_err = f"{type(e).__name__}: {str(e).strip()}"

        # Test ffmpeg integrity decode: ffmpeg -v error -i <file> -f null -
        cmd_ffmpeg = [FFMPEG_EXE, "-v", "error", "-i", str(p), "-f", "null", "-"]
        try:
            res_ff = subprocess.run(cmd_ffmpeg, capture_output=True, text=True)
            if res_ff.returncode == 0 and not res_ff.stderr.strip():
                ffmpeg_err = "OK"
            else:
                ffmpeg_err = res_ff.stderr.strip() if res_ff.stderr.strip() else f"Exit code {res_ff.returncode}"
        except Exception as e:
            ffmpeg_err = f"SubprocessError: {str(e)}"

    record = {
        "filename": filename,
        "split": split,
        "label": label,
        "attack_id": attack_id,
        "attack_category": attack_category,
        "speaker_id": speaker_id,
        "file_size_bytes": file_size,
        "has_valid_flac_header": has_flac_header,
        "soundfile_error": sf_err,
        "librosa_error": librosa_err,
        "ffmpeg_error": ffmpeg_err,
        "subset_filepath": raw_fp_str,
    }
    records.append(record)

    if ffmpeg_err == "OK":
        recoverable_files.append((p, filename))

# Save summary report
df_report = pd.DataFrame(records)
REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
df_report.to_csv(REPORT_CSV, index=False)
print(f"\nDiagnostic report saved to: {REPORT_CSV}")

# 3. Aggregate Statistical Findings
total_failing = len(df_report)
invalid_header_count = sum(~df_report["has_valid_flac_header"])
zero_or_tiny_size = sum(df_report["file_size_bytes"] < 1000)
min_size = df_report["file_size_bytes"].min()
max_size = df_report["file_size_bytes"].max()
mean_size = df_report["file_size_bytes"].mean()

ffmpeg_ok_count = sum(df_report["ffmpeg_error"] == "OK")
all_three_fail = sum((df_report["soundfile_error"] != "OK") & (df_report["librosa_error"] != "OK") & (df_report["ffmpeg_error"] != "OK"))

print("\n" + "=" * 75)
print("DIAGNOSTIC SUMMARY")
print("=" * 75)
print(f"Total Analyzed Failing Files : {total_failing}")
print(f"Invalid / Missing fLaC header: {invalid_header_count}")
print(f"Zero or Suspicious (<1KB) size: {zero_or_tiny_size}")
print(f"File size range              : min={min_size} bytes, max={max_size} bytes, mean={mean_size:.0f} bytes")
print(f"FFmpeg Decodes Cleanly ('OK'): {ffmpeg_ok_count} (Recoverable)")
print(f"Failed in ALL 3 Decoders     : {all_three_fail} (Truly Corrupt)")

# Display Soundfile error breakdown
print("\nSoundfile Error Categories:")
print(df_report["soundfile_error"].value_counts())

# Display Librosa error breakdown
print("\nLibrosa Error Categories:")
print(df_report["librosa_error"].value_counts())

# Display FFmpeg error breakdown
print("\nFFmpeg Error Categories:")
print(df_report["ffmpeg_error"].value_counts().head(10))

# Display Clustering Analysis
print("\n" + "=" * 75)
print("CLUSTERING ANALYSIS")
print("=" * 75)
print("\nBreakdown by Dataset / Partition (all from filename prefix):")
prefixes = df_report["filename"].apply(lambda f: f.split("_")[0] + "_" + f.split("_")[1])
print(prefixes.value_counts())

print("\nBreakdown by Split & Label:")
print(df_report.groupby(["split", "label"]).size())

print("\nBreakdown by Attack ID:")
print(df_report["attack_id"].value_counts())

print("\nBreakdown by Speaker ID (Top 10):")
print(df_report["speaker_id"].value_counts().head(10))

# Check filename ID ranges
def extract_numeric_id(fname):
    digits = "".join([c for c in fname if c.isdigit()])
    return int(digits) if digits else -1

ids = df_report["filename"].apply(extract_numeric_id)
print(f"\nNumeric ID range in filenames: min={ids.min()}, max={ids.max()}")

# 4. Perform Re-encoding for Recoverable Files
print("\n" + "=" * 75)
print("REPAIR ATTEMPT VIA FFMPEG")
print("=" * 75)

REPAIRED_DIR.mkdir(parents=True, exist_ok=True)
repaired_success = 0
repaired_sf_verified = 0

# Try re-encoding all files where ffmpeg can read or attempt repair
for idx, row in df_report.iterrows():
    src_p = PROJECT_ROOT / row["subset_filepath"]
    dst_p = REPAIRED_DIR / row["filename"]

    # Use ffmpeg -y -i <src> -c:a flac <dst>
    cmd_repair = [FFMPEG_EXE, "-y", "-loglevel", "error", "-i", str(src_p), "-c:a", "flac", str(dst_p)]
    res = subprocess.run(cmd_repair, capture_output=True, text=True)

    if res.returncode == 0 and dst_p.is_file() and dst_p.stat().st_size > 0:
        repaired_success += 1
        # Verify soundfile.read() on the repaired file
        try:
            data, sr = sf.read(str(dst_p))
            if len(data) > 0:
                repaired_sf_verified += 1
        except Exception:
            pass

print(f"FFmpeg Re-encoding attempted for : {len(df_report)} files")
print(f"FFmpeg Successfully Re-encoded   : {repaired_success} files into {REPAIRED_DIR.name}/")
print(f"Soundfile Verified on Re-encoded : {repaired_sf_verified} / {repaired_success} files")

# Print first 5 rows of report table
print("\nFirst 5 Rows of Diagnostic Report:")
cols_to_show = ["filename", "split", "label", "file_size_bytes", "has_valid_flac_header", "soundfile_error", "ffmpeg_error"]
print(df_report[cols_to_show].head(5).to_string(index=False))

print("\n" + "=" * 75)
print("DIAGNOSTIC COMPLETE")
print("=" * 75)
