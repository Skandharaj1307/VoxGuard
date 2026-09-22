"""
scratch/merge_repaired_files.py

Explicit, robust, per-file merge script:
1. Resolves absolute paths for data/subset_raw and data/subset_raw_repaired.
2. Enumerates and validates all 136 target filenames.
3. Checks source readability and clears destination read-only attributes on Windows if needed.
4. Performs per-file copy via shutil.copy2(), verifying immediately via MD5 hash and file size.
5. Emits verbose per-file verification output.
6. Upon 136/136 verified copies, runs the full feature extraction pipeline (feature_type='both').
"""

import os
import sys
import stat
import shutil
import hashlib
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = (PROJECT_ROOT / "data" / "subset_raw").resolve()
REPAIRED_DIR = (PROJECT_ROOT / "data" / "subset_raw_repaired").resolve()

print("=" * 80)
print("VoxGuard Repaired Audio Merge & Verification")
print("=" * 80)
print(f"Source Directory (Repaired) : {REPAIRED_DIR}")
print(f"Destination Directory (Raw) : {RAW_DIR}")
print(f"Source Directory Exists     : {REPAIRED_DIR.is_dir()}")
print(f"Destination Directory Exists: {RAW_DIR.is_dir()}")

if not REPAIRED_DIR.is_dir():
    print(f"FATAL: Source repaired directory does not exist: {REPAIRED_DIR}")
    sys.exit(1)

if not RAW_DIR.is_dir():
    print(f"FATAL: Destination raw directory does not exist: {RAW_DIR}")
    sys.exit(1)

# Enumerate 136 files
target_files = sorted([p.name for p in REPAIRED_DIR.glob("*.flac")])
print(f"\nTarget files found in {REPAIRED_DIR.name}: {len(target_files)}")

if len(target_files) != 136:
    print(f"FATAL: Expected exactly 136 files, but found {len(target_files)}. Aborting.")
    sys.exit(1)

print("\nStarting Per-File Copy & MD5 Verification:")
print("-" * 80)

success_count = 0
failed_count = 0
failures = []

for idx, filename in enumerate(target_files, 1):
    src_file = REPAIRED_DIR / filename
    dst_file = RAW_DIR / filename

    # Verify source readable
    if not src_file.is_file() or not os.access(src_file, os.R_OK):
        print(f"[{idx:3d}/136] [FAILED] {filename}: Source file not readable or missing")
        failed_count += 1
        failures.append((filename, "Source file not readable or missing"))
        continue

    # Clear Windows read-only attribute on destination if it exists
    if dst_file.exists():
        try:
            os.chmod(dst_file, stat.S_IWRITE)
        except Exception as e:
            pass

    try:
        # Calculate source MD5 and size
        src_bytes = src_file.read_bytes()
        src_md5 = hashlib.md5(src_bytes).hexdigest()
        src_size = len(src_bytes)

        # Copy with metadata preservation
        shutil.copy2(src_file, dst_file)

        # Ensure destination is writable
        try:
            os.chmod(dst_file, stat.S_IWRITE)
        except Exception:
            pass

        # Verify destination
        dst_bytes = dst_file.read_bytes()
        dst_md5 = hashlib.md5(dst_bytes).hexdigest()
        dst_size = len(dst_bytes)

        if src_md5 == dst_md5 and src_size == dst_size:
            print(f"[{idx:3d}/136] [OK] {filename} (Size: {dst_size} B, MD5: {dst_md5[:8]}...)")
            success_count += 1
        else:
            err_msg = f"Hash/size mismatch! src({src_size}B, {src_md5[:8]}) vs dst({dst_size}B, {dst_md5[:8]})"
            print(f"[{idx:3d}/136] [MISMATCH] {filename}: {err_msg}")
            failed_count += 1
            failures.append((filename, err_msg))

    except Exception as exc:
        print(f"[{idx:3d}/136] [FAILED] {filename}: {type(exc).__name__}: {exc}")
        failed_count += 1
        failures.append((filename, f"{type(exc).__name__}: {exc}"))

print("-" * 80)
print(f"Merge Complete: {success_count} Verified Successful | {failed_count} Failed")
print("-" * 80)

if failed_count > 0:
    print(f"\nFATAL: {failed_count} files failed to copy/verify. Listing failures:")
    for fname, err in failures:
        print(f"  - {fname}: {err}")
    print("\nAborting feature extraction pipeline run due to copy verification failures.")
    sys.exit(1)

print("\nAll 136 files successfully merged and MD5-verified into data/subset_raw/.")
print("\n" + "=" * 80)
print("RUNNING REAL FEATURE EXTRACTION PIPELINE (python dataset/extract_features.py --feature-type both)")
print("=" * 80 + "\n")

# Import and run directly to capture all execution details cleanly
from dataset.extract_features import process_feature_extraction

summary = process_feature_extraction(
    input_csv="data/subset_combined_metadata.csv",
    output_dir="data/features",
    output_metadata_csv="data/features_metadata.csv",
    feature_type="both",
    n_mels=80,
    n_mfcc=20,
    max_frames=400,
    sample_rate=16000,
    use_deltas=False,
)

print("\n" + "=" * 80)
print("FINAL PIPELINE EXECUTION SUMMARY")
print("=" * 80)
print(f"Total Input Rows Evaluated : {summary['input_rows']}")
print(f"Audio Files Processed      : {summary['processed_files']}")
print(f"Files Skipped / Errors     : {summary['skipped_files']}")
print(f"Rows with Unrecognized Split: {summary['unrecognized_splits']}")
print(f"Total Feature Records Saved: {summary['total_records']}")
print(f"Output Features Directory  : {summary['features_dir']}")
print(f"Output Features Metadata   : {summary['features_metadata']}")
print("=" * 80)
