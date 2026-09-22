"""
scratch/verify_repaired_waveforms.py

Step 1-3 verification:
1. Confirm recoverability: fLaC header, file size vs siblings, ffmpeg decode test.
2. Verify audio content of repaired files: duration, RMS, max amplitude, not silence.
3. Build temporary metadata CSV pointing to repaired files, import process_feature_extraction,
   and test extraction with feature_type='both' into scratch output directory.
"""

import sys
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.extract_features import process_feature_extraction

SUBSET_CSV = PROJECT_ROOT / "data" / "subset_combined_metadata.csv"
FEATURES_CSV = PROJECT_ROOT / "data" / "features_metadata.csv"
REPAIRED_DIR = PROJECT_ROOT / "data" / "subset_raw_repaired"
RAW_DIR = PROJECT_ROOT / "data" / "subset_raw"
TEMP_METADATA_CSV = PROJECT_ROOT / "scratch" / "metadata_repaired_check.csv"
TEMP_FEATURES_DIR = PROJECT_ROOT / "scratch" / "features_repaired_check"
TEMP_FEATURES_META = PROJECT_ROOT / "scratch" / "features_metadata_repaired_check.csv"

try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_EXE = shutil.which("ffmpeg")

print("=" * 75)
print("STEP 1: RECOVERABILITY AUDIT OF 136 FAILING FILES")
print("=" * 75)

df_subset = pd.read_csv(SUBSET_CSV)
df_features = pd.read_csv(FEATURES_CSV)

successful_fps = set(df_features["subset_filepath"].dropna().unique())
failing_rows = df_subset[~df_subset["subset_filepath"].isin(successful_fps)].copy()

print(f"Total failing files identified: {len(failing_rows)}")

failing_filenames = set(Path(f).name for f in failing_rows["subset_filepath"])
working_files = [p for p in RAW_DIR.glob("*.flac") if p.name not in failing_filenames]
working_sizes = [p.stat().st_size for p in working_files]
print(f"Working sibling files in data/subset_raw/: {len(working_files)}")
if working_sizes:
    print(f"  Sibling size stats: min={min(working_sizes)} B, max={max(working_sizes)} B, mean={np.mean(working_sizes):.0f} B")

# Audit each failing file
valid_flac_headers = 0
ffmpeg_decodable = 0
ffmpeg_failing = 0
failing_sizes = []

for idx, row in failing_rows.iterrows():
    p = PROJECT_ROOT / row["subset_filepath"]
    failing_sizes.append(p.stat().st_size)

    with open(p, "rb") as f:
        magic = f.read(4)
    if magic == b"fLaC":
        valid_flac_headers += 1

    cmd = [FFMPEG_EXE, "-v", "error", "-i", str(p), "-f", "null", "-"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and not res.stderr.strip():
        ffmpeg_decodable += 1
    else:
        ffmpeg_failing += 1

print(f"\nAudit Results for 136 Failing Files:")
print(f"  Valid fLaC Magic Header     : {valid_flac_headers} / {len(failing_rows)}")
print(f"  Failing file size stats     : min={min(failing_sizes)} B, max={max(failing_sizes)} B, mean={np.mean(failing_sizes):.0f} B")
print(f"  FFmpeg Clean Decode ('OK')  : {ffmpeg_decodable} / {len(failing_rows)} -> RECOVERABLE")
print(f"  FFmpeg Decode Errors/Failed : {ffmpeg_failing} / {len(failing_rows)} -> TRULY UNRECOVERABLE")

print("=" * 75)
print("STEP 2: AUDIO CONTENT & WAVEFORM VERIFICATION ON REPAIRED COPIES")
print("=" * 75)

REPAIRED_DIR.mkdir(parents=True, exist_ok=True)

# Ensure all 136 are re-encoded into data/subset_raw_repaired/
for idx, row in failing_rows.iterrows():
    src_p = PROJECT_ROOT / row["subset_filepath"]
    dst_p = REPAIRED_DIR / src_p.name
    if not dst_p.exists() or dst_p.stat().st_size == 0:
        cmd_repair = [FFMPEG_EXE, "-y", "-loglevel", "error", "-i", str(src_p), "-c:a", "flac", str(dst_p)]
        subprocess.run(cmd_repair, check=True)

durations = []
max_amps = []
rms_values = []
zero_sound_count = 0
soundfile_readable = 0

for idx, row in failing_rows.iterrows():
    p = REPAIRED_DIR / Path(row["subset_filepath"]).name
    try:
        data, sr = sf.read(str(p), dtype="float32")
        soundfile_readable += 1
        dur = len(data) / sr
        durations.append(dur)
        max_val = float(np.max(np.abs(data)))
        rms_val = float(np.sqrt(np.mean(data ** 2)))
        max_amps.append(max_val)
        rms_values.append(rms_val)

        if max_val < 1e-4 or rms_val < 1e-5:
            zero_sound_count += 1
    except Exception as e:
        print(f"  Error reading {p.name}: {e}")

print(f"Repaired files readable by soundfile: {soundfile_readable} / {len(failing_rows)}")
print(f"Duration stats across 136 repaired  : min={min(durations):.2f}s, max={max(durations):.2f}s, mean={np.mean(durations):.2f}s")
print(f"Peak amplitude stats (max |x|)      : min={min(max_amps):.4f}, max={max(max_amps):.4f}, mean={np.mean(max_amps):.4f}")
print(f"RMS Energy stats                    : min={min(rms_values):.4f}, max={max(rms_values):.4f}, mean={np.mean(rms_values):.4f}")
print(f"Silent / Flatline audio files (<1e-4): {zero_sound_count}")

assert zero_sound_count == 0, "Warning: Found silent or flatlined repaired files!"
assert soundfile_readable == len(failing_rows), "Warning: Some repaired files are not readable by soundfile!"

print("\nAll 136 repaired files contain legitimate, non-silent speech audio signals.")

print("=" * 75)
print("STEP 3: EXTRACTION DRY RUN AGAINST TEMPORARY METADATA CSV")
print("=" * 75)

# Build temporary metadata CSV pointing to repaired copies for the 136 rows
temp_df = df_subset.copy()
repaired_stems = set(p.name for p in REPAIRED_DIR.glob("*.flac"))

updated_count = 0
for idx, row in temp_df.iterrows():
    p_orig = Path(row["subset_filepath"])
    if p_orig.name in repaired_stems:
        # Repoint to repaired file
        temp_df.at[idx, "subset_filepath"] = f"data/subset_raw_repaired/{p_orig.name}"
        updated_count += 1

temp_df.to_csv(TEMP_METADATA_CSV, index=False)
print(f"Temporary metadata CSV created at: {TEMP_METADATA_CSV}")
print(f"Rows repointed to data/subset_raw_repaired/: {updated_count}")

# Run process_feature_extraction via direct import
res = process_feature_extraction(
    input_csv=str(TEMP_METADATA_CSV),
    output_dir=str(TEMP_FEATURES_DIR),
    output_metadata_csv=str(TEMP_FEATURES_META),
    feature_type="both",
    n_mels=80,
    n_mfcc=20,
    max_frames=400,
    sample_rate=16000,
    use_deltas=False,
)

print("\n" + "=" * 75)
print("DRY RUN EXTRACTION METRICS")
print("=" * 75)
print(f"Total Input Rows Evaluated : {res['input_rows']}")
print(f"Audio Files Processed      : {res['processed_files']}")
print(f"Files Skipped / Errors     : {res['skipped_files']}")
print(f"Total Feature Records Saved: {res['total_records']}")

assert res["processed_files"] == 1200, f"Expected 1200, got {res['processed_files']}"
assert res["skipped_files"] == 0, f"Expected 0 skipped, got {res['skipped_files']}"
assert res["total_records"] == 2400, f"Expected 2400 (1200 logmel + 1200 mfcc), got {res['total_records']}"

print("\nDRY RUN SUCCESS: 100% of all 1,200 files extracted with ZERO errors!")
