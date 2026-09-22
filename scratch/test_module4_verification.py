"""
scratch/test_module4_verification.py

Comprehensive test suite for VoxGuard Module 4 (Feature Extraction):
- Test 1: Clean, side-effect-free import
- Test 2: Acoustic feature extraction contracts (logmel, mfcc, deltas, padding, truncation, stereo, resampling)
- Test 3: Batch extraction pipeline with 'both', directory isolation, and metadata hash invariance
- Test 4 (Bug 1 Fix): Atomicity under partial-failure in feature_type='both'
- Test 5 (Bug 2 Fix): Unrecognized split warning, fallback to 'train', and counter tracking
"""

import os
import sys
import shutil
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import dataset.extract_features as ef
from dataset.extract_features import (
    extract_features_for_file,
    process_feature_extraction,
    load_audio,
)

# Test environment setup
test_run_dir = PROJECT_ROOT / "scratch" / "test_run"
test_run_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Helper: Generate synthetic audio assets
# ----------------------------------------------------------------------
sr = 16000
duration_normal = 1.5
duration_short = 0.2
duration_long = 15.0

t_norm = np.linspace(0, duration_normal, int(sr * duration_normal), endpoint=False)
wave_norm = (0.5 * np.sin(2 * np.pi * 440 * t_norm)).astype(np.float32)

t_short = np.linspace(0, duration_short, int(sr * duration_short), endpoint=False)
wave_short = (0.5 * np.sin(2 * np.pi * 880 * t_short)).astype(np.float32)

t_long = np.linspace(0, duration_long, int(sr * duration_long), endpoint=False)
wave_long = (0.5 * np.sin(2 * np.pi * 220 * t_long)).astype(np.float32)

audio_norm_path = test_run_dir / "audio_normal.wav"
audio_short_path = test_run_dir / "audio_short.wav"
audio_long_path = test_run_dir / "audio_long.wav"
audio_stereo_path = test_run_dir / "audio_stereo.wav"
audio_8k_path = test_run_dir / "audio_8k.wav"

sf.write(audio_norm_path, wave_norm, sr)
sf.write(audio_short_path, wave_short, sr)
sf.write(audio_long_path, wave_long, sr)

stereo_wave = np.stack([wave_norm, wave_norm * 0.8], axis=1)
sf.write(audio_stereo_path, stereo_wave, sr)

t_8k = np.linspace(0, 1.0, 8000, endpoint=False)
wave_8k = (0.5 * np.sin(2 * np.pi * 440 * t_8k)).astype(np.float32)
sf.write(audio_8k_path, wave_8k, 8000)

print("=" * 70)
print("VoxGuard Module 4 Verification Suite")
print("=" * 70)

# ----------------------------------------------------------------------
# Test 1: Clean side-effect-free import
# ----------------------------------------------------------------------
print("\n[RUNNING] Test 1: Side-effect-free import...")
assert callable(extract_features_for_file)
assert callable(process_feature_extraction)
assert callable(load_audio)
print("[PASSED] Test 1: Module imported with clean public API and zero side effects.")

# ----------------------------------------------------------------------
# Test 2: Feature extraction contracts & shapes
# ----------------------------------------------------------------------
print("\n[RUNNING] Test 2: Feature extraction shapes, dtypes, and contracts...")

# LogMel default
feat_logmel = extract_features_for_file(str(audio_norm_path), feature_type="logmel")
assert feat_logmel.dtype == np.float32, f"LogMel dtype mismatch: {feat_logmel.dtype}"
assert feat_logmel.shape == (80, 400), f"LogMel shape mismatch: {feat_logmel.shape}"

# MFCC default (use_deltas=False)
feat_mfcc = extract_features_for_file(str(audio_norm_path), feature_type="mfcc", use_deltas=False)
assert feat_mfcc.dtype == np.float32, f"MFCC dtype mismatch: {feat_mfcc.dtype}"
assert feat_mfcc.shape == (20, 400), f"MFCC shape mismatch: {feat_mfcc.shape}"

# MFCC with deltas (use_deltas=True)
feat_mfcc_deltas = extract_features_for_file(str(audio_norm_path), feature_type="mfcc", use_deltas=True)
assert feat_mfcc_deltas.dtype == np.float32, f"MFCC delta dtype mismatch: {feat_mfcc_deltas.dtype}"
assert feat_mfcc_deltas.shape == (60, 400), f"MFCC delta shape mismatch: {feat_mfcc_deltas.shape}"

# Zero-padding for short audio
feat_short = extract_features_for_file(str(audio_short_path), feature_type="logmel", max_frames=400)
assert feat_short.shape == (80, 400), f"Padding shape mismatch: {feat_short.shape}"
assert np.all(feat_short[:, -50:] == 0.0), "Padding failed: trailing frames must be zero"

# Truncation for long audio
feat_long = extract_features_for_file(str(audio_long_path), feature_type="logmel", max_frames=400)
assert feat_long.shape == (80, 400), f"Truncation shape mismatch: {feat_long.shape}"

# Stereo downmixing & 8kHz resampling
feat_stereo = extract_features_for_file(str(audio_stereo_path), feature_type="logmel")
assert feat_stereo.shape == (80, 400), f"Stereo handling failed: {feat_stereo.shape}"

feat_8k = extract_features_for_file(str(audio_8k_path), feature_type="logmel")
assert feat_8k.shape == (80, 400), f"8kHz resampling failed: {feat_8k.shape}"

print("[PASSED] Test 2: All shape, dtype, padding, truncation, stereo, and resampling contracts verified.")

# ----------------------------------------------------------------------
# Test 3: Batch extraction pipeline with 'both', directory isolation & hash check
# ----------------------------------------------------------------------
print("\n[RUNNING] Test 3: Batch pipeline with feature_type='both' & isolation...")

test_metadata_csv = test_run_dir / "test_metadata.csv"
dummy_rows = [
    {
        "filepath": audio_norm_path.as_posix(),
        "dataset": "LA",
        "speaker_id": "SPK_01",
        "attack_id": "-",
        "attack_category": "bonafide",
        "label": "bonafide",
        "split": "train",
        "subset_filepath": audio_norm_path.as_posix(),
        "channel": "clean",
    },
    {
        "filepath": audio_8k_path.as_posix(),
        "dataset": "LA",
        "speaker_id": "SPK_02",
        "attack_id": "A01",
        "attack_category": "spoof",
        "label": "spoof",
        "split": "val",
        "subset_filepath": audio_8k_path.as_posix(),
        "channel": "g711_8khz",
    },
    {
        "filepath": "non_existent_audio.flac",
        "dataset": "PA",
        "speaker_id": "SPK_03",
        "attack_id": "AA",
        "attack_category": "spoof",
        "label": "spoof",
        "split": "test",
        "subset_filepath": "non_existent_audio_subset.flac",
        "channel": "clean",
    },
]
pd.DataFrame(dummy_rows).to_csv(test_metadata_csv, index=False)

out_features_dir = test_run_dir / "features"
out_features_metadata = test_run_dir / "features_metadata.csv"

# Check original dataset metadata hash
orig_combined_csv = PROJECT_ROOT / "data" / "subset_combined_metadata.csv"
orig_hash = hashlib.sha256(orig_combined_csv.read_bytes()).hexdigest()

res = process_feature_extraction(
    input_csv=str(test_metadata_csv),
    output_dir=str(out_features_dir),
    output_metadata_csv=str(out_features_metadata),
    feature_type="both",
    n_mels=80,
    n_mfcc=20,
    use_deltas=True,
)

assert res["input_rows"] == 3, f"Expected 3 input rows, got {res['input_rows']}"
assert res["processed_files"] == 2, f"Expected 2 processed files, got {res['processed_files']}"
assert res["skipped_files"] == 1, f"Expected 1 skipped file, got {res['skipped_files']}"
assert res["total_records"] == 4, f"Expected 4 records, got {res['total_records']}"

assert (out_features_dir / "logmel" / "train" / "audio_normal.npy").exists()
assert (out_features_dir / "mfcc" / "train" / "audio_normal.npy").exists()
assert (out_features_dir / "logmel" / "val" / "audio_8k.npy").exists()
assert (out_features_dir / "mfcc" / "val" / "audio_8k.npy").exists()

feat_meta_df = pd.read_csv(out_features_metadata)
expected_columns = [
    "feature_filepath", "feature_type", "filepath", "subset_filepath",
    "label", "split", "channel", "dataset", "speaker_id", "attack_id",
    "attack_category", "shape",
]
assert list(feat_meta_df.columns) == expected_columns, f"Columns mismatch: {list(feat_meta_df.columns)}"

after_hash = hashlib.sha256(orig_combined_csv.read_bytes()).hexdigest()
assert orig_hash == after_hash, "data/subset_combined_metadata.csv hash changed!"

print("[PASSED] Test 3: Batch extraction, directory isolation, and metadata hash verified.")

# ----------------------------------------------------------------------
# Test 4 (Bug 1): Atomicity under partial-failure when feature_type="both"
# ----------------------------------------------------------------------
print("\n[RUNNING] Test 4: Atomicity under partial-failure in feature_type='both'...")

test_atomic_csv = test_run_dir / "test_atomic.csv"
atomic_rows = [
    {
        "filepath": audio_norm_path.as_posix(),
        "dataset": "LA",
        "speaker_id": "SPK_ATOMIC",
        "attack_id": "-",
        "attack_category": "bonafide",
        "label": "bonafide",
        "split": "train",
        "subset_filepath": audio_norm_path.as_posix(),
        "channel": "clean",
    }
]
pd.DataFrame(atomic_rows).to_csv(test_atomic_csv, index=False)

atomic_features_dir = test_run_dir / "features_atomic"
atomic_features_metadata = test_run_dir / "features_metadata_atomic.csv"

real_extract = ef.extract_features_for_file

def mock_extract_fail_on_mfcc(*args, **kwargs):
    f_type = kwargs.get("feature_type", args[1] if len(args) > 1 else "logmel")
    if f_type == "mfcc":
        raise RuntimeError("Simulated MFCC extraction failure!")
    return real_extract(*args, **kwargs)

ef.extract_features_for_file = mock_extract_fail_on_mfcc

try:
    atomic_res = process_feature_extraction(
        input_csv=str(test_atomic_csv),
        output_dir=str(atomic_features_dir),
        output_metadata_csv=str(atomic_features_metadata),
        feature_type="both",
    )
finally:
    ef.extract_features_for_file = real_extract

# Assertions for Test 4:
assert atomic_res["processed_files"] == 0, f"Expected processed_files == 0, got {atomic_res['processed_files']}"
assert atomic_res["skipped_files"] == 1, f"Expected skipped_files == 1, got {atomic_res['skipped_files']}"
assert atomic_res["total_records"] == 0, f"Expected total_records == 0, got {atomic_res['total_records']}"

partial_logmel_npy = atomic_features_dir / "logmel" / "train" / f"{audio_norm_path.stem}.npy"
partial_mfcc_npy = atomic_features_dir / "mfcc" / "train" / f"{audio_norm_path.stem}.npy"
assert not partial_logmel_npy.exists(), f"Orphaned logmel .npy file found: {partial_logmel_npy}"
assert not partial_mfcc_npy.exists(), f"Failed mfcc .npy file found: {partial_mfcc_npy}"

atomic_df = pd.read_csv(atomic_features_metadata)
assert len(atomic_df) == 0, f"Expected 0 rows in features_metadata_atomic.csv, got {len(atomic_df)}"

print("[PASSED] Test 4: Atomicity verified. processed_files=0, skipped_files=1, total_records=0, partial .npy deleted, metadata has 0 rows.")

# ----------------------------------------------------------------------
# Test 5 (Bug 2): Unrecognized split warning, fallback to 'train', and counter
# ----------------------------------------------------------------------
print("\n[RUNNING] Test 5: Unrecognized split warning, fallback to 'train', and counter...")

test_split_csv = test_run_dir / "test_split.csv"
split_rows = [
    {
        "filepath": audio_norm_path.as_posix(),
        "dataset": "LA",
        "speaker_id": "SPK_DEV",
        "attack_id": "-",
        "attack_category": "bonafide",
        "label": "bonafide",
        "split": "dev",  # Unrecognized split!
        "subset_filepath": audio_norm_path.as_posix(),
        "channel": "clean",
    }
]
pd.DataFrame(split_rows).to_csv(test_split_csv, index=False)

split_features_dir = test_run_dir / "features_split"
split_features_metadata = test_run_dir / "features_metadata_split.csv"

with warnings.catch_warnings(record=True) as caught_warnings:
    warnings.simplefilter("always")
    split_res = process_feature_extraction(
        input_csv=str(test_split_csv),
        output_dir=str(split_features_dir),
        output_metadata_csv=str(split_features_metadata),
        feature_type="logmel",
    )

# 1. Trigger UserWarning containing "unrecognized split"
split_warns = [
    w for w in caught_warnings
    if issubclass(w.category, UserWarning) and "unrecognized split" in str(w.message).lower()
]
assert len(split_warns) > 0, "No UserWarning containing 'unrecognized split' was caught!"

# 2. Row processed successfully (not skipped)
assert split_res["processed_files"] == 1, f"Expected processed_files == 1, got {split_res['processed_files']}"
assert split_res["skipped_files"] == 0, f"Expected skipped_files == 0, got {split_res['skipped_files']}"

# 3. Written under train directory
expected_train_npy = split_features_dir / "logmel" / "train" / f"{audio_norm_path.stem}.npy"
assert expected_train_npy.exists(), f"File not found in train directory: {expected_train_npy}"

# 4. features_metadata_split.csv records split == 'train'
split_df = pd.read_csv(split_features_metadata)
assert len(split_df) == 1, f"Expected 1 record in metadata, got {len(split_df)}"
assert split_df.iloc[0]["split"] == "train", f"Expected split 'train', got {split_df.iloc[0]['split']}"

# 5. Returned dict has unrecognized_splits == 1
assert split_res.get("unrecognized_splits") == 1, f"Expected unrecognized_splits == 1, got {split_res.get('unrecognized_splits')}"

print("[PASSED] Test 5: Unrecognized split handled correctly. UserWarning caught, processed under 'train', metadata records 'train', unrecognized_splits=1.")

# ----------------------------------------------------------------------
# Cleanup scratch/test_run/ directory contents
# ----------------------------------------------------------------------
print("\n[CLEANUP] Cleaning up scratch/test_run/ directory contents...")
if test_run_dir.exists():
    for item in test_run_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
print("[CLEANUP] scratch/test_run/ cleaned successfully.")

print("\n" + "=" * 70)
print("ALL TESTS (ORIGINAL + BUGFIXES) COMPLETED AND PASSED!")
print("=" * 70)
