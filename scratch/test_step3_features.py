"""
scratch/test_step3_features.py

Step 3 Verification: Tests replay feature extraction on bonafide and replay audio files.
Displays actual feature values and explains each acoustic indicator.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.detector_replay.preprocessing import load_and_preprocess, chunk_waveform
from modules.detector_replay.features import (
    extract_chunk_features,
    extract_chunk_features_dict,
    REPLAY_FEATURE_NAMES,
    compute_reverb_proxy
)


def run_step3_tests():
    print("=" * 75)
    print(" VoxGuard Step 3 -- Replay Feature Extraction Verification")
    print("=" * 75)

    repaired_dir = PROJECT_ROOT / "data" / "subset_raw_repaired"
    metadata_file = PROJECT_ROOT / "data" / "subset_metadata.csv"

    df_meta = pd.read_csv(metadata_file)
    pa_bonafide = df_meta[(df_meta["dataset"] == "PA") & (df_meta["label"] == "bonafide")]
    pa_replay = df_meta[(df_meta["dataset"] == "PA") & (df_meta["label"] == "spoof")]

    available_files = {p.name: p for p in repaired_dir.glob("*.flac")}

    # Select 2 bonafide and 2 replay files
    sample_bonafide = []
    for _, row in pa_bonafide.iterrows():
        fn = Path(str(row["subset_filepath"])).name
        if fn in available_files:
            sample_bonafide.append((available_files[fn], row["speaker_id"], row["label"]))
        if len(sample_bonafide) == 2:
            break

    sample_replay = []
    for _, row in pa_replay.iterrows():
        fn = Path(str(row["subset_filepath"])).name
        if fn in available_files:
            sample_replay.append((available_files[fn], row["speaker_id"], row["label"], row["attack_id"]))
        if len(sample_replay) == 2:
            break

    print(f"\nFeature Schema ({len(REPLAY_FEATURE_NAMES)} features):")
    for i, name in enumerate(REPLAY_FEATURE_NAMES, 1):
        print(f"  [{i:02d}] {name}")

    all_rows = []

    print("\n" + "-" * 75)
    print("Extracting features from sample Bonafide vs Replay chunks (2.0s @ 16kHz)...")
    print("-" * 75)

    # Process bonafide samples
    for fpath, spk, lbl in sample_bonafide:
        audio, sr = load_and_preprocess(fpath)
        chunks = chunk_waveform(audio, sample_rate=sr, chunk_duration=2.0)
        t0 = time.perf_counter()
        feat_dict = extract_chunk_features_dict(chunks[0], sample_rate=sr)
        t_ms = (time.perf_counter() - t0) * 1000.0

        feat_row = {
            "file": fpath.name[:24] + "...",
            "speaker": spk,
            "label": lbl,
            "attack": "-",
            "latency_ms": f"{t_ms:.2f}ms",
            **feat_dict
        }
        all_rows.append(feat_row)

    # Process replay samples
    for fpath, spk, lbl, atk in sample_replay:
        audio, sr = load_and_preprocess(fpath)
        chunks = chunk_waveform(audio, sample_rate=sr, chunk_duration=2.0)
        t0 = time.perf_counter()
        feat_dict = extract_chunk_features_dict(chunks[0], sample_rate=sr)
        t_ms = (time.perf_counter() - t0) * 1000.0

        feat_row = {
            "file": fpath.name[:24] + "...",
            "speaker": spk,
            "label": lbl,
            "attack": atk,
            "latency_ms": f"{t_ms:.2f}ms",
            **feat_dict
        }
        all_rows.append(feat_row)

    df_results = pd.DataFrame(all_rows)

    # Display side-by-side comparison table
    key_cols = [
        "file", "label", "attack",
        "spectral_flatness_mean",
        "spectral_rolloff_mean",
        "reverb_proxy_decay_ratio",
        "spectral_centroid_mean",
        "spectral_bandwidth_mean",
        "zero_crossing_rate_mean",
        "rms_energy_mean"
    ]
    print("\n--- Key Acoustic Features Comparison (Chunk 1) ---")
    print(df_results[key_cols].to_string(index=False))

    print("\n--- Full Feature Vector Validation ---")
    for idx, row in df_results.iterrows():
        feats_vec = np.array([row[name] for name in REPLAY_FEATURE_NAMES], dtype=np.float32)
        has_nan = np.isnan(feats_vec).any()
        has_inf = np.isinf(feats_vec).any()
        print(f"Sample {idx+1} ({row['label']}): shape={feats_vec.shape}, NaNs={has_nan}, Infs={has_inf}")
        assert not has_nan, f"NaN detected in sample {idx+1}"
        assert not has_inf, f"Inf detected in sample {idx+1}"
        assert len(feats_vec) == 13, f"Expected 13 features, got {len(feats_vec)}"

    print("\n" + "=" * 75)
    print(" ALL FEATURE EXTRACTION TESTS PASSED SUCCESSFULLY!")
    print("=" * 75)
    return True


if __name__ == "__main__":
    success = run_step3_tests()
    if not success:
        sys.exit(1)
