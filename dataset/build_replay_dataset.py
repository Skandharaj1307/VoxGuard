"""
dataset/build_replay_dataset.py

Builds the feature dataset for the VoxGuard Replay Attack Detector.
Segments each PA audio file into 2.0-second chunks (32,000 samples @ 16kHz),
extracts all 13 replay features using modules.detector_replay.features,
and saves the compiled dataset to data/replay_features_dataset.csv.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.detector_replay.preprocessing import load_and_preprocess, chunk_waveform
from modules.detector_replay.features import (
    extract_chunk_features_dict,
    REPLAY_FEATURE_NAMES
)


def build_replay_dataset(
    repaired_dir: str = "data/subset_raw_repaired",
    metadata_csv: str = "data/subset_metadata.csv",
    output_csv: str = "data/replay_features_dataset.csv",
    chunk_duration: float = 2.0,
    target_sr: int = 16000,
) -> pd.DataFrame:
    print("=" * 75)
    print(" VoxGuard Step 4 -- Building Replay Feature Dataset")
    print("=" * 75)

    repaired_path = PROJECT_ROOT / repaired_dir
    meta_path = PROJECT_ROOT / metadata_csv
    out_path = PROJECT_ROOT / output_csv

    if not repaired_path.exists():
        raise FileNotFoundError(f"Audio directory not found: {repaired_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {meta_path}")

    df_meta = pd.read_csv(meta_path)
    audio_files = sorted(list(repaired_path.glob("*.flac")) + list(repaired_path.glob("*.wav")))
    print(f"Total audio files to process: {len(audio_files)}")

    # Index metadata by filename
    meta_by_name = {}
    for _, row in df_meta.iterrows():
        sf_name = Path(str(row.get("subset_filepath", ""))).name
        orig_stem = Path(str(row.get("filepath", ""))).stem
        if sf_name:
            meta_by_name[sf_name] = row
        if orig_stem:
            meta_by_name[orig_stem] = row

    records = []

    for fpath in tqdm(audio_files, desc="Extracting 2-second chunk features"):
        fname = fpath.name
        # Match metadata
        row = meta_by_name.get(fname)
        if row is None:
            # Match stem
            for key, r in meta_by_name.items():
                if key in fname:
                    row = r
                    break

        if row is None:
            print(f"Warning: No metadata match for {fname}")
            continue

        raw_label = str(row.get("label", "bonafide")).strip().lower()
        label_int = 0 if raw_label == "bonafide" else 1
        speaker_id = str(row.get("speaker_id", "UNKNOWN"))
        attack_id = str(row.get("attack_id", "-"))
        split = str(row.get("split", "train")).strip().lower()

        # Load and chunk audio
        audio, sr = load_and_preprocess(fpath, target_sr=target_sr)
        chunks = chunk_waveform(audio, sample_rate=sr, chunk_duration=chunk_duration, pad_short=True)

        for chunk_idx, chunk in enumerate(chunks):
            feat_dict = extract_chunk_features_dict(chunk, sample_rate=sr)

            record = {
                "filename": fname,
                "chunk_index": chunk_idx,
                "speaker_id": speaker_id,
                "label": label_int,
                "label_name": raw_label,
                "attack_id": attack_id,
                "split": split,
                **feat_dict
            }
            records.append(record)

    df_dataset = pd.DataFrame(records)

    # Reorder columns: metadata first, then features
    meta_cols = ["filename", "chunk_index", "speaker_id", "label", "label_name", "attack_id", "split"]
    ordered_cols = meta_cols + REPLAY_FEATURE_NAMES
    df_dataset = df_dataset[ordered_cols]

    # Verification and Statistics
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_dataset.to_csv(out_path, index=False)

    print("\n" + "=" * 75)
    print(" Feature Dataset Generation Summary")
    print("=" * 75)
    print(f"Total Rows (2s Chunks)     : {len(df_dataset)}")
    print(f"Total Audio Files          : {df_dataset['filename'].nunique()}")
    print(f"Number of Feature Columns  : {len(REPLAY_FEATURE_NAMES)}")
    print(f"Unique Speakers            : {df_dataset['speaker_id'].nunique()}")

    # Class breakdown
    bonafide_rows = (df_dataset['label'] == 0).sum()
    replay_rows = (df_dataset['label'] == 1).sum()
    bonafide_files = df_dataset[df_dataset['label'] == 0]['filename'].nunique()
    replay_files = df_dataset[df_dataset['label'] == 1]['filename'].nunique()
    print(f"Bonafide (Label 0)         : {bonafide_rows} chunks ({bonafide_files} files)")
    print(f"Replay / Spoof (Label 1)   : {replay_rows} chunks ({replay_files} files)")

    # Data Integrity Checks
    feature_matrix = df_dataset[REPLAY_FEATURE_NAMES].values
    nan_count = int(np.isnan(feature_matrix).sum())
    inf_count = int(np.isinf(feature_matrix).sum())
    print(f"NaN Count Across Features  : {nan_count}")
    print(f"Inf Count Across Features  : {inf_count}")

    # Split breakdown
    print("\nSplit Breakdown (Chunks):")
    print(df_dataset.groupby(['split', 'label_name']).size())

    print(f"\nSaved feature dataset to   : {out_path.resolve()}")
    print("=" * 75)

    assert nan_count == 0, "Dataset contains NaN values!"
    assert inf_count == 0, "Dataset contains Infinite values!"

    return df_dataset


if __name__ == "__main__":
    build_replay_dataset()
