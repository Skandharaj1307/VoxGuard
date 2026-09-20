"""
dataset/create_subset.py

Creates a balanced, speaker-disjoint subset from data/metadata.csv for prototyping
and fine-tuning on Colab / Kaggle.

Key features:
1. Balanced sampling across datasets (LA synthetic, PA replay) and labels (bonafide vs spoof).
2. Strict speaker-disjoint train/val/test splits (70% train, 15% val, 15% test).
3. Option to copy sampled audio files into data/subset_raw/ for easy zipping & upload.
"""

import os
import argparse
import pandas as pd
import numpy as np
import shutil
from pathlib import Path

# Random seed for reproducible splits
RANDOM_SEED = 42

def create_subset(
    metadata_path: str = "data/metadata.csv",
    output_csv: str = "data/subset_metadata.csv",
    output_audio_dir: str = "data/subset_raw",
    la_bonafide_n: int = 150,
    la_spoof_n: int = 150,
    pa_bonafide_n: int = 150,
    pa_spoof_n: int = 150,
    copy_files: bool = True
):
    print("=" * 60)
    print(" VoxGuard Dataset Sampler & Speaker-Disjoint Splitter")
    print("=" * 60)

    # 1. Load full metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    print(f"Loaded full metadata: {len(df)} total rows across {df['speaker_id'].nunique()} unique speakers.")

    # Filter only rows with valid, existing filepaths
    df['exists'] = df['filepath'].apply(os.path.exists)
    valid_df = df[df['exists']].copy()
    print(f"Verified {len(valid_df)} files exist on disk.")

    if len(valid_df) == 0:
        raise ValueError("No valid audio files found on disk!")

    # 2. Speaker-Disjoint Train / Val / Test Split
    np.random.seed(RANDOM_SEED)
    unique_speakers = sorted(valid_df['speaker_id'].unique())
    np.random.shuffle(unique_speakers)

    n_speakers = len(unique_speakers)
    n_train = int(0.70 * n_speakers)
    n_val = int(0.15 * n_speakers)

    train_speakers = set(unique_speakers[:n_train])
    val_speakers = set(unique_speakers[n_train:n_train + n_val])
    test_speakers = set(unique_speakers[n_train + n_val:])

    def get_split(speaker):
        if speaker in train_speakers:
            return "train"
        elif speaker in val_speakers:
            return "val"
        else:
            return "test"

    valid_df['split'] = valid_df['speaker_id'].apply(get_split)

    print(f"\nSpeaker Split Summary:")
    print(f"  Train Speakers ({len(train_speakers)}): {sorted(list(train_speakers))}")
    print(f"  Val Speakers   ({len(val_speakers)}): {sorted(list(val_speakers))}")
    print(f"  Test Speakers  ({len(test_speakers)}): {sorted(list(test_speakers))}")

    # 3. Stratified Balanced Sampling per category
    categories = [
        ('LA', 'bonafide', la_bonafide_n),
        ('LA', 'spoof', la_spoof_n),
        ('PA', 'bonafide', pa_bonafide_n),
        ('PA', 'spoof', pa_spoof_n)
    ]

    sampled_dfs = []
    for dataset_name, label_name, target_n in categories:
        subset = valid_df[(valid_df['dataset'] == dataset_name) & (valid_df['label'] == label_name)]
        if len(subset) == 0:
            print(f"WARNING: No samples found for dataset={dataset_name}, label={label_name}")
            continue

        n_sample = min(target_n, len(subset))
        sampled = subset.sample(n=n_sample, random_state=RANDOM_SEED)
        sampled_dfs.append(sampled)
        print(f"Sampled {len(sampled)} files for {dataset_name} - {label_name}")

    final_df = pd.concat(sampled_dfs, ignore_index=True)
    final_df.drop(columns=['exists'], inplace=True)

    print(f"\nFinal Subset Size: {len(final_df)} rows")
    print(final_df.groupby(['dataset', 'label', 'split']).size())

    # 4. Save metadata CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # 5. Optionally Copy Files to a Standalone Directory
    if copy_files:
        print(f"\nCopying sampled audio files to standalone folder: {output_audio_dir} ...")
        out_dir = Path(output_audio_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        new_filepaths = []
        for idx, row in final_df.iterrows():
            src_path = Path(row['filepath'])
            rel_name = f"{row['dataset']}_{row['split']}_{src_path.name}"
            dst_path = out_dir / rel_name
            shutil.copy2(src_path, dst_path)
            new_filepaths.append(dst_path.as_posix())

        final_df['subset_filepath'] = new_filepaths

    final_df.to_csv(output_csv, index=False)
    print(f"\nSubset metadata saved successfully to: {output_csv}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a balanced speaker-disjoint subset for VoxGuard")
    parser.add_argument("--metadata", type=str, default="data/metadata.csv", help="Path to input metadata.csv")
    parser.add_argument("--out_csv", type=str, default="data/subset_metadata.csv", help="Path to output subset CSV")
    parser.add_argument("--out_dir", type=str, default="data/subset_raw", help="Directory to store sampled audio files")
    parser.add_argument("--la_bonafide", type=int, default=150, help="Number of LA bonafide samples")
    parser.add_argument("--la_spoof", type=int, default=150, help="Number of LA spoof samples")
    parser.add_argument("--pa_bonafide", type=int, default=150, help="Number of PA bonafide samples")
    parser.add_argument("--pa_spoof", type=int, default=150, help="Number of PA spoof samples")
    parser.add_argument("--no_copy", action="store_true", help="Do not copy audio files to out_dir")

    args = parser.parse_args()

    create_subset(
        metadata_path=args.metadata,
        output_csv=args.out_csv,
        output_audio_dir=args.out_dir,
        la_bonafide_n=args.la_bonafide,
        la_spoof_n=args.la_spoof,
        pa_bonafide_n=args.pa_bonafide,
        pa_spoof_n=args.pa_spoof,
        copy_files=not args.no_copy
    )
