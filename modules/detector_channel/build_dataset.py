"""
modules/detector_channel/build_dataset.py

Batch feature extraction orchestrator for VoxGuard G.711 Telephone Channel Detector.
Reads from data/subset_combined_metadata.csv (read-only).
Extracts diagnostic channel features and saves to data/channel_features.csv.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

# Add repo root to path if needed
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from modules.detector_channel.features import (
    extract_channel_features,
    CHANNEL_FEATURE_NAMES,
)

CSV_COLUMNS = [
    "filepath",
    "split",
    "channel",
    "high_freq_energy_ratio",
    "low_freq_energy_ratio",
    "spectral_rolloff",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_flatness",
    "spectral_flux",
]


def resolve_audio_path(row: pd.Series, base_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Safely resolves local on-disk path for a metadata row.
    Prefers subset_filepath (e.g. data/subset_raw/ or data/subset_g711/),
    falling back to filepath (e.g. data/raw/). Matches Module 4 resolution logic.
    """
    if base_dir is None:
        base_dir = REPO_ROOT

    for key in ["subset_filepath", "filepath"]:
        val = row.get(key)
        if val is not None and isinstance(val, str) and val.strip():
            candidate = val.strip()
            p = Path(candidate)
            if p.is_file():
                return p
            rel_p = base_dir / p
            if rel_p.is_file():
                return rel_p

    return None


def build_channel_features_dataset(
    metadata_csv: Union[str, Path] = "data/subset_combined_metadata.csv",
    output_csv: Union[str, Path] = "data/channel_features.csv",
    sample_rate: int = 16000,
) -> pd.DataFrame:
    """
    Extracts G.711 channel features for all rows in metadata_csv and writes to output_csv.

    Parameters:
        metadata_csv: Path to input metadata CSV (read-only)
        output_csv: Path to destination channel features CSV
        sample_rate: Audio sampling rate (default: 16000)

    Returns:
        pd.DataFrame containing the extracted dataset.
    """
    meta_path = Path(metadata_csv)
    out_path = Path(output_csv)

    if not meta_path.is_file():
        meta_path = REPO_ROOT / meta_path
        if not meta_path.is_file():
            raise FileNotFoundError(f"Input metadata CSV not found: {metadata_csv}")

    print("=" * 75)
    print(" VoxGuard Channel Detector: Building Feature Dataset")
    print(f" Input Metadata (read-only): {meta_path.as_posix()}")
    print(f" Output Features CSV       : {out_path.as_posix()}")
    print("=" * 75)

    df_meta = pd.read_csv(meta_path)
    total_files = len(df_meta)
    print(f"Loaded {total_files} rows from metadata.")

    records: List[Dict[str, Union[str, float]]] = []
    processed_count = 0
    skipped_count = 0

    start_time = time.time()
    row_iterator = df_meta.iterrows()
    if TQDM_AVAILABLE:
        row_iterator = tqdm(row_iterator, total=total_files, desc="Extracting channel features")

    for idx, row in row_iterator:
        audio_path = resolve_audio_path(row)

        if audio_path is None:
            skipped_count += 1
            cand_fp = row.get("subset_filepath", row.get("filepath", f"row_{idx}"))
            print(f"  [WARNING] Audio file not found for index {idx} ({cand_fp}). Skipping.")
            continue

        try:
            feats = extract_channel_features(audio_path, sample_rate=sample_rate)
            record: Dict[str, Union[str, float]] = {
                "filepath": audio_path.as_posix(),
                "split": str(row.get("split", "train")).strip().lower(),
                "channel": str(row.get("channel", "clean")).strip().lower(),
            }
            # Add all 7 features in exact order
            for feat_name in CHANNEL_FEATURE_NAMES:
                record[feat_name] = feats[feat_name]

            records.append(record)
            processed_count += 1

        except Exception as exc:
            skipped_count += 1
            print(f"  [ERROR] Failed extracting features for {audio_path}: {exc}. Skipping.")

        if not TQDM_AVAILABLE and ((idx + 1) % 100 == 0 or (idx + 1) == total_files):
            print(f"  Processed {idx + 1}/{total_files} files...")

    elapsed_time = time.time() - start_time

    # Construct DataFrame
    out_df = pd.DataFrame(records, columns=CSV_COLUMNS)

    # Ensure output parent directory exists and save CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nSuccessfully wrote {len(out_df)} records to {out_path.as_posix()}")

    # Print comprehensive summary
    print("\n" + "=" * 75)
    print(" Channel Dataset Extraction Summary")
    print("=" * 75)
    print(f"Total Input Files Evaluated : {total_files}")
    print(f"Successfully Processed      : {processed_count}")
    print(f"Skipped / Errors            : {skipped_count}")
    print(f"Elapsed Processing Time     : {elapsed_time:.2f} s ({elapsed_time / max(1, processed_count):.3f} s/file)")
    print(f"Splits Breakdown            : {out_df['split'].value_counts().to_dict() if not out_df.empty else {}}")
    print(f"Channels Breakdown          : {out_df['channel'].value_counts().to_dict() if not out_df.empty else {}}")

    if not out_df.empty:
        print("\nFeature Value Ranges by Channel Type (Sanity Check):")
        print("-" * 75)
        for feat in CHANNEL_FEATURE_NAMES:
            stats = out_df.groupby("channel")[feat].agg(["min", "mean", "max"])
            print(f"Feature: {feat}")
            for ch in stats.index:
                row_s = stats.loc[ch]
                print(f"  [{ch:11s}] min: {row_s['min']:10.4f} | mean: {row_s['mean']:10.4f} | max: {row_s['max']:10.4f}")
        print("=" * 75)

    return out_df


def main():
    parser = argparse.ArgumentParser(
        description="VoxGuard Module: Extract channel features for G.711 telephone detection"
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="data/subset_combined_metadata.csv",
        help="Path to input metadata CSV (read-only)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/channel_features.csv",
        help="Path to output channel features CSV",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sampling rate (default: 16000)",
    )
    args = parser.parse_args()

    build_channel_features_dataset(
        metadata_csv=args.input_csv,
        output_csv=args.output_csv,
        sample_rate=args.sample_rate,
    )


if __name__ == "__main__":
    main()
