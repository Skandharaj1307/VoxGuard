"""
scratch/validate_replay_data.py

Step 1: Diagnostic script to validate PA (Physical Access - Replay Attack) audio files and metadata.
Checks:
- File count and location
- Audio format integrity (soundfile & librosa loadable)
- Sample rates and channel counts
- Audio durations (min, max, mean, distribution)
- Corrupted or unreadable files
- Class balance (bonafide vs replay/spoof)
- Speaker ID distribution and recording IDs
- Metadata consistency between disk files and metadata CSVs
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import librosa
except ImportError:
    librosa = None


def validate_dataset():
    print("=" * 70)
    print(" VoxGuard Replay Dataset Validation (Step 1)")
    print("=" * 70)

    # 1. Check data directory
    repaired_dir = PROJECT_ROOT / "data" / "subset_raw_repaired"
    metadata_file = PROJECT_ROOT / "data" / "subset_metadata.csv"
    combined_meta_file = PROJECT_ROOT / "data" / "subset_combined_metadata.csv"

    print(f"Checking repaired audio folder: {repaired_dir}")
    if not repaired_dir.exists():
        print(f"ERROR: Folder {repaired_dir} does not exist.")
        return False

    audio_files = sorted(list(repaired_dir.glob("*.flac")) + list(repaired_dir.glob("*.wav")))
    print(f"Total audio files found on disk: {len(audio_files)}")

    if len(audio_files) == 0:
        print("ERROR: No audio files found in repaired folder.")
        return False

    # 2. Check metadata
    if not metadata_file.exists():
        print(f"ERROR: Metadata file {metadata_file} not found.")
        return False

    df_meta = pd.read_csv(metadata_file)
    print(f"Loaded subset_metadata.csv: {len(df_meta)} rows total.")

    pa_meta = df_meta[df_meta["dataset"] == "PA"].copy()
    print(f"PA rows in subset_metadata.csv: {len(pa_meta)}")

    # 3. Match audio files with metadata
    file_records = []
    corrupted_files = []

    for idx, fpath in enumerate(audio_files):
        fname = fpath.name
        # Match by filename
        meta_match = df_meta[df_meta["subset_filepath"].apply(lambda p: Path(str(p)).name == fname if pd.notna(p) else False)]
        
        if len(meta_match) == 0:
            # Try matching by original flac filename stem
            stem_match = df_meta[df_meta["filepath"].apply(lambda p: Path(str(p)).stem in fname if pd.notna(p) else False)]
            if len(stem_match) > 0:
                meta_row = stem_match.iloc[0]
            else:
                meta_row = None
        else:
            meta_row = meta_match.iloc[0]

        # Audio file read test
        is_valid = True
        sr_actual = None
        channels = None
        duration_sec = None
        samples_count = None
        err_msg = ""

        try:
            if sf is not None:
                info = sf.info(str(fpath))
                sr_actual = info.samplerate
                channels = info.channels
                duration_sec = info.duration
                samples_count = info.frames
            elif librosa is not None:
                y, sr_actual = librosa.load(str(fpath), sr=None, mono=False)
                channels = 1 if y.ndim == 1 else y.shape[0]
                samples_count = len(y) if y.ndim == 1 else y.shape[1]
                duration_sec = samples_count / sr_actual
        except Exception as e:
            is_valid = False
            err_msg = str(e)
            corrupted_files.append((fname, err_msg))

        speaker_id = meta_row["speaker_id"] if meta_row is not None else "UNKNOWN"
        label = meta_row["label"] if meta_row is not None else "UNKNOWN"
        attack_id = meta_row["attack_id"] if meta_row is not None else "UNKNOWN"
        split = meta_row["split"] if meta_row is not None else "UNKNOWN"
        dataset = meta_row["dataset"] if meta_row is not None else "UNKNOWN"

        file_records.append({
            "filename": fname,
            "filepath": str(fpath),
            "is_valid": is_valid,
            "sample_rate": sr_actual,
            "channels": channels,
            "duration_sec": duration_sec,
            "samples_count": samples_count,
            "dataset": dataset,
            "speaker_id": speaker_id,
            "label": label,
            "attack_id": attack_id,
            "split": split,
            "error": err_msg
        })

    df_validated = pd.DataFrame(file_records)

    # 4. Summary Statistics
    print("\n--- Audio File Integrity ---")
    valid_count = df_validated["is_valid"].sum()
    print(f"Successfully decoded files: {valid_count} / {len(df_validated)}")
    if len(corrupted_files) > 0:
        print(f"WARNING: {len(corrupted_files)} corrupted files encountered:")
        for cf, err in corrupted_files:
            print(f"  - {cf}: {err}")
    else:
        print("All audio files read cleanly with zero decoding errors!")

    print("\n--- Audio Properties ---")
    sr_dist = df_validated["sample_rate"].value_counts().to_dict()
    ch_dist = df_validated["channels"].value_counts().to_dict()
    dur_stats = df_validated["duration_sec"].describe()
    print(f"Sample rates: {sr_dist}")
    print(f"Channel counts: {ch_dist}")
    print(f"Duration stats (seconds):")
    print(f"  Min   : {dur_stats['min']:.2f} s")
    print(f"  Mean  : {dur_stats['mean']:.2f} s")
    print(f"  Median: {dur_stats['50%']:.2f} s")
    print(f"  Max   : {dur_stats['max']:.2f} s")

    print("\n--- Class Balance (Bonafide vs Replay/Spoof) ---")
    label_dist = df_validated["label"].value_counts().to_dict()
    print(f"Class distribution: {label_dist}")
    bonafide_n = label_dist.get("bonafide", 0)
    spoof_n = label_dist.get("spoof", 0)
    print(f"  Bonafide (Label 0): {bonafide_n} ({bonafide_n/len(df_validated)*100:.1f}%)")
    print(f"  Replay/Spoof (Label 1): {spoof_n} ({spoof_n/len(df_validated)*100:.1f}%)")

    print("\n--- Replay Attack IDs Breakdown ---")
    attack_dist = df_validated[df_validated["label"] == "spoof"]["attack_id"].value_counts().to_dict()
    print(f"Replay attack configurations (room/device conditions): {attack_dist}")

    print("\n--- Speaker Distribution & Split Status ---")
    unique_speakers = df_validated["speaker_id"].unique()
    print(f"Unique speakers: {len(unique_speakers)}")
    print(f"Speakers list: {sorted(list(unique_speakers))}")
    print(f"Split distribution:")
    split_dist = df_validated.groupby(["split", "label"]).size()
    print(split_dist)

    # Verify speaker disjointness in existing metadata
    train_spk = set(df_validated[df_validated["split"] == "train"]["speaker_id"].unique())
    val_spk = set(df_validated[df_validated["split"] == "val"]["speaker_id"].unique())
    test_spk = set(df_validated[df_validated["split"] == "test"]["speaker_id"].unique())

    print("\n--- Disjoint Speaker Split Verification ---")
    print(f"Train speakers ({len(train_spk)}): {sorted(list(train_spk))}")
    print(f"Val speakers ({len(val_spk)}): {sorted(list(val_spk))}")
    print(f"Test speakers ({len(test_spk)}): {sorted(list(test_spk))}")

    train_val_overlap = train_spk.intersection(val_spk)
    train_test_overlap = train_spk.intersection(test_spk)
    val_test_overlap = val_spk.intersection(test_spk)

    print(f"Train & Val overlap : {len(train_val_overlap)} {train_val_overlap}")
    print(f"Train & Test overlap: {len(train_test_overlap)} {train_test_overlap}")
    print(f"Val & Test overlap  : {len(val_test_overlap)} {val_test_overlap}")

    if len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0:
        print(">> SPEAKER-DISJOINT SPLIT VERIFIED: Zero speaker overlap across splits!")
    else:
        print(">> WARNING: Speaker overlap detected!")

    # Save diagnostic results
    diag_csv = PROJECT_ROOT / "scratch" / "data_validation_report.csv"
    df_validated.to_csv(diag_csv, index=False)
    print(f"\nSaved diagnostic report to: {diag_csv}")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = validate_dataset()
    if not success:
        sys.exit(1)
