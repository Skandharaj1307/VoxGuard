"""
Module 4 — Feature Extraction. Reads from data/subset_combined_metadata.csv (read-only). Writes only to data/features/. Does not modify any other module's files.

VoxGuard Audio Deepfake / Spoof Detection Pipeline
Supports ASVspoof2019 (LA + PA subsets) across clean and channel-degraded (G.711 8kHz) audio.

Key Architectural Guarantees:
1. Strict Read-Only Access: Input metadata (e.g. data/subset_combined_metadata.csv) is never overwritten or mutated.
2. Isolated Storage: All feature tensors are written exclusively to data/features/<feature_type>/<split>/<original_filename>.npy.
3. Decoupled Metadata: Feature mappings are written to data/features_metadata.csv for downstream training modules to join on.
4. Fault Tolerance: Corrupt or missing audio files log a warning and are safely skipped without halting batch processing.
5. Clean Public API: extract_features_for_file() is a pure, side-effect-free function suitable for downstream imports.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd

# Audio processing library imports with graceful fallbacks
try:
    # pyrefly: ignore [missing-import]
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    # pyrefly: ignore [missing-import]
    import scipy.signal as signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    # pyrefly: ignore [missing-import]
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def load_audio(
    filepath: Union[str, Path],
    target_sr: int = 16000,
) -> Tuple[np.ndarray, int]:
    """
    Loads an audio file into a 1D mono float32 numpy array resampled to target_sr.

    Rationale for 16kHz across all channels:
    Even for G.711 degraded files (which simulate 8kHz telephone bandwidth), maintaining
    a uniform 16kHz sampling rate preserves the 0-4kHz band-limiting and mu-law codec
    artifacts while ensuring consistent time-frequency resolution for downstream models
    (e.g. AASIST, RawNet2, 2D-CNNs) that expect a fixed 16kHz input grid.

    Parameters:
        filepath: Path to audio file (.flac, .wav, etc.)
        target_sr: Target sample rate in Hz (default: 16000)

    Returns:
        Tuple of (audio_waveform, sample_rate)
        - audio_waveform: 1D np.ndarray of dtype np.float32
        - sample_rate: int equal to target_sr
    """
    path_obj = Path(filepath)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Audio file not found: {path_obj.resolve()}")

    audio = None
    sr = None

    # Primary loader: soundfile (fast and robust for standard WAV/FLAC)
    if SOUNDFILE_AVAILABLE:
        try:
            audio, sr = sf.read(str(path_obj), dtype="float32")
        except Exception:
            audio = None
            sr = None

    # Secondary loader: librosa (handles wide variety of formats and containers)
    if audio is None:
        if LIBROSA_AVAILABLE:
            audio, sr = librosa.load(str(path_obj), sr=None, mono=False)
        else:
            raise RuntimeError(
                f"Failed to load audio from {path_obj}. Neither soundfile nor librosa could read the file."
            )

    # Convert multi-channel (stereo) to mono
    if audio.ndim > 1:
        # soundfile returns (samples, channels), librosa returns (channels, samples)
        if audio.shape[0] > audio.shape[1]:
            audio = np.mean(audio, axis=1)
        else:
            audio = np.mean(audio, axis=0)

    # Ensure 1D float32 contiguous array
    audio = np.ascontiguousarray(audio, dtype=np.float32)

    # Resample if audio sample rate differs from target_sr
    if sr != target_sr:
        if LIBROSA_AVAILABLE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        elif SCIPY_AVAILABLE:
            gcd = np.gcd(sr, target_sr)
            up = target_sr // gcd
            down = sr // gcd
            audio = signal.resample_poly(audio, up, down).astype(np.float32)
        else:
            warnings.warn(
                f"Audio sample rate {sr} does not match target {target_sr}, "
                "and neither librosa nor scipy is available to resample. Using original rate."
            )
        sr = target_sr

    return audio, sr


def extract_features_for_file(
    filepath: str,
    feature_type: str = "logmel",
    n_mels: int = 80,
    n_mfcc: int = 20,
    max_frames: int = 400,
    sample_rate: int = 16000,
    use_deltas: bool = False,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Public Feature Extraction Contract for VoxGuard modules.

    Extracts fixed-dimension spectral representations (log-mel spectrogram or MFCC)
    from a single audio file. Shorter signals are zero-padded along the temporal axis;
    longer signals are truncated to max_frames to produce homogeneous tensors suitable
    for batch training.

    Parameters:
        filepath: str
            Path to the audio file (.flac, .wav, etc.).
        feature_type: str
            Type of acoustic feature to extract. Options: "logmel", "mfcc".
            Default: "logmel".
        n_mels: int
            Number of Mel frequency bands when feature_type="logmel".
            Default: 80.
        n_mfcc: int
            Number of Mel-Frequency Cepstral Coefficients when feature_type="mfcc".
            Default: 20.
        max_frames: int
            Fixed number of temporal frames (columns). If audio is shorter,
            it is zero-padded on the right. If longer, it is truncated.
            Set <= 0 or None to return variable-length unpadded frames.
            Default: 400.
        sample_rate: int
            Target sampling rate in Hz. Resamples if input audio differs.
            Default: 16000.
        use_deltas: bool
            If True and feature_type="mfcc", computes first-order (delta) and
            second-order (delta-delta) temporal derivatives and stacks them
            along the feature axis, expanding the dimension from n_mfcc to 3 * n_mfcc.
            Default: False.
        n_fft: int
            FFT analysis window size in samples.
            Default: 1024.
        hop_length: int
            Frame advance / hop size in samples. At 16kHz with hop_length=512,
            each frame represents 32ms (~31.25 fps).
            Default: 512.

    Returns:
        np.ndarray:
            Dtype: np.float32
            Shape:
                - feature_type == "logmel": (n_mels, max_frames) -> (80, 400)
                - feature_type == "mfcc" (use_deltas=False): (n_mfcc, max_frames) -> (20, 400)
                - feature_type == "mfcc" (use_deltas=True):  (3 * n_mfcc, max_frames) -> (60, 400)
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError(
            "librosa is required for acoustic feature extraction. "
            "Please install it using: pip install librosa or pip install -r requirements.txt"
        )

    # 1. Load and resample audio
    audio, _ = load_audio(filepath, target_sr=sample_rate)

    if len(audio) == 0:
        raise ValueError(f"Audio file '{filepath}' contains zero samples.")

    # 2. Extract acoustic feature
    feature_type_norm = feature_type.strip().lower()

    if feature_type_norm == "logmel":
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        # Convert power spectrogram to decibels (log scale)
        features = librosa.power_to_db(mel_spec, ref=np.max)

    elif feature_type_norm == "mfcc":
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        if use_deltas:
            delta1 = librosa.feature.delta(mfcc, order=1)
            delta2 = librosa.feature.delta(mfcc, order=2)
            features = np.concatenate([mfcc, delta1, delta2], axis=0)
        else:
            features = mfcc

    else:
        raise ValueError(
            f"Unsupported feature_type '{feature_type}'. Supported choices: 'logmel', 'mfcc'."
        )

    # 3. Fixed-length alignment (padding or truncation)
    if max_frames is not None and max_frames > 0:
        current_frames = features.shape[1]
        if current_frames > max_frames:
            features = features[:, :max_frames]
        elif current_frames < max_frames:
            pad_width = max_frames - current_frames
            features = np.pad(
                features,
                ((0, 0), (0, pad_width)),
                mode="constant",
                constant_values=0.0,
            )

    return features.astype(np.float32)


def resolve_audio_path(row: pd.Series, base_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Safely resolves the local path to an audio file for a given metadata row.
    Prefers copied subset_filepath (e.g. data/subset_raw/ or data/subset_g711/)
    if present on disk, falling back to original filepath (e.g. data/raw/).

    Returns:
        Path if found, else None.
    """
    candidate_keys = ["subset_filepath", "filepath"]
    candidates = []

    for key in candidate_keys:
        val = row.get(key)
        if val is not None and isinstance(val, str) and val.strip():
            candidates.append(val.strip())

    if base_dir is None:
        base_dir = Path.cwd()

    for cand in candidates:
        p = Path(cand)
        # Direct check
        if p.is_file():
            return p
        # Check relative to base_dir
        rel_p = base_dir / p
        if rel_p.is_file():
            return rel_p

    return None


def process_feature_extraction(
    input_csv: str = "data/subset_combined_metadata.csv",
    output_dir: str = "data/features",
    output_metadata_csv: str = "data/features_metadata.csv",
    feature_type: str = "logmel",
    n_mels: int = 80,
    n_mfcc: int = 20,
    max_frames: int = 400,
    sample_rate: int = 16000,
    use_deltas: bool = False,
) -> Dict[str, Union[int, str]]:
    """
    Batch feature extraction orchestrator for VoxGuard.

    Reads metadata from input_csv in strictly read-only mode, extracts acoustic features,
    writes .npy arrays into dedicated subfolders data/features/<feature_type>/<split>/,
    and saves output mapping to output_metadata_csv.

    Parameters:
        input_csv: Path to input metadata CSV (read-only)
        output_dir: Root directory for output .npy files
        output_metadata_csv: Path to separate features metadata CSV
        feature_type: "logmel", "mfcc", or "both"
        n_mels: Number of Mel bands (for logmel)
        n_mfcc: Number of MFCC coefficients (for mfcc)
        max_frames: Target frame length for padding/truncation
        sample_rate: Target audio sampling rate in Hz
        use_deltas: Include delta and delta-delta for MFCC

    Returns:
        Dict summarizing execution metrics (processed count, skipped count, output paths).
    """
    print("=" * 70)
    print(" VoxGuard Feature Extraction Pipeline (Module 4)")
    print("=" * 70)

    input_csv_path = Path(input_csv)
    if not input_csv_path.is_file():
        raise FileNotFoundError(
            f"Input metadata CSV not found: {input_csv_path.resolve()}.\n"
            "Please verify data/subset_combined_metadata.csv exists before running."
        )

    # Read-only metadata load
    df = pd.read_csv(input_csv_path)
    print(f"Loaded input metadata: {len(df)} rows from {input_csv_path.as_posix()} (read-only).")

    # Determine active feature types
    feature_type_norm = feature_type.strip().lower()
    if feature_type_norm == "both":
        active_feature_types = ["logmel", "mfcc"]
    elif feature_type_norm in ["logmel", "mfcc"]:
        active_feature_types = [feature_type_norm]
    else:
        raise ValueError(
            f"Invalid feature_type '{feature_type}'. Allowed choices: 'logmel', 'mfcc', 'both'."
        )

    print(f"Target feature types: {active_feature_types}")
    print(f"Sampling rate: {sample_rate} Hz | Max frames: {max_frames} | Use deltas: {use_deltas}")

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Pre-create directory tree: <output_dir>/<feature_type>/<split>/
    known_splits = ["train", "val", "test"]
    for f_type in active_feature_types:
        for split_name in known_splits:
            (out_root / f_type / split_name).mkdir(parents=True, exist_ok=True)

    output_records: List[Dict[str, str]] = []
    processed_files = 0
    skipped_files = 0
    unrecognized_splits = 0

    row_iterator = df.iterrows()
    if TQDM_AVAILABLE:
        row_iterator = tqdm(row_iterator, total=len(df), desc="Extracting features")

    for idx, row in row_iterator:
        src_path = resolve_audio_path(row)

        if src_path is None:
            skipped_files += 1
            cand_fp = row.get("subset_filepath", row.get("filepath", "unknown"))
            print(f"  [WARNING] Audio file not found for index {idx} ({cand_fp}). Skipping.")
            continue

        raw_split = str(row.get("split", "train")).strip().lower()
        if raw_split not in known_splits:
            unrecognized_splits += 1
            cand_fp = row.get("subset_filepath", row.get("filepath", src_path.name))
            warnings.warn(
                f"Row {idx} ({cand_fp}) has unrecognized split '{raw_split}'. "
                f"Defaulting to 'train'.",
                UserWarning,
                stacklevel=2,
            )
            print(
                f"  [WARNING] Row {idx} ({cand_fp}): unrecognized split '{raw_split}'. "
                "Defaulting to 'train'."
            )
            split = "train"
        else:
            split = raw_split

        original_stem = src_path.stem
        npy_filename = f"{original_stem}.npy"

        # Extract each requested feature type atomically per row
        row_success = True
        row_records: List[Dict[str, str]] = []
        row_written_paths: List[Path] = []

        for f_type in active_feature_types:
            split_dir = out_root / f_type / split
            split_dir.mkdir(parents=True, exist_ok=True)
            dst_npy_path = split_dir / npy_filename

            try:
                feat = extract_features_for_file(
                    filepath=str(src_path),
                    feature_type=f_type,
                    n_mels=n_mels,
                    n_mfcc=n_mfcc,
                    max_frames=max_frames,
                    sample_rate=sample_rate,
                    use_deltas=use_deltas,
                )

                # Save tensor as .npy and track path
                np.save(dst_npy_path, feat)
                row_written_paths.append(dst_npy_path)

                # Stage metadata catalog entry
                record = {
                    "feature_filepath": dst_npy_path.as_posix(),
                    "feature_type": f_type,
                    "filepath": str(row.get("filepath", "")),
                    "subset_filepath": str(row.get("subset_filepath", "")),
                    "label": str(row.get("label", "")),
                    "split": split,
                    "channel": str(row.get("channel", "")),
                    "dataset": str(row.get("dataset", "")),
                    "speaker_id": str(row.get("speaker_id", "")),
                    "attack_id": str(row.get("attack_id", "")),
                    "attack_category": str(row.get("attack_category", "")),
                    "shape": str(list(feat.shape)),
                }
                row_records.append(record)

            except Exception as exc:
                row_success = False
                print(f"  [ERROR] Failed to extract {f_type} for {src_path}: {exc}")
                # Clean up any partial files written for this row in this iteration
                for p in row_written_paths:
                    try:
                        if p.is_file():
                            p.unlink()
                    except OSError as cleanup_err:
                        print(f"  [WARNING] Failed to clean up partial file {p}: {cleanup_err}")
                break

        if row_success:
            output_records.extend(row_records)
            processed_files += 1
        else:
            skipped_files += 1

        if not TQDM_AVAILABLE and ((idx + 1) % 100 == 0 or (idx + 1) == len(df)):
            print(f"  Processed {idx + 1}/{len(df)} files...")

    # Save output metadata CSV
    out_meta_path = Path(output_metadata_csv)
    out_meta_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "feature_filepath",
        "feature_type",
        "filepath",
        "subset_filepath",
        "label",
        "split",
        "channel",
        "dataset",
        "speaker_id",
        "attack_id",
        "attack_category",
        "shape",
    ]

    out_df = pd.DataFrame(output_records, columns=fieldnames)
    out_df.to_csv(out_meta_path, index=False)

    print("\n" + "=" * 70)
    print(" Feature Extraction Run Summary")
    print("=" * 70)
    print(f"Total Input Rows Evaluated : {len(df)}")
    print(f"Audio Files Processed      : {processed_files}")
    print(f"Files Skipped / Errors     : {skipped_files}")
    print(f"Rows with Unrecognized Split (defaulted to 'train'): {unrecognized_splits}")
    print(f"Total Feature Records Saved: {len(out_df)}")
    print(f"Output Features Directory  : {out_root.resolve()}")
    print(f"Output Features Metadata   : {out_meta_path.resolve()}")

    if len(out_df) > 0:
        print("\nSummary Breakdown (Records by split, channel, feature_type, label):")
        print(out_df.groupby(["split", "channel", "feature_type", "label"]).size())

    print("=" * 70)

    return {
        "input_rows": len(df),
        "processed_files": processed_files,
        "skipped_files": skipped_files,
        "unrecognized_splits": unrecognized_splits,
        "total_records": len(out_df),
        "features_dir": out_root.as_posix(),
        "features_metadata": out_meta_path.as_posix(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="VoxGuard Module 4: Feature Extraction for ASVspoof2019"
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="data/subset_combined_metadata.csv",
        help="Path to input subset combined metadata CSV (read-only)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/features",
        help="Root directory to store extracted .npy feature files",
    )
    parser.add_argument(
        "--output-metadata",
        type=str,
        default="data/features_metadata.csv",
        help="Path to output features metadata CSV",
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        default="logmel",
        choices=["logmel", "mfcc", "both"],
        help="Feature type to extract: 'logmel', 'mfcc', or 'both' (default: 'logmel')",
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=80,
        help="Number of Mel frequency bins for logmel (default: 80)",
    )
    parser.add_argument(
        "--n-mfcc",
        type=int,
        default=20,
        help="Number of MFCC coefficients (default: 20)",
    )
    parser.add_argument(
        "--use-deltas",
        action="store_true",
        help="Compute delta and delta-delta coefficients for MFCC (output shape becomes 3*n_mfcc x max_frames)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=400,
        help="Fixed temporal frame length for truncation/padding (default: 400)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target audio sample rate in Hz (default: 16000)",
    )

    args = parser.parse_args()

    process_feature_extraction(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        output_metadata_csv=args.output_metadata,
        feature_type=args.feature_type,
        n_mels=args.n_mels,
        n_mfcc=args.n_mfcc,
        max_frames=args.max_frames,
        sample_rate=args.sample_rate,
        use_deltas=args.use_deltas,
    )


if __name__ == "__main__":
    main()
