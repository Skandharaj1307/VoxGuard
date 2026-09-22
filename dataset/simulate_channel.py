"""
dataset/simulate_channel.py

Applies G.711 mu-law (8kHz mono) channel/codec degradation to the 600 sampled audio files.
Creates parallel degraded files and updates metadata to produce 1,200 combined rows
(600 clean + 600 g711_8khz) while strictly preserving speaker-disjoint train/val/test splits.
"""

import os
import shutil
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path

# Audio processing fallback imports
try:
    import soundfile as sf
    import scipy.signal as signal
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


try:
    import imageio_ffmpeg
    IMAGEIO_FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    IMAGEIO_FFMPEG_EXE = None


def check_ffmpeg():
    """Checks if ffmpeg binary is available on system PATH or via imageio_ffmpeg."""
    if IMAGEIO_FFMPEG_EXE:
        return IMAGEIO_FFMPEG_EXE
    try:
        res = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0:
            return "ffmpeg"
    except FileNotFoundError:
        pass
    return None


def degrade_file_ffmpeg(ffmpeg_bin: str, input_path: str, output_path: str, codec: str = "pcm_mulaw"):
    """Resamples audio to 8kHz mono and encodes with G.711 mu-law via ffmpeg, then relabels sample rate to 16kHz for AASIST timing."""
    temp_8k = f"{output_path}.8k.wav"
    try:
        cmd_degrade = [
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-i", input_path,
            "-ar", "8000",
            "-ac", "1",
            "-c:a", codec,
            temp_8k
        ]
        subprocess.run(cmd_degrade, check=True)

        cmd_16k = [
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-i", temp_8k,
            "-ar", "16000",
            output_path
        ]
        subprocess.run(cmd_16k, check=True)
    finally:
        if os.path.exists(temp_8k):
            os.remove(temp_8k)


def degrade_file_soundfile(input_path: str, output_path: str):
    """Fallback: Resamples to 8kHz mono G.711 mu-law, then upsamples to 16kHz for AASIST timing."""
    data, sr = sf.read(input_path)
    
    # Convert stereo to mono if necessary
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample to 8000 Hz if needed
    if sr != 8000:
        gcd = np.gcd(sr, 8000)
        up = 8000 // gcd
        down = sr // gcd
        data = signal.resample_poly(data, up, down)

    # Resample/interpolate to 16000 Hz so AASIST reads timing correctly
    data_16k = signal.resample_poly(data, 2, 1)

    # Write as 16kHz WAV
    sf.write(output_path, data_16k, 16000)


def process_channel_degradation(
    subset_csv: str = "data/subset_metadata.csv",
    output_dir: str = "data/subset_g711",
    output_combined_csv: str = "data/subset_combined_metadata.csv"
):
    print("=" * 60)
    print(" VoxGuard Channel / Codec Degradation Generator (G.711 mu-law 8kHz)")
    print("=" * 60)

    if not os.path.exists(subset_csv):
        raise FileNotFoundError(f"Subset CSV not found: {subset_csv}. Run create_subset.py first!")

    df = pd.read_csv(subset_csv)
    print(f"Loaded subset metadata: {len(df)} files across {df['speaker_id'].nunique()} speakers.")

    ffmpeg_bin = check_ffmpeg()
    if ffmpeg_bin:
        print(f"Using ffmpeg engine ('{ffmpeg_bin}') for G.711 mu-law encoding.")
    elif SOUNDFILE_AVAILABLE:
        print("Using Python 'soundfile + scipy' engine for G.711 mu-law encoding.")
    else:
        raise RuntimeError("Neither ffmpeg nor soundfile/scipy is available!")

    out_path_dir = Path(output_dir)
    out_path_dir.mkdir(parents=True, exist_ok=True)

    # Add channel tag to existing clean rows
    clean_df = df.copy()
    clean_df['channel'] = 'clean'

    degraded_records = []
    print(f"\nProcessing {len(df)} audio files into {output_dir}...")

    for idx, row in df.iterrows():
        # Determine original file path (prefer copied subset_filepath if present)
        src_path = row.get('subset_filepath', row['filepath'])
        if not os.path.exists(src_path):
            src_path = row['filepath']

        src_p = Path(src_path)
        dst_name = f"{src_p.stem}_g711.wav"
        dst_path = out_path_dir / dst_name

        try:
            if ffmpeg_bin:
                degrade_file_ffmpeg(ffmpeg_bin, str(src_p), str(dst_path))
            else:
                degrade_file_soundfile(str(src_p), str(dst_path))

            degraded_row = row.to_dict()
            degraded_row['filepath'] = dst_path.as_posix()
            if 'subset_filepath' in degraded_row:
                degraded_row['subset_filepath'] = dst_path.as_posix()
            degraded_row['channel'] = 'g711_8khz'
            degraded_records.append(degraded_row)

            if (idx + 1) % 100 == 0 or (idx + 1) == len(df):
                print(f"  Processed {idx + 1}/{len(df)} files...")

        except Exception as e:
            print(f"  ERROR processing {src_path}: {e}")

    degraded_df = pd.DataFrame(degraded_records)

    # Combine clean and degraded rows (600 + 600 = 1200 rows)
    combined_df = pd.concat([clean_df, degraded_df], ignore_index=True)

    # Save combined metadata
    combined_df.to_csv(output_combined_csv, index=False)

    print("\n" + "=" * 60)
    print(" Degradation Processing Complete!")
    print("=" * 60)
    print(f"Total Rows: {len(combined_df)} (600 clean + {len(degraded_df)} g711_8khz)")
    print(f"Combined Metadata saved to: {output_combined_csv}")
    print(f"G.711 Audio files stored in: {output_dir}/")
    print("\nSummary breakdown by split and channel:")
    print(combined_df.groupby(['split', 'channel', 'label']).size())
    print("=" * 60)


if __name__ == "__main__":
    process_channel_degradation()
