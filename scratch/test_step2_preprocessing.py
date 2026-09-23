"""
scratch/test_step2_preprocessing.py

Step 2 Verification: Tests preprocessing pipeline on bonafide & replay files,
along with edge cases (stereo, variable SR, short audio, silence).
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.detector_replay.preprocessing import (
    load_and_preprocess,
    chunk_waveform,
    is_silence,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_CHUNK_DURATION
)

try:
    import soundfile as sf
except ImportError:
    sf = None


def run_step2_tests():
    print("=" * 70)
    print(" VoxGuard Step 2 -- Audio Preprocessing & Chunking Verification")
    print("=" * 70)

    repaired_dir = PROJECT_ROOT / "data" / "subset_raw_repaired"
    metadata_file = PROJECT_ROOT / "data" / "subset_metadata.csv"

    df_meta = pd.read_csv(metadata_file)
    pa_bonafide = df_meta[(df_meta["dataset"] == "PA") & (df_meta["label"] == "bonafide")]
    pa_replay = df_meta[(df_meta["dataset"] == "PA") & (df_meta["label"] == "spoof")]

    # Find available files
    available_files = {p.name: p for p in repaired_dir.glob("*.flac")}

    # Select 3 bonafide and 3 replay files
    test_bonafide = []
    for _, row in pa_bonafide.iterrows():
        fn = Path(str(row["subset_filepath"])).name
        if fn in available_files:
            test_bonafide.append((available_files[fn], row["speaker_id"], row["label"]))
        if len(test_bonafide) == 3:
            break

    test_replay = []
    for _, row in pa_replay.iterrows():
        fn = Path(str(row["subset_filepath"])).name
        if fn in available_files:
            test_replay.append((available_files[fn], row["speaker_id"], row["label"], row["attack_id"]))
        if len(test_replay) == 3:
            break

    print(f"\n--- 1. Testing Bonafide Files ({len(test_bonafide)} files) ---")
    for fpath, spk, lbl in test_bonafide:
        raw_info = sf.info(str(fpath)) if sf else None
        orig_sr = raw_info.samplerate if raw_info else 16000
        orig_ch = raw_info.channels if raw_info else 1
        orig_dur = raw_info.duration if raw_info else 0.0

        audio, proc_sr = load_and_preprocess(fpath)
        chunks = chunk_waveform(audio, sample_rate=proc_sr, chunk_duration=2.0)

        print(f"File: {fpath.name}")
        print(f"  Speaker: {spk} | Label: {lbl}")
        print(f"  Original   : {orig_sr} Hz, {orig_ch} ch, {orig_dur:.3f} s ({len(audio)} raw samples)")
        print(f"  Processed  : {proc_sr} Hz, 1 ch, {len(audio)/proc_sr:.3f} s")
        print(f"  Chunks (2s): {len(chunks)} chunk(s), each shape {chunks[0].shape}, dtype {chunks[0].dtype}")
        print()

    print(f"--- 2. Testing Replay / Spoof Files ({len(test_replay)} files) ---")
    for fpath, spk, lbl, atk in test_replay:
        raw_info = sf.info(str(fpath)) if sf else None
        orig_sr = raw_info.samplerate if raw_info else 16000
        orig_ch = raw_info.channels if raw_info else 1
        orig_dur = raw_info.duration if raw_info else 0.0

        audio, proc_sr = load_and_preprocess(fpath)
        chunks = chunk_waveform(audio, sample_rate=proc_sr, chunk_duration=2.0)

        print(f"File: {fpath.name}")
        print(f"  Speaker: {spk} | Label: {lbl} (Attack Config: {atk})")
        print(f"  Original   : {orig_sr} Hz, {orig_ch} ch, {orig_dur:.3f} s")
        print(f"  Processed  : {proc_sr} Hz, 1 ch, {len(audio)/proc_sr:.3f} s")
        print(f"  Chunks (2s): {len(chunks)} chunk(s), each shape {chunks[0].shape}, dtype {chunks[0].dtype}")
        print()

    print("--- 3. Edge Cases & Synthetic Robustness ---")
    # Edge case A: Stereo input at 44.1 kHz
    sr_44k = 44100
    t_stereo = np.linspace(0, 3.5, int(sr_44k * 3.5), endpoint=False)
    ch1 = (0.5 * np.sin(2 * np.pi * 300 * t_stereo)).astype(np.float32)
    ch2 = (0.3 * np.cos(2 * np.pi * 600 * t_stereo)).astype(np.float32)
    stereo_audio = np.stack([ch1, ch2], axis=1)  # Shape: (samples, 2)

    proc_stereo, sr_out = load_and_preprocess(stereo_audio, orig_sr=sr_44k, target_sr=16000)
    chunks_stereo = chunk_waveform(proc_stereo, sample_rate=sr_out, chunk_duration=2.0)
    print(f"[Edge Case A] Stereo 44.1kHz (3.5s):")
    print(f"  Input  : shape {stereo_audio.shape} @ {sr_44k} Hz")
    print(f"  Output : shape {proc_stereo.shape} (mono) @ {sr_out} Hz")
    print(f"  Chunks : {len(chunks_stereo)} chunks of {len(chunks_stereo[0])} samples (2.0s each)")
    assert proc_stereo.ndim == 1, "Failed mono conversion"
    assert sr_out == 16000, "Failed resampling"
    assert len(chunks_stereo) == 2, f"Expected 2 chunks for 3.5s, got {len(chunks_stereo)}"
    print("  -> PASSED [OK]")

    # Edge case B: Short audio (0.8s < 2.0s chunk)
    sr_16k = 16000
    t_short = np.linspace(0, 0.8, int(sr_16k * 0.8), endpoint=False)
    short_audio = (0.4 * np.sin(2 * np.pi * 500 * t_short)).astype(np.float32)

    proc_short, _ = load_and_preprocess(short_audio, orig_sr=sr_16k)
    chunks_short = chunk_waveform(proc_short, sample_rate=16000, chunk_duration=2.0, pad_short=True)
    print(f"\n[Edge Case B] Short Audio 0.8s (< 2.0s):")
    print(f"  Input  : {len(short_audio)} samples ({0.8:.1f}s)")
    print(f"  Chunks : {len(chunks_short)} chunk, padded to {len(chunks_short[0])} samples (2.0s)")
    assert len(chunks_short[0]) == 32000, f"Expected 32000 padded samples, got {len(chunks_short[0])}"
    print("  -> PASSED [OK]")

    # Edge case C: 8 kHz telephone audio
    t_8k = np.linspace(0, 2.5, int(8000 * 2.5), endpoint=False)
    audio_8k = (0.3 * np.sin(2 * np.pi * 400 * t_8k)).astype(np.float32)
    proc_8k, sr_8k_out = load_and_preprocess(audio_8k, orig_sr=8000, target_sr=16000)
    print(f"\n[Edge Case C] 8kHz G.711 style audio:")
    print(f"  Input  : 8000 Hz, {len(audio_8k)} samples")
    print(f"  Output : {sr_8k_out} Hz, {len(proc_8k)} samples (upsampled to 16kHz)")
    assert sr_8k_out == 16000
    assert len(proc_8k) == 40000  # 2.5s * 16000
    print("  -> PASSED [OK]")

    # Edge case D: Silence detection
    silent_chunk = np.zeros(32000, dtype=np.float32)
    tone_chunk = (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 2.0, 32000))).astype(np.float32)
    print(f"\n[Edge Case D] Silence check:")
    print(f"  Silent chunk is_silence: {is_silence(silent_chunk)}")
    print(f"  Active tone chunk is_silence: {is_silence(tone_chunk)}")
    assert is_silence(silent_chunk) is True
    assert is_silence(tone_chunk) is False
    print("  -> PASSED [OK]")

    print("\n" + "=" * 70)
    print(" ALL PREPROCESSING TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_step2_tests()
    if not success:
        sys.exit(1)
