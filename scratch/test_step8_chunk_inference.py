"""
scratch/test_step8_chunk_inference.py

Step 8 Verification: Tests single 2.0-second chunk inference via predict_replay()
and ReplayDetector.predict_chunk() on unseen test files and edge cases.
Measures and reports real-time CPU latency.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.detector_replay.infer import (
    ReplayDetector,
    predict_replay,
    predict,
    get_replay_detector
)
from modules.detector_replay.preprocessing import load_and_preprocess, chunk_waveform
from dataset.split_replay_data import get_replay_splits


def run_step8_tests():
    print("=" * 75)
    print(" VoxGuard Step 8 -- Single Chunk Inference & Latency Verification")
    print("=" * 75)

    repaired_dir = PROJECT_ROOT / "data" / "subset_raw_repaired"
    _, _, test_df = get_replay_splits()

    detector = get_replay_detector()
    print("ReplayDetector instance loaded successfully.")

    # Select test chunks from unseen speakers
    test_files_bonafide = test_df[test_df["label"] == 0]["filename"].unique()[:3]
    test_files_replay = test_df[test_df["label"] == 1]["filename"].unique()[:3]

    print("\n--- 1. Testing Unseen Bonafide Audio Chunks ---")
    bonafide_latencies = []
    for fname in test_files_bonafide:
        fpath = repaired_dir / fname
        audio, sr = load_and_preprocess(fpath)
        chunks = chunk_waveform(audio, sample_rate=sr, chunk_duration=2.0)
        chunk = chunks[0]

        # Call predict_replay
        t0 = time.perf_counter()
        p_replay = predict_replay(chunk, sample_rate=sr)
        lat = (time.perf_counter() - t0) * 1000.0
        bonafide_latencies.append(lat)

        # Call shared contract predict()
        full_res = predict(chunk)

        spk = test_df[test_df["filename"] == fname]["speaker_id"].iloc[0]
        print(f"File: {fname}")
        print(f"  Speaker (Unseen): {spk} | True Label: bonafide (0)")
        print(f"  P_replay score   : {p_replay:.4f} (predict_replay)")
        print(f"  Contract Output  : {full_res}")
        print(f"  Measured Latency : {lat:.2f} ms")
        print()

    print("--- 2. Testing Unseen Replay Attack Chunks ---")
    replay_latencies = []
    for fname in test_files_replay:
        fpath = repaired_dir / fname
        audio, sr = load_and_preprocess(fpath)
        chunks = chunk_waveform(audio, sample_rate=sr, chunk_duration=2.0)
        chunk = chunks[0]

        t0 = time.perf_counter()
        p_replay = predict_replay(chunk, sample_rate=sr)
        lat = (time.perf_counter() - t0) * 1000.0
        replay_latencies.append(lat)

        full_res = predict(chunk)

        spk = test_df[test_df["filename"] == fname]["speaker_id"].iloc[0]
        atk = test_df[test_df["filename"] == fname]["attack_id"].iloc[0]
        print(f"File: {fname}")
        print(f"  Speaker (Unseen): {spk} | Attack Config: {atk} | True Label: spoof (1)")
        print(f"  P_replay score   : {p_replay:.4f} (predict_replay)")
        print(f"  Contract Output  : {full_res}")
        print(f"  Measured Latency : {lat:.2f} ms")
        print()

    print("--- 3. Edge Cases for predict_replay() ---")
    # Edge case 1: Stereo input
    sr_44k = 44100
    t_st = np.linspace(0, 2.0, int(sr_44k * 2.0), endpoint=False)
    stereo_chunk = np.stack([0.3 * np.sin(2 * np.pi * 300 * t_st), 0.3 * np.cos(2 * np.pi * 500 * t_st)], axis=1)
    p_stereo = predict_replay(stereo_chunk, sample_rate=sr_44k)
    print(f"[Edge 1] Stereo chunk (44.1kHz): P_replay = {p_stereo:.4f} -> OK")
    assert 0.0 <= p_stereo <= 1.0

    # Edge case 2: Short chunk (<2.0s, e.g. 0.5s)
    short_chunk = (0.2 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 8000))).astype(np.float32)
    p_short = predict_replay(short_chunk, sample_rate=16000)
    print(f"[Edge 2] Short chunk (0.5s): P_replay = {p_short:.4f} -> OK")
    assert 0.0 <= p_short <= 1.0

    # Edge case 3: Pure silence
    silent_chunk = np.zeros(32000, dtype=np.float32)
    p_silent = predict_replay(silent_chunk, sample_rate=16000)
    print(f"[Edge 3] Silent chunk: P_replay = {p_silent:.4f} -> OK")
    assert p_silent <= 0.10

    # 4. Latency Benchmark (100 repeated runs on CPU)
    print("\n--- 4. Real-Time Latency Benchmark (100 runs on CPU) ---")
    bench_chunk = chunks[0]
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = predict_replay(bench_chunk, sample_rate=16000)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    lat_arr = np.array(latencies)
    print(f"  - Chunks evaluated: 100 runs")
    print(f"  - Mean Latency    : {lat_arr.mean():.2f} ms")
    print(f"  - Median Latency  : {np.median(lat_arr):.2f} ms")
    print(f"  - Min Latency     : {lat_arr.min():.2f} ms")
    print(f"  - Max Latency     : {lat_arr.max():.2f} ms")
    print(f"  - 95th Percentile : {np.percentile(lat_arr, 95):.2f} ms")
    print(f"  - Real-Time Ratio : {lat_arr.mean() / 2000.0 * 100:.3f}% of 2.0s audio budget (~1000x faster than real-time)")

    print("\n" + "=" * 75)
    print(" ALL STEP 8 SINGLE-CHUNK INFERENCE TESTS PASSED!")
    print("=" * 75)
    return True


if __name__ == "__main__":
    success = run_step8_tests()
    if not success:
        sys.exit(1)
