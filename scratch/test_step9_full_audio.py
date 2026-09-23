"""
scratch/test_step9_full_audio.py

Step 9 Verification: Complete Audio File Inference & Multi-Chunk Aggregation
- Tests: complete audio -> 2-second chunks -> chunk probabilities -> aggregate score.
- Tests aggregation strategies: 'mean', 'max', 'median'.
- Tests direct CLI execution: python modules/detector_replay/infer.py <audio_file>
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.detector_replay.infer import (
    ReplayDetector,
    predict_audio_file,
    get_replay_detector
)
from dataset.split_replay_data import get_replay_splits


def run_step9_tests():
    print("=" * 75)
    print(" VoxGuard Step 9 -- Complete Audio File Inference & Aggregation")
    print("=" * 75)

    repaired_dir = PROJECT_ROOT / "data" / "subset_raw_repaired"
    _, _, test_df = get_replay_splits()

    detector = get_replay_detector()

    # Pick 2 bonafide and 2 replay test files
    sample_files_bonafide = test_df[test_df["label"] == 0]["filename"].unique()[:2]
    sample_files_replay = test_df[test_df["label"] == 1]["filename"].unique()[:2]

    all_test_files = [
        (repaired_dir / fn, "bonafide", 0, test_df[test_df["filename"] == fn]["speaker_id"].iloc[0], "-")
        for fn in sample_files_bonafide
    ] + [
        (repaired_dir / fn, "replay", 1, test_df[test_df["filename"] == fn]["speaker_id"].iloc[0], test_df[test_df["filename"] == fn]["attack_id"].iloc[0])
        for fn in sample_files_replay
    ]

    print("\n1. Testing Full Audio File Inference Across Aggregation Modes:")
    print("-" * 75)

    results_table = []
    for fpath, label_str, label_int, spk, atk in all_test_files:
        res_mean = detector.predict_file(fpath, aggregation="mean")
        res_max = detector.predict_file(fpath, aggregation="max")
        res_med = detector.predict_file(fpath, aggregation="median")

        print(f"\nFile: {fpath.name}")
        print(f"  Speaker (Unseen): {spk} | True Label: {label_str} (Attack: {atk})")
        print(f"  Chunks analyzed : {res_mean['chunk_count']} chunks (2.0s each)")
        print(f"  Chunk trace     : {res_mean['chunk_scores']}")
        print(f"  Aggregated Scores:")
        print(f"    - Mean   : {res_mean['score']:.4f}")
        print(f"    - Max    : {res_max['score']:.4f}")
        print(f"    - Median : {res_med['score']:.4f}")
        print(f"  Top Feature     : '{res_mean['top_feature']}'")
        print(f"  Total Latency   : {res_mean['latency_ms']:.2f} ms")

        results_table.append({
            "File": fpath.name[:22] + "...",
            "Speaker": spk,
            "True Label": label_str,
            "Chunks": res_mean["chunk_count"],
            "Mean Score": res_mean["score"],
            "Max Score": res_max["score"],
            "Median Score": res_med["score"],
            "Top Feature": res_mean["top_feature"][:25] + "..."
        })

    print("\n" + "-" * 75)
    print("Aggregation Summary Table:")
    print(pd.DataFrame(results_table).to_string(index=False))

    # 2. Test Direct CLI Execution
    print("\n" + "-" * 75)
    print("2. Testing Direct CLI Execution: python modules/detector_replay/infer.py")
    print("-" * 75)

    test_cli_file = all_test_files[0][0]
    cmd = [sys.executable, "modules/detector_replay/infer.py", str(test_cli_file), "mean"]
    print(f"Running command: {' '.join(cmd)}")

    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("CLI Output:")
    print(proc.stdout)

    assert proc.returncode == 0, f"CLI exited with error: {proc.stderr}"
    assert "score" in proc.stdout, "CLI output did not contain 'score'"
    assert "top_feature" in proc.stdout, "CLI output did not contain 'top_feature'"
    print("  -> CLI EXECUTION VERIFIED! [PASS]")

    print("\n" + "=" * 75)
    print(" ALL STEP 9 FULL AUDIO INFERENCE TESTS PASSED!")
    print("=" * 75)
    return True


if __name__ == "__main__":
    success = run_step9_tests()
    if not success:
        sys.exit(1)
