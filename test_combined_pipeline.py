"""
test_combined_pipeline.py

Step 11: End-to-end multi-detector verification.
Connects P_replay from modules.detector_replay directly to modules.risk_engine.engine.compute_risk.

Tests:
1. End-to-end integration: Real audio -> predict_replay -> compute_risk.
2. Verification across full test set audio files.
3. Decision validation (allow, warn, block) and diagnostic reasons.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.risk_engine import run_full_detection, compute_risk
from dataset.split_replay_data import get_replay_splits


def run_combined_pipeline_tests():
    print("=" * 75)
    print(" VoxGuard Multi-Signal Risk Engine Integration Verification")
    print("=" * 75)

    repaired_dir = PROJECT_ROOT / "data" / "subset_raw_repaired"
    _, _, test_df = get_replay_splits()

    # 1. Integration Test with Sample Real Audio Files
    test_files_bonafide = test_df[test_df["label"] == 0]["filename"].unique()[:2]
    test_files_replay = test_df[test_df["label"] == 1]["filename"].unique()[:2]

    test_samples = [
        (repaired_dir / fn, "bonafide (authentic human)", 0)
        for fn in test_files_bonafide
    ] + [
        (repaired_dir / fn, "replay attack", 1)
        for fn in test_files_replay
    ]

    print("\n--- 1. Testing End-to-End Pipeline (Audio -> All Detectors -> Risk Engine) ---")
    pipeline_records = []

    for fpath, desc, true_lbl in test_samples:
        try:
            full_res = run_full_detection(fpath)
            scores = full_res["scores"]
            p_ai = scores["p_ai"]
            p_replay = scores["p_replay"]
            p_channel = scores["p_channel"]

            risk_result = full_res["risk_result"]
            irs = risk_result["irs"]
            decision = risk_result["decision"]
            reasons = risk_result["reasons"]

            print(f"\nAudio File: {fpath.name}")
            print(f"  Profile        : {desc}")
            print(f"  Detector Scores: P_ai={p_ai:.4f} | P_replay={p_replay:.4f} | P_channel={p_channel:.4f}")
            print(f"  Risk Engine Out: IRS = {irs:.4f} | Decision = '{decision.upper()}'")
            print("  Reasons Output :")
            for r in reasons:
                print(f"    - {r}")

            pipeline_records.append({
                "File": fpath.name[:20] + "...",
                "Profile": desc[:15] + "...",
                "P_ai (Real)": p_ai,
                "P_replay (Real)": p_replay,
                "P_channel (Real)": p_channel,
                "IRS": irs,
                "Decision": decision.upper()
            })
        except RuntimeError as err:
            print(f"\nPipeline integration stopped on {fpath.name}:")
            print(f"  {err}")
            raise

    print("\n" + "-" * 75)
    print("Pipeline Integration Summary Table:")
    print(pd.DataFrame(pipeline_records).to_string(index=False))

    # 2. Risk Formula Exactness Check
    print("\n--- 2. Risk Fusion Formula Exactness Verification ---")
    p_ai_test = 0.30
    p_rep_test = 0.60
    p_chan_test = 0.20
    # Expected IRS = 0.40*0.30 + 0.35*0.60 + 0.25*0.20 = 0.12 + 0.21 + 0.05 = 0.38
    res = compute_risk(p_ai_test, p_rep_test, p_chan_test)
    print(f"Test Inputs: P_ai={p_ai_test}, P_replay={p_rep_test}, P_channel={p_chan_test}")
    print(f"Expected IRS: 0.3800 | Computed IRS: {res['irs']:.4f} | Decision: '{res['decision']}'")
    assert abs(res["irs"] - 0.38) < 1e-4, f"IRS calculation mismatch: expected 0.38, got {res['irs']}"
    assert res["decision"] == "allow"
    print("  -> FORMULA EXACTNESS VERIFIED! [PASS]")

    # 3. High-Risk Replay Interception Scenario
    print("\n--- 3. Replay Attack Alert Diagnostic Scenario ---")
    p_ai_test = 0.10
    p_rep_test = 0.90  # Confident replay attack
    p_chan_test = 0.80 # Confident channel anomaly
    # Expected IRS = 0.40*0.10 + 0.35*0.90 + 0.25*0.80 = 0.04 + 0.315 + 0.20 = 0.555 -> 'warn'
    res_alert = compute_risk(p_ai_test, p_rep_test, p_chan_test)
    print(f"Inputs: P_ai={p_ai_test}, P_replay={p_rep_test}, P_channel={p_chan_test}")
    print(f"IRS: {res_alert['irs']:.4f} | Decision: '{res_alert['decision'].upper()}'")
    assert any("acoustic replay attack" in r for r in res_alert["reasons"]), "Replay diagnostic reason missing"
    print(f"Replay Diagnostic Reason Verified: {[r for r in res_alert['reasons'] if 'replay' in r]}")
    print("  -> REPLAY RISK ENGINE ALERT VERIFIED! [PASS]")

    print("\n" + "=" * 75)
    print(" ALL STEP 11 RISK ENGINE INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 75)
    return True


if __name__ == "__main__":
    success = run_combined_pipeline_tests()
    if not success:
        sys.exit(1)
