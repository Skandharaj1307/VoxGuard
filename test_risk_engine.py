"""
Test script for VoxGuard Risk Engine module.

Verifies compute_risk behavior across various scenarios:
  1. Low Risk (Authentic human voice) -> Expect 'allow'
  2. Mixed / Medium Risk (Channel anomaly & replay) -> Expect 'warn'
  3. High Risk (Deepfake synthetic speech attack) -> Expect 'block'
  4. Boundary cases (0.40 and 0.70 thresholds)
  5. Error validation (out-of-bounds input values)
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path when running script directly
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.risk_engine.engine import compute_risk


def run_tests():
    print("=" * 70)
    print(" VoxGuard Risk Engine Verification Test Suite")
    print("=" * 70)

    test_cases = [
        {
            "name": "Scenario 1: Low Risk (Authentic Audio)",
            "p_ai": 0.05,
            "p_replay": 0.08,
            "p_channel": 0.10,
            # Expected: 0.40*0.05 + 0.35*0.08 + 0.25*0.10 = 0.02 + 0.028 + 0.025 = 0.073
            "expected_decision": "allow",
            "expected_irs_max": 0.40,
        },
        {
            "name": "Scenario 2: Mixed / Moderate Risk (Suspicious Channel & Replay)",
            "p_ai": 0.15,
            "p_replay": 0.65,
            "p_channel": 0.70,
            # Expected: 0.40*0.15 + 0.35*0.65 + 0.25*0.70 = 0.06 + 0.2275 + 0.175 = 0.4625
            "expected_decision": "warn",
            "expected_irs_min": 0.40,
            "expected_irs_max": 0.70,
        },
        {
            "name": "Scenario 3: High Risk (AI Deepfake Voice Clone)",
            "p_ai": 0.95,
            "p_replay": 0.80,
            "p_channel": 0.65,
            # Expected: 0.40*0.95 + 0.35*0.80 + 0.25*0.65 = 0.38 + 0.28 + 0.1625 = 0.8225
            "expected_decision": "block",
            "expected_irs_min": 0.70,
        },
        {
            "name": "Scenario 4: Boundary Threshold Exact 0.40",
            "p_ai": 0.40,
            "p_replay": 0.40,
            "p_channel": 0.40,
            # Expected IRS: 0.40 -> "warn" (0.40 to 0.70)
            "expected_decision": "warn",
            "expected_irs_exact": 0.40,
        },
        {
            "name": "Scenario 5: Boundary Threshold Exact 0.70",
            "p_ai": 0.70,
            "p_replay": 0.70,
            "p_channel": 0.70,
            # Expected IRS: 0.70 -> "warn" (0.40 to 0.70)
            "expected_decision": "warn",
            "expected_irs_exact": 0.70,
        },
        {
            "name": "Scenario 6: Boundary Just Above 0.70",
            "p_ai": 0.75,
            "p_replay": 0.70,
            "p_channel": 0.70,
            # Expected: 0.40*0.75 + 0.35*0.70 + 0.25*0.70 = 0.30 + 0.245 + 0.175 = 0.72 -> "block"
            "expected_decision": "block",
            "expected_irs_min": 0.70,
        },
    ]

    all_passed = True

    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        p_ai, p_replay, p_channel = case["p_ai"], case["p_replay"], case["p_channel"]
        print(f"Inputs  : P_ai={p_ai}, P_replay={p_replay}, P_channel={p_channel}")

        result = compute_risk(p_ai, p_replay, p_channel)
        irs = result["irs"]
        decision = result["decision"]
        reasons = result["reasons"]

        print(f"Result  : IRS = {irs:.4f} | Decision = '{decision}'")
        print("Reasons :")
        for r in reasons:
            print(f"  - {r}")

        # Assertions
        passed = True
        if decision != case["expected_decision"]:
            print(f"  FAILED: Expected decision '{case['expected_decision']}', got '{decision}'")
            passed = False

        if "expected_irs_exact" in case and abs(irs - case["expected_irs_exact"]) > 1e-4:
            print(f"  FAILED: Expected IRS {case['expected_irs_exact']}, got {irs}")
            passed = False

        if "expected_irs_max" in case and irs > case["expected_irs_max"]:
            print(f"  FAILED: Expected IRS <= {case['expected_irs_max']}, got {irs}")
            passed = False

        if "expected_irs_min" in case and irs < case["expected_irs_min"]:
            print(f"  FAILED: Expected IRS >= {case['expected_irs_min']}, got {irs}")
            passed = False

        if passed:
            print("  Status: PASSED [OK]")
        else:
            all_passed = False

    # Input validation tests
    print("\n--- Scenario 7: Input Validation / Error Handling ---")
    error_cases = [
        ("Negative value (-0.1)", -0.1, 0.5, 0.5, ValueError),
        ("Value exceeding 1.0 (1.5)", 0.2, 1.5, 0.5, ValueError),
        ("String input ('0.5')", "0.5", 0.5, 0.5, TypeError),
    ]

    for desc, ai, replay, channel, expected_err in error_cases:
        try:
            compute_risk(ai, replay, channel)
            print(f"  FAILED: Expected {expected_err.__name__} for {desc}, but no exception was raised.")
            all_passed = False
        except expected_err as e:
            print(f"  PASSED [OK]: Correctly caught {expected_err.__name__} for {desc}: {e}")

    print("\n" + "=" * 70)
    if all_passed:
        print(" ALL TESTS PASSED SUCCESSFULLY!")
    else:
        print(" SOME TESTS FAILED!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
