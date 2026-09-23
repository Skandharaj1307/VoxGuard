"""
test_risk_integration.py
========================
Verification test suite for the VoxGuard Risk Engine Integration Pipeline.

Tests:
  1. Finds an existing audio sample in the repository.
  2. Passes the audio filepath to `run_full_detection()`.
  3. Executes Module 2 (AI Voice), Module 3 (Replay), and Module 4 (Channel).
  4. Verifies each returned score is a real numeric value in [0.0, 1.0].
  5. Verifies Module 5 produces IRS, decision, and reasons.
  6. Strict partial-failure enforcement: if any module lacks weights/dependencies,
     fails cleanly with the exact failure diagnostic without fallback scores.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.risk_engine.integration import run_full_detection


def find_test_audio_sample() -> Path:
    """Finds an existing audio sample in the repository for integration testing."""
    search_dirs = [
        PROJECT_ROOT / "data" / "subset_raw_repaired",
        PROJECT_ROOT / "data" / "subset_raw",
        PROJECT_ROOT / "data",
    ]

    # Prefer .wav if available, otherwise .flac
    for d in search_dirs:
        if d.is_dir():
            wav_files = list(d.glob("*.wav"))
            if wav_files:
                return wav_files[0]

    for d in search_dirs:
        if d.is_dir():
            flac_files = list(d.glob("*.flac"))
            if flac_files:
                return flac_files[0]

    # Search entire repo recursively for any audio file
    all_audio = list(PROJECT_ROOT.rglob("*.wav")) + list(PROJECT_ROOT.rglob("*.flac"))
    if all_audio:
        return all_audio[0]

    raise FileNotFoundError(
        "No audio sample (.wav or .flac) found in the repository for integration testing."
    )


def run_integration_test():
    print("=" * 75)
    print(" VoxGuard Multi-Detector Pipeline Integration Test")
    print("=" * 75)

    try:
        sample_path = find_test_audio_sample()
        print(f"Target Audio Sample: {sample_path.relative_to(PROJECT_ROOT)}")
    except FileNotFoundError as err:
        print(f"\nFATAL: {err}")
        sys.exit(1)

    print("\nExecuting End-to-End Pipeline: run_full_detection()...")

    try:
        result = run_full_detection(sample_path)
    except RuntimeError as err:
        print("\n" + "!" * 75)
        print(" PIPELINE INTEGRATION FAILURE (Strict Partial-Failure Enforced)")
        print("!" * 75)
        print(f"\n{err}\n")
        print("Zero dummy/fallback scores were injected. Pipeline halted as intended.")
        print("=" * 75)
        sys.exit(1)
    except Exception as err:
        print("\n" + "!" * 75)
        print(f" UNEXPECTED PIPELINE FAILURE: {type(err).__name__}")
        print("!" * 75)
        print(f"\n{err}\n")
        sys.exit(1)

    # Validate output structure and scores
    scores = result.get("scores", {})
    p_ai = scores.get("p_ai")
    p_replay = scores.get("p_replay")
    p_channel = scores.get("p_channel")

    risk_result = result.get("risk_result", {})
    irs = risk_result.get("irs")
    decision = risk_result.get("decision")
    reasons = risk_result.get("reasons")

    # Score range assertions
    for name, val in [("P_ai", p_ai), ("P_replay", p_replay), ("P_channel", p_channel)]:
        assert isinstance(val, (int, float)), f"{name} must be numeric, got {type(val)}"
        assert 0.0 <= val <= 1.0, f"{name} must be in [0.0, 1.0], got {val}"

    # Risk result assertions
    assert isinstance(irs, float) and 0.0 <= irs <= 1.0, f"IRS must be float in [0, 1], got {irs}"
    assert decision in {"allow", "warn", "block"}, f"Invalid decision: {decision}"
    assert isinstance(reasons, list) and len(reasons) > 0, "Reasons must be non-empty list"

    # Print success report
    print("\n" + "-" * 75)
    print(" REAL DETECTOR OUTPUTS:")
    print(f"   Module 2 (AI Voice) : P_ai      = {p_ai:.4f}")
    print(f"   Module 3 (Replay)   : P_replay  = {p_replay:.4f}")
    print(f"   Module 4 (Channel)  : P_channel = {p_channel:.4f}")
    print("-" * 75)
    print(" FINAL RISK ENGINE RESULT:")
    print(f"   IRS Score           : {irs:.4f}")
    print(f"   Security Decision   : {decision.upper()}")
    print("   Diagnostic Reasons  :")
    for r in reasons:
        print(f"     * {r}")
    print("=" * 75)
    print(" INTEGRATION TEST PASSED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    run_integration_test()
