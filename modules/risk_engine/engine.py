"""
VoxGuard Risk Engine Module
===========================
Combines spoofing detection scores across three specialized detectors
(AI synthetic voice, acoustic replay, and transmission channel anomalies)
into a unified Integrated Risk Score (IRS) and provides decision recommendations
along with explainable diagnostic reasons.
"""

from typing import Dict, List, Any


# -----------------------------------------------------------------------------
# Configuration & Model Weights
# -----------------------------------------------------------------------------
# Weights assigned based on detector sensitivity and spoof threat severity.
# Total weights sum to 1.00.
WEIGHT_AI: float = 0.40       # Weight for AI synthetic speech / deepfake detector
WEIGHT_REPLAY: float = 0.35   # Weight for acoustic replay detector
WEIGHT_CHANNEL: float = 0.25  # Weight for transmission channel / codec detector

# Decision Thresholds
THRESHOLD_ALLOW: float = 0.40  # IRS strictly below 0.40 -> 'allow'
THRESHOLD_BLOCK: float = 0.70  # IRS strictly above 0.70 -> 'block', [0.40, 0.70] -> 'warn'

# Feature alert threshold for generating granular explainability reasons
DETECTOR_ALERT_THRESHOLD: float = 0.50


# -----------------------------------------------------------------------------
# Core Risk Computation
# -----------------------------------------------------------------------------
def compute_risk(p_ai: float, p_replay: float, p_channel: float) -> Dict[str, Any]:
    """
    Computes the Integrated Risk Score (IRS) and determines the security action.

    Formula:
        IRS = (0.40 * P_ai) + (0.35 * P_replay) + (0.25 * P_channel)

    Decision Rules:
        - IRS < 0.40:  "allow" (Low risk / authentic voice)
        - 0.40 <= IRS <= 0.70: "warn"  (Medium risk / suspicious activity)
        - IRS > 0.70:  "block" (High risk / spoofing detected)

    Args:
        p_ai (float): Probability of synthetic / AI-generated voice [0.0, 1.0].
        p_replay (float): Probability of acoustic replay attack [0.0, 1.0].
        p_channel (float): Probability of transmission channel anomaly [0.0, 1.0].

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'irs' (float): Integrated Risk Score rounded to 4 decimal places.
            - 'decision' (str): Recommended action ('allow', 'warn', or 'block').
            - 'reasons' (List[str]): Explanations and diagnostic highlights.

    Raises:
        ValueError: If any input score is outside the valid range [0.0, 1.0].
        TypeError: If any input score is not a numeric value (int or float).
    """
    # 1. Input Validation
    scores = {"P_ai": p_ai, "P_replay": p_replay, "P_channel": p_channel}
    for name, score in scores.items():
        if not isinstance(score, (int, float)):
            raise TypeError(f"Score '{name}' must be numeric (int or float), got {type(score).__name__}.")
        if not (0.0 <= score <= 1.0):
            raise ValueError(f"Score '{name}' must be between 0.0 and 1.0, got {score}.")

    # Convert to float
    p_ai = float(p_ai)
    p_replay = float(p_replay)
    p_channel = float(p_channel)

    # 2. Compute Integrated Risk Score (IRS)
    raw_irs = (WEIGHT_AI * p_ai) + (WEIGHT_REPLAY * p_replay) + (WEIGHT_CHANNEL * p_channel)
    irs = round(raw_irs, 4)

    # 3. Determine Decision Action
    if irs < THRESHOLD_ALLOW:
        decision = "allow"
    elif irs <= THRESHOLD_BLOCK:
        decision = "warn"
    else:
        decision = "block"

    # 4. Generate Explainable Reasons
    reasons: List[str] = []

    # Primary decision explanation
    if decision == "allow":
        reasons.append(f"Overall risk score (IRS = {irs:.4f}) is below the caution threshold ({THRESHOLD_ALLOW:.2f}).")
    elif decision == "warn":
        reasons.append(
            f"Overall risk score (IRS = {irs:.4f}) falls within the warning range "
            f"[{THRESHOLD_ALLOW:.2f}, {THRESHOLD_BLOCK:.2f}]. Further verification is advised."
        )
    else:  # decision == "block"
        reasons.append(
            f"Overall risk score (IRS = {irs:.4f}) exceeds the critical threshold "
            f"({THRESHOLD_BLOCK:.2f}). Access blocked."
        )

    # Granular detector-level diagnostics
    flagged_detectors = False
    if p_ai >= DETECTOR_ALERT_THRESHOLD:
        reasons.append(f"High probability of AI synthetic speech detected (P_ai = {p_ai:.1%}).")
        flagged_detectors = True

    if p_replay >= DETECTOR_ALERT_THRESHOLD:
        reasons.append(f"High probability of acoustic replay attack detected (P_replay = {p_replay:.1%}).")
        flagged_detectors = True

    if p_channel >= DETECTOR_ALERT_THRESHOLD:
        reasons.append(f"High transmission channel / acoustic environment anomaly detected (P_channel = {p_channel:.1%}).")
        flagged_detectors = True

    # If allowed and no detectors triggered alerts, note healthy state
    if not flagged_detectors and decision == "allow":
        reasons.append("All detector confidence scores (AI, Replay, Channel) are within normal baseline ranges.")

    return {
        "irs": irs,
        "decision": decision,
        "reasons": reasons
    }


# -----------------------------------------------------------------------------
# Standalone Quick Execution / Smoke Test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print(" VoxGuard Risk Engine - Quick Diagnostic Run")
    print("=" * 60)

    sample_runs = [
        ("Low Risk (Authentic Speech)", 0.05, 0.08, 0.12),
        ("Moderate Risk (Suspicious Channel & Replay)", 0.30, 0.65, 0.60),
        ("High Risk (AI Voice Clone / Deepfake)", 0.95, 0.85, 0.70),
    ]

    for label, p_ai, p_replay, p_channel in sample_runs:
        print(f"\nScenario: {label}")
        print(f"Inputs  : P_ai={p_ai:.2f}, P_replay={p_replay:.2f}, P_channel={p_channel:.2f}")
        result = compute_risk(p_ai, p_replay, p_channel)
        print(f"Result  : IRS={result['irs']} | Decision={result['decision'].upper()}")
        print("Reasons :")
        for reason in result["reasons"]:
            print(f"  - {reason}")
