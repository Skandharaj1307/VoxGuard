"""
models/evaluate_replay_detector.py

Step 7: Evaluates the trained Replay Attack Detector strictly on the held-out test set.
Computes:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC
- Confusion Matrix (TN, FP, FN, TP)
- False Positive Rate (FPR) and False Negative Rate (FNR)
- Equal Error Rate (EER) and optimal threshold
- Chunk-level and File-level aggregated metrics
- Saves metrics to models/replay_detector/metrics.json
"""

import os
import sys
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.split_replay_data import get_replay_splits
from modules.detector_replay.features import REPLAY_FEATURE_NAMES


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    Computes Equal Error Rate (EER) and the associated threshold where FPR == FNR.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr

    # Find index where |fpr - fnr| is minimized
    idx_opt = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx_opt] + fnr[idx_opt]) / 2.0)
    eer_threshold = float(thresholds[idx_opt])

    return eer, eer_threshold


def evaluate_replay_detector(
    model_path: str = "models/replay_detector/replay_model.pkl",
    scaler_path: str = "models/replay_detector/scaler.pkl",
    output_metrics_json: str = "models/replay_detector/metrics.json"
) -> Dict[str, Any]:
    print("=" * 75)
    print(" VoxGuard Step 7 -- Held-Out Test Set Evaluation")
    print("=" * 75)

    model_p = PROJECT_ROOT / model_path
    scaler_p = PROJECT_ROOT / scaler_path
    out_metrics_p = PROJECT_ROOT / output_metrics_json

    if not model_p.exists() or not scaler_p.exists():
        raise FileNotFoundError(f"Model or scaler artifact not found at {model_p} / {scaler_p}")

    with open(model_p, "rb") as f:
        model = pickle.load(f)
    with open(scaler_p, "rb") as f:
        scaler = pickle.load(f)

    # 1. Load held-out test data only
    _, _, test_df = get_replay_splits()
    print(f"\n1. Held-Out Test Set Profile:")
    print(f"  - Total test chunks  : {len(test_df)} (from {test_df['filename'].nunique()} files)")
    print(f"  - Unseen speakers    : {test_df['speaker_id'].nunique()} {sorted(list(test_df['speaker_id'].unique()))}")
    print(f"  - Bonafide chunks (0): {(test_df['label'] == 0).sum()} ({test_df[test_df['label']==0]['filename'].nunique()} files)")
    print(f"  - Replay chunks (1)  : {(test_df['label'] == 1).sum()} ({test_df[test_df['label']==1]['filename'].nunique()} files)")

    # 2. Scale features using PRE-FITTED training scaler
    X_test_raw = test_df[REPLAY_FEATURE_NAMES].values
    y_test_chunks = test_df["label"].values.astype(int)

    X_test_scaled = scaler.transform(X_test_raw)

    # 3. Chunk-Level Predictions
    chunk_probs = model.predict_proba(X_test_scaled)[:, 1]
    chunk_preds_05 = (chunk_probs >= 0.50).astype(int)

    acc = float(accuracy_score(y_test_chunks, chunk_preds_05))
    prec = float(precision_score(y_test_chunks, chunk_preds_05, zero_division=0))
    rec = float(recall_score(y_test_chunks, chunk_preds_05, zero_division=0))
    f1 = float(f1_score(y_test_chunks, chunk_preds_05, zero_division=0))
    auc = float(roc_auc_score(y_test_chunks, chunk_probs))
    cm = confusion_matrix(y_test_chunks, chunk_preds_05)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    eer, eer_threshold = compute_eer(y_test_chunks, chunk_probs)

    print(f"\n2. Chunk-Level Held-Out Test Results (Threshold = 0.50):")
    print(f"  - Accuracy        : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  - Precision       : {prec:.4f} ({prec*100:.2f}%)")
    print(f"  - Recall          : {rec:.4f} ({rec*100:.2f}%)")
    print(f"  - F1-Score        : {f1:.4f}")
    print(f"  - ROC-AUC         : {auc:.4f}")
    print(f"  - Equal Error Rate: {eer:.4f} ({eer*100:.2f}% @ threshold = {eer_threshold:.4f})")
    print(f"  - False Pos Rate  : {fpr:.4f} ({fpr*100:.2f}%)  [Bonafide falsely flagged as replay]")
    print(f"  - False Neg Rate  : {fnr:.4f} ({fnr*100:.2f}%)  [Replay attack missed]")

    print(f"\n3. Chunk-Level Confusion Matrix:")
    print(f"                   Actual Bonafide (0)   Actual Replay (1)")
    print(f"  Pred Bonafide:       {tn:3d} (TN)             {fn:3d} (FN)")
    print(f"  Pred Replay  :       {fp:3d} (FP)             {tp:3d} (TP)")

    # 4. File-Level Aggregated Evaluation (Averaging chunks per audio file)
    test_df["pred_prob"] = chunk_probs
    file_agg = test_df.groupby("filename").agg({
        "label": "first",
        "pred_prob": "mean",
        "speaker_id": "first",
        "attack_id": "first"
    }).reset_index()

    y_test_files = file_agg["label"].values.astype(int)
    file_probs = file_agg["pred_prob"].values
    file_preds_05 = (file_probs >= 0.50).astype(int)

    file_acc = float(accuracy_score(y_test_files, file_preds_05))
    file_prec = float(precision_score(y_test_files, file_preds_05, zero_division=0))
    file_rec = float(recall_score(y_test_files, file_preds_05, zero_division=0))
    file_f1 = float(f1_score(y_test_files, file_preds_05, zero_division=0))
    file_auc = float(roc_auc_score(y_test_files, file_probs))
    file_eer, file_eer_th = compute_eer(y_test_files, file_probs)
    file_cm = confusion_matrix(y_test_files, file_preds_05)

    print(f"\n4. File-Level Aggregated Evaluation (17 unique audio files):")
    print(f"  - Accuracy        : {file_acc:.4f} ({file_acc*100:.2f}%)")
    print(f"  - Precision       : {file_prec:.4f}")
    print(f"  - Recall          : {file_rec:.4f}")
    print(f"  - F1-Score        : {file_f1:.4f}")
    print(f"  - ROC-AUC         : {file_auc:.4f}")
    print(f"  - Equal Error Rate: {file_eer:.4f} ({file_eer*100:.2f}%)")
    print(f"  - File Confusion Matrix (TN, FP / FN, TP):")
    print(f"      [[{file_cm[0, 0]:2d} (TN), {file_cm[0, 1]:2d} (FP)]")
    print(f"       [{file_cm[1, 0]:2d} (FN), {file_cm[1, 1]:2d} (TP)]]")

    # 5. Security & Trade-Off Analysis
    print(f"\n5. Security Analysis (False Positives vs False Negatives):")
    print(f"  - False Positives (FP = {fp}): Genuine user speech is flagged as replay.")
    print(f"    -> Impact: Causes user friction, routing genuine callers to step-up authentication.")
    print(f"  - False Negatives (FN = {fn}): Replay attack is allowed as authentic speech.")
    print(f"    -> Impact: Security vulnerability (impersonation bypass). In the VoxGuard architecture,")
    print(f"       this risk is mitigated by multi-signal fusion (IRS combines P_ai, P_replay, P_channel).")

    # 6. Save Metrics JSON
    metrics_data = {
        "evaluation_dataset": "ASVspoof2019_PA_held_out_test",
        "evaluation_type": "speaker_disjoint_test_split",
        "chunk_level": {
            "test_chunks_count": len(y_test_chunks),
            "bonafide_chunks": int((y_test_chunks == 0).sum()),
            "replay_chunks": int((y_test_chunks == 1).sum()),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc,
            "equal_error_rate": eer,
            "eer_threshold": eer_threshold,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "confusion_matrix": {
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp
            }
        },
        "file_level": {
            "test_files_count": len(y_test_files),
            "bonafide_files": int((y_test_files == 0).sum()),
            "replay_files": int((y_test_files == 1).sum()),
            "accuracy": file_acc,
            "precision": file_prec,
            "recall": file_rec,
            "f1_score": file_f1,
            "roc_auc": file_auc,
            "equal_error_rate": file_eer,
            "eer_threshold": file_eer_th,
            "confusion_matrix": {
                "true_negatives": int(file_cm[0, 0]),
                "false_positives": int(file_cm[0, 1]),
                "false_negatives": int(file_cm[1, 0]),
                "true_positives": int(file_cm[1, 1])
            }
        }
    }

    with open(out_metrics_p, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"\nSaved metrics report to: {out_metrics_p.resolve()}")

    print("\n" + "=" * 75)
    print(" STEP 7 EVALUATION COMPLETE!")
    print("=" * 75)

    return metrics_data


if __name__ == "__main__":
    evaluate_replay_detector()
