"""
models/train_replay_detector.py

Step 6: Trains the lightweight LogisticRegression Replay Attack Detector.
- Standardizes features using StandardScaler fitted strictly on X_train only.
- Validates model on validation split (without touching test set).
- Saves replay_model.pkl, scaler.pkl, and feature_config.json to models/replay_detector/.
"""

import os
import sys
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.split_replay_data import get_replay_splits
from modules.detector_replay.features import REPLAY_FEATURE_NAMES


def train_replay_model(
    output_model_dir: str = "models/replay_detector",
    C: float = 1.0,
    max_iter: int = 1000,
    class_weight: Optional[str] = "balanced",
    random_state: int = 42,
    solver: str = "lbfgs"
) -> Dict[str, Any]:
    print("=" * 75)
    print(" VoxGuard Step 6 -- Train LogisticRegression Replay Detector")
    print("=" * 75)

    # 1. Load strictly speaker-disjoint splits
    train_df, val_df, test_df = get_replay_splits()

    X_train_raw = train_df[REPLAY_FEATURE_NAMES].values
    y_train = train_df["label"].values.astype(int)

    X_val_raw = val_df[REPLAY_FEATURE_NAMES].values
    y_val = val_df["label"].values.astype(int)

    print(f"\n1. Training & Validation Dataset Sizes:")
    print(f"  - Training samples   : {len(X_train_raw)} chunks (Bonafide: {(y_train==0).sum()}, Replay: {(y_train==1).sum()})")
    print(f"  - Validation samples : {len(X_val_raw)} chunks (Bonafide: {(y_val==0).sum()}, Replay: {(y_val==1).sum()})")
    print(f"  - Number of Features : {len(REPLAY_FEATURE_NAMES)}")

    # 2. Strict Feature Scaling: Fit on X_train ONLY
    print(f"\n2. Feature Scaling (StandardScaler):")
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    print("  [OK] StandardScaler fitted STRICTLY on X_train.")

    X_train_scaled = scaler.transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    print("  [OK] X_train and X_val transformed using training scaler.")

    # 3. Model Configuration & Training
    model_config = {
        "classifier": "sklearn.linear_model.LogisticRegression",
        "C": C,
        "max_iter": max_iter,
        "class_weight": class_weight,
        "random_state": random_state,
        "solver": solver,
        "feature_count": len(REPLAY_FEATURE_NAMES),
        "feature_names": REPLAY_FEATURE_NAMES
    }

    print(f"\n3. Model Configuration:")
    for k, v in model_config.items():
        if k != "feature_names":
            print(f"  - {k}: {v}")

    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
        solver=solver
    )

    print("\nTraining LogisticRegression...")
    model.fit(X_train_scaled, y_train)
    print("  [OK] Model training complete.")

    # 4. Validation Performance (Tuning Check)
    val_probs = model.predict_proba(X_val_scaled)[:, 1]
    val_preds = (val_probs >= 0.50).astype(int)

    val_acc = accuracy_score(y_val, val_preds)
    val_prec = precision_score(y_val, val_preds, zero_division=0)
    val_rec = recall_score(y_val, val_preds, zero_division=0)
    val_f1 = f1_score(y_val, val_preds, zero_division=0)
    val_auc = roc_auc_score(y_val, val_probs)
    val_cm = confusion_matrix(y_val, val_preds)

    print(f"\n4. Validation Set Performance (82 held-out validation chunks across 4 unseen speakers):")
    print(f"  - Accuracy  : {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"  - Precision : {val_prec:.4f}")
    print(f"  - Recall    : {val_rec:.4f}")
    print(f"  - F1-Score  : {val_f1:.4f}")
    print(f"  - ROC-AUC   : {val_auc:.4f}")
    print(f"  - Confusion Matrix (TN, FP / FN, TP):")
    print(f"      [[{val_cm[0, 0]:2d} (TN), {val_cm[0, 1]:2d} (FP)]")
    print(f"       [{val_cm[1, 0]:2d} (FN), {val_cm[1, 1]:2d} (TP)]]")

    # 5. Learned Feature Coefficients & Interpretability
    print(f"\n5. Learned Model Coefficients (Feature Importance / Direction):")
    coefs = model.coef_[0]
    coef_df = pd.DataFrame({
        "Feature": REPLAY_FEATURE_NAMES,
        "Coefficient": coefs,
        "Abs_Importance": np.abs(coefs),
        "Direction": ["P(Replay) increases (+)" if c > 0 else "P(Bonafide) increases (-)" for c in coefs]
    }).sort_values(by="Abs_Importance", ascending=False)
    print(coef_df[["Feature", "Coefficient", "Direction"]].to_string(index=False))

    # 6. Save Model Artifacts
    out_dir = PROJECT_ROOT / output_model_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "replay_model.pkl"
    scaler_path = out_dir / "scaler.pkl"
    config_path = out_dir / "feature_config.json"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nSaved trained model to: {model_path.resolve()}")

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to       : {scaler_path.resolve()}")

    artifact_config = {
        **model_config,
        "classes": [0, 1],
        "class_names": {0: "bonafide", 1: "replay"},
        "intercept": float(model.intercept_[0]),
        "coefficients": {name: float(coef) for name, coef in zip(REPLAY_FEATURE_NAMES, coefs)},
        "validation_metrics": {
            "accuracy": float(val_acc),
            "precision": float(val_prec),
            "recall": float(val_rec),
            "f1_score": float(val_f1),
            "roc_auc": float(val_auc)
        }
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(artifact_config, f, indent=2)
    print(f"Saved feature config to: {config_path.resolve()}")

    print("\n" + "=" * 75)
    print(" STEP 6 TRAINING COMPLETE AND ARTIFACTS PERSISTED!")
    print("=" * 75)

    return {
        "model": model,
        "scaler": scaler,
        "config": artifact_config
    }


if __name__ == "__main__":
    train_replay_model()
