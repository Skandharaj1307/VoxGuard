"""
modules/detector_channel/train.py

Training pipeline for VoxGuard G.711 Telephone Channel Detector.
Loads data/channel_features.csv, strictly preserves existing train/val/test splits,
trains an interpretable classifier, evaluates performance, and saves model artifacts.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Any
import pandas as pd
import numpy as np
import joblib

# Resolve repository root
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from modules.detector_channel.features import CHANNEL_FEATURE_NAMES

DEFAULT_DATASET_CSV = REPO_ROOT / "data" / "channel_features.csv"
DEFAULT_MODEL_PATH = Path(__file__).parent / "model.joblib"
DEFAULT_CONFIG_PATH = Path(__file__).parent / "feature_config.json"


def train_channel_detector(
    dataset_csv: Path = DEFAULT_DATASET_CSV,
    model_output_path: Path = DEFAULT_MODEL_PATH,
    config_output_path: Path = DEFAULT_CONFIG_PATH,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Trains a channel detector classifier and saves model artifacts.
    """
    dataset_path = Path(dataset_csv)
    if not dataset_path.is_file():
        dataset_path = REPO_ROOT / dataset_path
        if not dataset_path.is_file():
            raise FileNotFoundError(
                f"Features dataset not found at {dataset_csv}. "
                "Please run modules/detector_channel/build_dataset.py first."
            )

    df = pd.read_csv(dataset_path)

    # Validate required columns
    required_cols = ["split", "channel"] + CHANNEL_FEATURE_NAMES
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in dataset: '{col}'")

    # Map target channel: clean -> 0, g711_8khz -> 1
    # Verify split column exactly as-is
    train_mask = df["split"].str.strip().str.lower() == "train"
    val_mask = df["split"].str.strip().str.lower() == "val"
    test_mask = df["split"].str.strip().str.lower() == "test"

    train_df = df[train_mask]
    val_df = df[val_mask]
    test_df = df[test_mask]

    X_train = train_df[CHANNEL_FEATURE_NAMES].values.astype(np.float32)
    y_train = (train_df["channel"].str.strip().str.lower() == "g711_8khz").astype(int).values

    X_val = val_df[CHANNEL_FEATURE_NAMES].values.astype(np.float32)
    y_val = (val_df["channel"].str.strip().str.lower() == "g711_8khz").astype(int).values

    X_test = test_df[CHANNEL_FEATURE_NAMES].values.astype(np.float32)
    y_test = (test_df["channel"].str.strip().str.lower() == "g711_8khz").astype(int).values

    # Train Random Forest Classifier
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Evaluate on validation split
    y_val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_prec = precision_score(y_val, y_val_pred, zero_division=0)
    val_rec = recall_score(y_val, y_val_pred, zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred, zero_division=0)

    # Evaluate on test split
    y_test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred, zero_division=0)
    test_rec = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

    # Feature importances
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    feature_importance_dict = {
        CHANNEL_FEATURE_NAMES[i]: float(importances[i]) for i in sorted_idx
    }
    top_feature_global = CHANNEL_FEATURE_NAMES[sorted_idx[0]]

    # Compute baseline reference statistics per feature for explainability
    ref_stats = {}
    for feat in CHANNEL_FEATURE_NAMES:
        ref_stats[feat] = {
            "clean_mean": float(train_df[train_df["channel"] == "clean"][feat].mean()),
            "clean_std": float(train_df[train_df["channel"] == "clean"][feat].std() + 1e-9),
            "g711_mean": float(train_df[train_df["channel"] == "g711_8khz"][feat].mean()),
            "g711_std": float(train_df[train_df["channel"] == "g711_8khz"][feat].std() + 1e-9),
        }

    # Print requested output format exactly
    print("Channel Classifier Training")
    print("---------------------------")
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    print()
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation Precision: {val_prec:.4f}")
    print(f"Validation Recall: {val_rec:.4f}")
    print(f"Validation F1-score: {val_f1:.4f}")
    print()
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_rec:.4f}")
    print(f"Test F1-score: {test_f1:.4f}")
    print()
    print("Feature Importances:")
    for feat_name, imp in feature_importance_dict.items():
        print(f"  {feat_name:24s}: {imp:.4f}")
    print()

    # Save model and config
    model_path = Path(model_output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"Model saved to: {model_path.as_posix()}")

    # Also save as model.pkl for compatibility
    pkl_path = model_path.with_suffix(".pkl")
    if pkl_path != model_path:
        joblib.dump(clf, pkl_path)
        print(f"Model alias saved to: {pkl_path.as_posix()}")

    config_path = Path(config_output_path)
    config_data = {
        "feature_names": CHANNEL_FEATURE_NAMES,
        "positive_class": "g711_8khz",
        "negative_class": "clean",
        "model_type": "RandomForestClassifier",
        "top_feature_global": top_feature_global,
        "feature_importances": feature_importance_dict,
        "reference_stats": ref_stats,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    print(f"Feature config saved to: {config_path.as_posix()}")

    return {
        "model": clf,
        "val_metrics": {
            "accuracy": val_acc,
            "precision": val_prec,
            "recall": val_rec,
            "f1": val_f1,
        },
        "test_metrics": {
            "accuracy": test_acc,
            "precision": test_prec,
            "recall": test_rec,
            "f1": test_f1,
        },
        "feature_importances": feature_importance_dict,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train VoxGuard G.711 Telephone Channel Detector"
    )
    parser.add_argument(
        "--dataset-csv",
        type=str,
        default=str(DEFAULT_DATASET_CSV),
        help="Path to channel features CSV",
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to save trained model",
    )
    parser.add_argument(
        "--config-output",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to save feature configuration JSON",
    )
    args = parser.parse_args()

    train_channel_detector(
        dataset_csv=Path(args.dataset_csv),
        model_output_path=Path(args.model_output),
        config_output_path=Path(args.config_output),
    )


if __name__ == "__main__":
    main()
