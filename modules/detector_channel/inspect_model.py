"""
modules/detector_channel/inspect_model.py

Inspects and prints human-readable details of modules/detector_channel/model.joblib.
Usage:
    python modules/detector_channel/inspect_model.py
"""

from pathlib import Path
import json
import joblib

MODEL_PATH = Path(__file__).parent / "model.joblib"
CONFIG_PATH = Path(__file__).parent / "feature_config.json"


def inspect_channel_model():
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print("=" * 70)
    print(" VoxGuard Channel Detector: Model Inspection")
    print("=" * 70)

    # 1. Load model with joblib
    model = joblib.load(MODEL_PATH)
    print(f"\nModel File: {MODEL_PATH.name}")
    print(f"Model Class: {model.__class__.__name__}")
    print(f"Number of Trees: {len(model.estimators_)}")
    print(f"Max Depth: {model.max_depth}")
    print(f"Classes: {model.classes_} (0 = clean, 1 = g711_8khz)")

    # 2. Load feature names from config
    feature_names = []
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            feature_names = cfg.get("feature_names", [])

    # 3. Print feature importances
    print("\nFeature Importances:")
    print("-" * 50)
    importances = model.feature_importances_
    for i, imp in enumerate(importances):
        name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        bar = "#" * int(imp * 40)
        print(f"  {name:24s} | {imp:7.4f} | {bar}")

    print("\nModel Parameters:")
    print("-" * 50)
    for k, v in model.get_params().items():
        print(f"  {k:24s}: {v}")

    print("=" * 70)


if __name__ == "__main__":
    inspect_channel_model()
