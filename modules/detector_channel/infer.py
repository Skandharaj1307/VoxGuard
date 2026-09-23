"""
modules/detector_channel/infer.py

VoxGuard Module: G.711 Telephone Channel Inference Engine.
Predicts whether an audio file or chunk is clean or G.711-degraded.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Optional, Union
import numpy as np
import joblib

# Resolve repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.detector_channel.features import (
    extract_channel_features,
    CHANNEL_FEATURE_NAMES,
)

DEFAULT_MODEL_PATH = Path(__file__).parent / "model.joblib"
DEFAULT_PKL_PATH = Path(__file__).parent / "model.pkl"
DEFAULT_CONFIG_PATH = Path(__file__).parent / "feature_config.json"

_CACHED_MODEL = None
_CACHED_CONFIG = None


def load_model_and_config(
    model_path: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
):
    """
    Loads and caches the channel detector model and feature configuration.
    """
    global _CACHED_MODEL, _CACHED_CONFIG

    if _CACHED_MODEL is not None and _CACHED_CONFIG is not None:
        return _CACHED_MODEL, _CACHED_CONFIG

    m_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not m_path.exists():
        # Fallback to model.pkl if model.joblib does not exist
        if DEFAULT_PKL_PATH.exists():
            m_path = DEFAULT_PKL_PATH
        else:
            raise FileNotFoundError(
                f"Model file not found at {m_path}. "
                "Please run modules/detector_channel/train.py first."
            )

    c_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not c_path.exists():
        raise FileNotFoundError(
            f"Feature config not found at {c_path}. "
            "Please run modules/detector_channel/train.py first."
        )

    model = joblib.load(m_path)
    with open(c_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    _CACHED_MODEL = model
    _CACHED_CONFIG = config
    return _CACHED_MODEL, _CACHED_CONFIG


def determine_top_feature(
    feature_dict: Dict[str, float],
    score: float,
    config: Dict,
) -> str:
    """
    Determines the single feature that most influenced the prediction.
    Combines global feature importance with per-sample deviation from baseline.
    """
    feature_names = config.get("feature_names", CHANNEL_FEATURE_NAMES)
    global_top = config.get("top_feature_global", feature_names[0])
    ref_stats = config.get("reference_stats", {})

    if not ref_stats:
        return global_top

    # Calculate normalized deviation towards predicted class
    max_dev = -1.0
    best_feature = global_top

    for feat in feature_names:
        stats = ref_stats.get(feat)
        if not stats:
            continue

        val = feature_dict.get(feat, 0.0)
        c_mean = stats.get("clean_mean", 0.0)
        c_std = stats.get("clean_std", 1.0)
        g_mean = stats.get("g711_mean", 0.0)
        g_std = stats.get("g711_std", 1.0)

        # Distance from clean vs distance from g711
        z_from_clean = abs(val - c_mean) / c_std
        z_from_g711 = abs(val - g_mean) / g_std

        if score >= 0.5:
            # G.711 prediction: feature that moved furthest away from clean towards G.711
            dev = z_from_clean - z_from_g711
        else:
            # Clean prediction: feature that moved furthest away from G.711 towards clean
            dev = z_from_g711 - z_from_clean

        if dev > max_dev:
            max_dev = dev
            best_feature = feat

    return best_feature


def predict(audio_chunk_path: Union[str, Path, np.ndarray]) -> dict:
    """
    Loads the trained channel classifier and predicts whether the given
    audio file/chunk is clean or g711-degraded.

    Returns:
        {
            "score": float,           # probability of being g711/degraded (0.0-1.0)
            "top_feature": str,       # the single feature that most influenced this prediction
            "predicted_channel": str  # "clean" or "g711_8khz"
        }
    """
    model, config = load_model_and_config()
    feature_names = config.get("feature_names", CHANNEL_FEATURE_NAMES)

    # 1. Extract 7 diagnostic channel features
    feats_dict = extract_channel_features(audio_chunk_path)

    # 2. Prepare feature vector in exact configured order
    feat_vector = np.array(
        [[feats_dict[name] for name in feature_names]], dtype=np.float32
    )

    # 3. Model inference (probability of positive class: g711_8khz)
    proba = model.predict_proba(feat_vector)[0]
    classes = list(model.classes_)

    # If classes are [0, 1], positive index is where class == 1
    if 1 in classes:
        g711_idx = classes.index(1)
    else:
        g711_idx = 1 if len(classes) > 1 else 0

    score = float(proba[g711_idx])
    predicted_channel = "g711_8khz" if score >= 0.50 else "clean"

    # 4. Determine top explaining feature
    top_feature = determine_top_feature(feats_dict, score, config)

    return {
        "score": round(score, 4),
        "top_feature": top_feature,
        "predicted_channel": predicted_channel,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict channel degradation (clean vs G.711) for an audio file"
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to audio file (.flac, .wav, etc.) to analyze",
    )
    args = parser.parse_args()

    result = predict(args.audio_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
