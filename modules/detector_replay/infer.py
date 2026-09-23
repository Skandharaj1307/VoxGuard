"""
modules/detector_replay/infer.py

VoxGuard Module: Replay Attack Detector (Signal-Level LogisticRegression)

Interface Contract:
- predict_replay(audio_chunk, sample_rate=16000) -> float in [0.0, 1.0]
- predict(audio_chunk) -> {"score": float, "top_feature": str}

Architecture:
- Lightweight, 13-dimensional signal-level acoustic feature extractor (spectral flatness,
  roll-off, reverberation proxy, spectral centroid/bandwidth, zero crossing rate, RMS energy)
- Pre-fitted StandardScaler (trained strictly on disjoint training split)
- Pre-trained sklearn LogisticRegression binary classifier
- $O(1)$ fast CPU inference (~2ms latency per 2.0-second chunk)
"""

import os
import sys
import json
import time
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

# Resolve repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.detector_replay.preprocessing import (
    load_and_preprocess,
    chunk_waveform,
    is_silence,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_CHUNK_DURATION
)
from modules.detector_replay.features import (
    extract_chunk_features,
    extract_chunk_features_dict,
    REPLAY_FEATURE_NAMES
)


class ReplayDetector:
    """Lightweight signal-level acoustic replay attack detector."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        scaler_path: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None
    ):
        models_dir = REPO_ROOT / "models" / "replay_detector"

        self.model_path = Path(model_path) if model_path else models_dir / "replay_model.pkl"
        self.scaler_path = Path(scaler_path) if scaler_path else models_dir / "scaler.pkl"
        self.config_path = Path(config_path) if config_path else models_dir / "feature_config.json"

        if not self.model_path.exists():
            raise FileNotFoundError(f"Replay model not found at: {self.model_path}")
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at: {self.scaler_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Feature config not found at: {self.config_path}")

        # Load model and scaler
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.feature_names = self.config.get("feature_names", REPLAY_FEATURE_NAMES)
        self.coefficients = self.config.get("coefficients", {})

    def _determine_top_feature(self, scaled_features: np.ndarray, p_replay: float) -> str:
        """Determines the primary explainability driver based on feature contributions."""
        if is_silence(scaled_features):
            return "silence baseline"

        # Contribution = scaled_feature * model_coefficient
        coef_vec = self.model.coef_[0]
        contributions = scaled_features * coef_vec

        if p_replay >= 0.50:
            # Look for highest positive contributor towards replay
            top_idx = int(np.argmax(contributions))
            top_feat_name = self.feature_names[top_idx]
            clean_name = top_feat_name.replace("_", " ")
            return f"elevated {clean_name} (acoustic replay signature)"
        else:
            # Look for highest negative contributor towards bonafide
            top_idx = int(np.argmin(contributions))
            top_feat_name = self.feature_names[top_idx]
            clean_name = top_feat_name.replace("_", " ")
            return f"natural {clean_name} (direct bonafide speech profile)"

    def predict_chunk(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int = DEFAULT_SAMPLE_RATE
    ) -> Dict[str, Any]:
        """
        Inference on a single ~2.0-second audio chunk.

        Args:
            audio_chunk: 1D or 2D numpy array.
            sample_rate: Sample rate in Hz (default: 16000).

        Returns:
            Dict containing:
                "score": float in [0.0, 1.0] representing P_replay
                "top_feature": str explainability description
                "latency_ms": measured execution time in milliseconds
        """
        t0 = time.perf_counter()

        # 1. Preprocess & ensure 1D 16kHz float32
        chunk_clean, sr = load_and_preprocess(audio_chunk, orig_sr=sample_rate, target_sr=DEFAULT_SAMPLE_RATE)

        # Ensure ~32,000 samples (2.0s)
        target_samples = int(DEFAULT_CHUNK_DURATION * DEFAULT_SAMPLE_RATE)
        if len(chunk_clean) < target_samples:
            chunk_clean = np.pad(chunk_clean, (0, target_samples - len(chunk_clean)), mode="constant")
        elif len(chunk_clean) > target_samples:
            chunk_clean = chunk_clean[:target_samples]

        # Check silence
        if is_silence(chunk_clean):
            t_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "score": 0.05,
                "top_feature": "silent audio chunk (silence baseline)",
                "latency_ms": round(t_ms, 3)
            }

        # 2. Extract 13 signal-level features
        raw_feats = extract_chunk_features(chunk_clean, sample_rate=DEFAULT_SAMPLE_RATE)
        feats_2d = raw_feats.reshape(1, -1)

        # 3. Scale features using fitted scaler
        scaled_feats = self.scaler.transform(feats_2d)[0]

        # 4. Predict probability (P_replay = class 1)
        prob_replay = float(self.model.predict_proba(scaled_feats.reshape(1, -1))[0, 1])

        # 5. Explainability
        top_feature = self._determine_top_feature(scaled_feats, prob_replay)

        t_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "score": round(prob_replay, 4),
            "top_feature": top_feature,
            "latency_ms": round(t_ms, 3)
        }

    def predict_file(
        self,
        audio_input: Union[str, Path, np.ndarray],
        sample_rate: Optional[int] = None,
        aggregation: str = "mean",
        chunk_duration: float = DEFAULT_CHUNK_DURATION
    ) -> Dict[str, Any]:
        """
        Inference across a complete audio file or long recording.
        Segments audio into 2.0s chunks, scores each chunk, and aggregates.

        Args:
            audio_input: Path to audio file or raw numpy array.
            sample_rate: Sample rate if numpy array is passed.
            aggregation: Aggregation method ("mean", "max", or "median"). Default: "mean".
            chunk_duration: Chunk size in seconds (default: 2.0s).

        Returns:
            Dict containing aggregated score, top feature, chunk count, and chunk scores.
        """
        t0 = time.perf_counter()
        audio, sr = load_and_preprocess(audio_input, orig_sr=sample_rate, target_sr=DEFAULT_SAMPLE_RATE)
        chunks = chunk_waveform(audio, sample_rate=sr, chunk_duration=chunk_duration, pad_short=True)

        if not chunks:
            return {
                "score": 0.05,
                "top_feature": "empty audio input",
                "chunk_count": 0,
                "chunk_scores": [],
                "latency_ms": 0.0
            }

        chunk_scores: List[float] = []
        chunk_features: List[str] = []

        for chunk in chunks:
            res = self.predict_chunk(chunk, sample_rate=sr)
            chunk_scores.append(res["score"])
            chunk_features.append(res["top_feature"])

        # Aggregate chunk probabilities
        if aggregation == "max":
            agg_score = float(np.max(chunk_scores))
        elif aggregation == "median":
            agg_score = float(np.median(chunk_scores))
        else:  # "mean" baseline
            agg_score = float(np.mean(chunk_scores))

        # Select representative top feature (from chunk with highest score)
        max_idx = int(np.argmax(chunk_scores))
        top_feature = chunk_features[max_idx]

        t_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "score": round(agg_score, 4),
            "top_feature": top_feature,
            "chunk_count": len(chunks),
            "chunk_scores": chunk_scores,
            "latency_ms": round(t_ms, 3)
        }


# Global lazy singleton
_GLOBAL_REPLAY_DETECTOR: Optional[ReplayDetector] = None


def get_replay_detector() -> ReplayDetector:
    """Returns or initializes the global singleton ReplayDetector instance."""
    global _GLOBAL_REPLAY_DETECTOR
    if _GLOBAL_REPLAY_DETECTOR is None:
        _GLOBAL_REPLAY_DETECTOR = ReplayDetector()
    return _GLOBAL_REPLAY_DETECTOR


def predict_replay(
    audio_chunk: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE
) -> float:
    """
    Public Function: Returns scalar probability P_replay in [0.0, 1.0].

    Args:
        audio_chunk: 1D or 2D numpy array.
        sample_rate: Sample rate in Hz (default: 16000).

    Returns:
        float: Probability that the audio chunk contains a replay attack.
    """
    detector = get_replay_detector()
    result = detector.predict_chunk(audio_chunk, sample_rate=sample_rate)
    return result["score"]


def predict(
    audio_chunk: np.ndarray
) -> Dict[str, Any]:
    """
    Module-level inference function implementing the shared VoxGuard detector contract.

    predict(audio_chunk) -> {"score": float, "top_feature": str}
    """
    detector = get_replay_detector()
    res = detector.predict_chunk(audio_chunk)
    return {
        "score": res["score"],
        "top_feature": res["top_feature"]
    }


def predict_audio_file(
    audio_path: Union[str, Path],
    aggregation: str = "mean"
) -> Dict[str, Any]:
    """
    Module-level function: Runs full-audio inference by chunking into 2.0s segments,
    predicting per chunk, and aggregating.
    """
    detector = get_replay_detector()
    return detector.predict_file(audio_path, aggregation=aggregation)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python modules/detector_replay/infer.py <path_to_audio_file> [mean|max|median]")
        sys.exit(1)

    input_file = sys.argv[1]
    agg_mode = sys.argv[2] if len(sys.argv) > 2 else "mean"

    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    detector = get_replay_detector()
    result = detector.predict_file(input_file, aggregation=agg_mode)

    print("\n--- Replay Attack Detector Inference Result ---")
    print(json.dumps(result, indent=2))

