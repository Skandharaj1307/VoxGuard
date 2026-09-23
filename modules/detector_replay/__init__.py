"""
VoxGuard Replay Attack Detector Module.

Exports:
- predict: Module-level inference function implementing shared detector contract:
           predict(audio_chunk) -> {"score": float, "top_feature": str}
- predict_replay: Scalar probability function: predict_replay(audio_chunk, sr=16000) -> float
- predict_audio_file: Multi-chunk complete audio file prediction
- ReplayDetector: Detector class wrapper
"""

from .infer import (
    predict,
    predict_replay,
    predict_audio_file,
    ReplayDetector,
    get_replay_detector
)

__all__ = [
    "predict",
    "predict_replay",
    "predict_audio_file",
    "ReplayDetector",
    "get_replay_detector"
]
