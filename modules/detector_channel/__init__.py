"""
modules/detector_channel/

VoxGuard Module: G.711 Telephone Codec Channel Detector
Detects whether an audio recording passed through a band-limited G.711 (8kHz) telephone codec
or is clean/unprocessed speech.
"""

from modules.detector_channel.features import (
    extract_channel_features,
    CHANNEL_FEATURE_NAMES,
)
from modules.detector_channel.infer import predict

__all__ = [
    "extract_channel_features",
    "CHANNEL_FEATURE_NAMES",
    "predict",
]
