"""
modules/detector_replay/features.py

Feature extraction pipeline for VoxGuard Replay Attack Detection.

Implements lightweight, interpretable signal-level acoustic features:
1. Spectral Flatness (mean, std)
2. Spectral Roll-off (mean, std)
3. Reverberation Proxy (late-to-early energy decay ratio)
4. Spectral Centroid (mean, std)
5. Spectral Bandwidth (mean, std)
6. Zero Crossing Rate (mean, std)
7. RMS Energy (mean, std)

Note on Reverberation:
We compute a signal-level energy decay proxy (late_to_early_energy_ratio / reverb_proxy)
based on short-time temporal envelope decay. This captures room reflection signatures
and secondary acoustic channel artifacts without making calibrated physical room claims.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


# Ordered list of feature column names (Single Source of Truth)
REPLAY_FEATURE_NAMES: List[str] = [
    "spectral_flatness_mean",
    "spectral_flatness_std",
    "spectral_rolloff_mean",
    "spectral_rolloff_std",
    "reverb_proxy_decay_ratio",
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",
    "zero_crossing_rate_mean",
    "zero_crossing_rate_std",
    "rms_energy_mean",
    "rms_energy_std",
]


def compute_reverb_proxy(
    audio: np.ndarray,
    sr: int = 16000,
    frame_length: int = 512,
    hop_length: int = 256,
    late_threshold_ms: float = 50.0,
) -> float:
    """
    Computes a signal-level reverberation proxy: late-to-early energy decay ratio.

    Acoustic Intuition:
    Replayed audio recorded in real environments contains room reflections and speaker-enclosure
    resonances that cause lingering energy after phoneme onsets / speech energy peaks.
    We compute the short-time energy envelope and measure the ratio of energy persisting in the
    'late' window (>50ms after local energy peaks) relative to the 'early' direct window (<=50ms).

    Returns:
        float: Dimensionless ratio representing late-to-early energy decay.
    """
    if len(audio) < frame_length:
        return 0.0

    # 1. Compute frame-level RMS energy envelope
    if LIBROSA_AVAILABLE:
        rms_env = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    else:
        # Fallback frame RMS computation
        num_frames = 1 + (len(audio) - frame_length) // hop_length
        rms_env = np.array([
            np.sqrt(np.mean(audio[i * hop_length : i * hop_length + frame_length] ** 2))
            for i in range(num_frames)
        ], dtype=np.float32)

    if len(rms_env) < 4 or np.max(rms_env) < 1e-6:
        return 0.0

    # 2. Identify local energy onset peaks
    peak_indices = []
    for i in range(1, len(rms_env) - 1):
        if rms_env[i] > rms_env[i - 1] and rms_env[i] > rms_env[i + 1] and rms_env[i] > 0.2 * np.max(rms_env):
            peak_indices.append(i)

    if not peak_indices:
        # If no distinct peaks, compare overall second-half to first-half energy decay
        mid = len(rms_env) // 2
        early_energy = np.sum(rms_env[:mid] ** 2) + 1e-9
        late_energy = np.sum(rms_env[mid:] ** 2) + 1e-9
        return float(np.clip(late_energy / early_energy, 0.0, 10.0))

    # Frame duration in milliseconds
    frame_ms = (hop_length / sr) * 1000.0
    split_frames = int(max(1, round(late_threshold_ms / frame_ms)))

    early_energy_total = 1e-9
    late_energy_total = 1e-9

    for p in peak_indices:
        # Early direct energy: peak to peak + split_frames
        early_end = min(len(rms_env), p + split_frames)
        early_energy_total += np.sum(rms_env[p:early_end] ** 2)

        # Late reflection energy: peak + split_frames to peak + 3*split_frames
        late_end = min(len(rms_env), p + 3 * split_frames)
        if late_end > early_end:
            late_energy_total += np.sum(rms_env[early_end:late_end] ** 2)

    ratio = late_energy_total / early_energy_total
    return float(np.clip(ratio, 0.0, 10.0))


def extract_chunk_features(
    audio_chunk: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 512,
    roll_percent: float = 0.85,
) -> np.ndarray:
    """
    Extracts fixed 1D vector of 13 signal-level replay features from a single 2-second chunk.

    Parameters:
        audio_chunk: 1D float32 numpy array (typically 32,000 samples @ 16kHz).
        sample_rate: Sample rate in Hz (default: 16000).
        n_fft: FFT window size (default: 1024).
        hop_length: Hop length in samples (default: 512).
        roll_percent: Roll-off cumulative energy percentage (default: 0.85).

    Returns:
        1D float32 numpy array of length len(REPLAY_FEATURE_NAMES) = 13.
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for acoustic feature extraction.")

    # Guard for empty or purely silent chunk
    if len(audio_chunk) == 0:
        return np.zeros(len(REPLAY_FEATURE_NAMES), dtype=np.float32)

    # 1. Spectral Flatness
    flatness = librosa.feature.spectral_flatness(
        y=audio_chunk, n_fft=n_fft, hop_length=hop_length
    )[0]
    flatness_mean = float(np.mean(flatness))
    flatness_std = float(np.std(flatness))

    # 2. Spectral Roll-off (high-frequency decay)
    rolloff = librosa.feature.spectral_rolloff(
        y=audio_chunk, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, roll_percent=roll_percent
    )[0]
    rolloff_mean = float(np.mean(rolloff))
    rolloff_std = float(np.std(rolloff))

    # 3. Reverberation Proxy (energy decay ratio)
    reverb_proxy = compute_reverb_proxy(audio_chunk, sr=sample_rate, frame_length=n_fft, hop_length=hop_length)

    # 4. Spectral Centroid
    centroid = librosa.feature.spectral_centroid(
        y=audio_chunk, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
    )[0]
    centroid_mean = float(np.mean(centroid))
    centroid_std = float(np.std(centroid))

    # 5. Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio_chunk, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
    )[0]
    bandwidth_mean = float(np.mean(bandwidth))
    bandwidth_std = float(np.std(bandwidth))

    # 6. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(
        y=audio_chunk, frame_length=n_fft, hop_length=hop_length
    )[0]
    zcr_mean = float(np.mean(zcr))
    zcr_std = float(np.std(zcr))

    # 7. RMS Energy
    rms = librosa.feature.rms(
        y=audio_chunk, frame_length=n_fft, hop_length=hop_length
    )[0]
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))

    feature_values = [
        flatness_mean,
        flatness_std,
        rolloff_mean,
        rolloff_std,
        reverb_proxy,
        centroid_mean,
        centroid_std,
        bandwidth_mean,
        bandwidth_std,
        zcr_mean,
        zcr_std,
        rms_mean,
        rms_std,
    ]

    return np.array(feature_values, dtype=np.float32)


def extract_chunk_features_dict(
    audio_chunk: np.ndarray,
    sample_rate: int = 16000,
) -> Dict[str, float]:
    """
    Convenience helper: returns feature name-value dictionary for a chunk.
    """
    feats = extract_chunk_features(audio_chunk, sample_rate=sample_rate)
    return {name: float(val) for name, val in zip(REPLAY_FEATURE_NAMES, feats)}
