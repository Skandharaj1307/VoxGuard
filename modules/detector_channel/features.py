"""
modules/detector_channel/features.py

Acoustic Feature Extraction for G.711 Telephone Channel Degradation Detection.

Extracts 7 signal-level spectral features specifically diagnostic of G.711 codec artifacts:
1. high_freq_energy_ratio: Ratio of spectral energy above ~3.4 kHz to total energy (near zero for G.711)
2. low_freq_energy_ratio: Ratio of spectral energy below ~300 Hz to total energy
3. spectral_rolloff: Frequency below which 85% of spectral energy is contained (librosa.feature.spectral_rolloff)
4. spectral_centroid: Center of mass of the spectrum (librosa.feature.spectral_centroid)
5. spectral_bandwidth: Spread of the spectrum around the centroid (librosa.feature.spectral_bandwidth)
6. spectral_flatness: Quantifies tonal vs noise-like quality (captures companding quantization distortion)
7. spectral_flux: Normalized frame-to-frame spectral change measure as a proxy for quantization noise
"""

from pathlib import Path
from typing import Dict, List, Union
import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from dataset.extract_features import load_audio

CHANNEL_FEATURE_NAMES: List[str] = [
    "high_freq_energy_ratio",
    "low_freq_energy_ratio",
    "spectral_rolloff",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_flatness",
    "spectral_flux",
]


def extract_channel_features(
    filepath: Union[str, Path, np.ndarray],
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> Dict[str, float]:
    """
    Extracts 7 diagnostic channel features for G.711 telephone codec detection.

    Parameters:
        filepath: Path to audio file (.flac, .wav, etc.) or 1D float32 audio waveform.
        sample_rate: Target sample rate in Hz (default: 16000).
        n_fft: STFT window size (default: 1024).
        hop_length: STFT hop length (default: 512).

    Returns:
        Dict mapping feature names to scalar floats (mean across frames):
        {
            "high_freq_energy_ratio": float,
            "low_freq_energy_ratio": float,
            "spectral_rolloff": float,
            "spectral_centroid": float,
            "spectral_bandwidth": float,
            "spectral_flatness": float,
            "spectral_flux": float,
        }
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for channel feature extraction.")

    # 1. Load audio if path provided, or use waveform array directly
    if isinstance(filepath, np.ndarray):
        audio = filepath.astype(np.float32)
        sr = sample_rate
    else:
        audio, sr = load_audio(filepath, target_sr=sample_rate)

    # Guard for empty or purely silent audio
    if len(audio) == 0 or float(np.max(np.abs(audio))) < 1e-9:
        return {name: 0.0 for name in CHANNEL_FEATURE_NAMES}

    # 2. STFT representation
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(D)
    power = mag ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # 3. Frequency band energy ratios (G.711 telephone passband is ~300 Hz to 3400 Hz)
    high_mask = freqs >= 3400.0
    low_mask = freqs <= 300.0

    total_energy_per_frame = np.sum(power, axis=0)
    high_energy_per_frame = np.sum(power[high_mask, :], axis=0)
    low_energy_per_frame = np.sum(power[low_mask, :], axis=0)

    valid_energy_frames = total_energy_per_frame > 1e-12
    if np.any(valid_energy_frames):
        high_freq_energy_ratio = float(
            np.mean(high_energy_per_frame[valid_energy_frames] / total_energy_per_frame[valid_energy_frames])
        )
        low_freq_energy_ratio = float(
            np.mean(low_energy_per_frame[valid_energy_frames] / total_energy_per_frame[valid_energy_frames])
        )
    else:
        high_freq_energy_ratio = 0.0
        low_freq_energy_ratio = 0.0

    # 4. Standard spectral shape descriptors
    rolloff = librosa.feature.spectral_rolloff(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85
    )[0]
    spectral_rolloff = float(np.mean(rolloff))

    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]
    spectral_centroid = float(np.mean(centroid))

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]
    spectral_bandwidth = float(np.mean(bandwidth))

    flatness = librosa.feature.spectral_flatness(
        y=audio, n_fft=n_fft, hop_length=hop_length
    )[0]
    spectral_flatness = float(np.mean(flatness))

    # 5. Normalized spectral flux (frame-to-frame spectral variation)
    mag_sum = np.sum(mag, axis=0, keepdims=True)
    valid_cols = mag_sum[0] > 1e-12
    if np.sum(valid_cols) > 1:
        mag_norm = mag[:, valid_cols] / mag_sum[:, valid_cols]
        diff = np.diff(mag_norm, axis=1)
        flux = np.sqrt(np.sum(diff ** 2, axis=0))
        spectral_flux = float(np.mean(flux))
    else:
        spectral_flux = 0.0

    return {
        "high_freq_energy_ratio": high_freq_energy_ratio,
        "low_freq_energy_ratio": low_freq_energy_ratio,
        "spectral_rolloff": spectral_rolloff,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_flatness": spectral_flatness,
        "spectral_flux": spectral_flux,
    }
