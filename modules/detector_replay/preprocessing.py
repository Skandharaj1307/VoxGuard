"""
modules/detector_replay/preprocessing.py

Audio preprocessing and chunking pipeline for the VoxGuard Replay Attack Detector.

Key Responsibilities:
1. Load audio files or accept numpy arrays.
2. Standardize audio: mono conversion, float32, resampled to 16,000 Hz.
3. Handle short audio (< 2.0s), long audio, silence, stereo, and variable sample rates.
4. Segment audio into ~2.0 second chunks (32,000 samples @ 16 kHz) for analysis.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np

# Audio processing fallback imports
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    import scipy.signal as signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


DEFAULT_SAMPLE_RATE: int = 16000
DEFAULT_CHUNK_DURATION: float = 2.0  # seconds (32,000 samples at 16kHz)
SILENCE_RMS_THRESHOLD: float = 1e-5


def standardize_waveform(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    """
    Standardizes in-memory audio array:
    - Multi-channel (stereo) converted to mono via channel averaging
    - Cast to float32
    - Resampled to target_sr if different
    """
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio, dtype=np.float32)

    # 1. Convert multichannel / stereo to mono
    if audio.ndim > 1:
        if audio.shape[0] > audio.shape[1]:
            # Shape: (samples, channels)
            audio = np.mean(audio, axis=1)
        else:
            # Shape: (channels, samples)
            audio = np.mean(audio, axis=0)

    # 2. Ensure float32 contiguous 1D
    audio = np.ascontiguousarray(audio, dtype=np.float32)

    # 3. Resample if necessary
    if orig_sr != target_sr:
        if LIBROSA_AVAILABLE:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        elif SCIPY_AVAILABLE:
            gcd = np.gcd(orig_sr, target_sr)
            up = target_sr // gcd
            down = orig_sr // gcd
            audio = signal.resample_poly(audio, up, down).astype(np.float32)
        else:
            raise RuntimeError(
                f"Cannot resample from {orig_sr} Hz to {target_sr} Hz without librosa or scipy."
            )

    return audio.astype(np.float32)


def load_and_preprocess(
    audio_input: Union[str, Path, np.ndarray],
    orig_sr: Optional[int] = None,
    target_sr: int = DEFAULT_SAMPLE_RATE,
) -> Tuple[np.ndarray, int]:
    """
    Loads an audio file or standardizes a raw array into a 16kHz mono float32 array.

    Args:
        audio_input: File path or raw 1D/2D numpy array.
        orig_sr: Original sample rate if audio_input is a numpy array.
        target_sr: Target sample rate (default: 16000 Hz).

    Returns:
        Tuple of (processed_waveform_1d, sample_rate)
    """
    if isinstance(audio_input, (str, Path)):
        path_obj = Path(audio_input)
        if not path_obj.is_file():
            raise FileNotFoundError(f"Audio file not found: {path_obj.resolve()}")

        audio = None
        sr = None

        if SOUNDFILE_AVAILABLE:
            try:
                audio, sr = sf.read(str(path_obj), dtype="float32")
            except Exception:
                audio = None
                sr = None

        if audio is None:
            if LIBROSA_AVAILABLE:
                audio, sr = librosa.load(str(path_obj), sr=None, mono=False)
            else:
                raise RuntimeError(
                    f"Failed to load audio from {path_obj}. Neither soundfile nor librosa available."
                )

        return standardize_waveform(audio, orig_sr=sr, target_sr=target_sr), target_sr

    elif isinstance(audio_input, np.ndarray):
        sr = orig_sr if orig_sr is not None else target_sr
        return standardize_waveform(audio_input, orig_sr=sr, target_sr=target_sr), target_sr

    else:
        raise TypeError(f"Unsupported audio input type: {type(audio_input)}")


def chunk_waveform(
    audio: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    chunk_duration: float = DEFAULT_CHUNK_DURATION,
    hop_duration: Optional[float] = None,
    pad_short: bool = True,
) -> List[np.ndarray]:
    """
    Splits a 1D audio waveform into fixed-length ~2.0-second chunks (32,000 samples).

    Args:
        audio: 1D float32 numpy array.
        sample_rate: Audio sample rate in Hz (default: 16000).
        chunk_duration: Target duration in seconds per chunk (default: 2.0s).
        hop_duration: Hop size between successive chunks in seconds.
                      If None, non-overlapping sequential chunks (hop_duration = chunk_duration) are used.
        pad_short: If True and the audio (or remaining tail) is shorter than chunk_size,
                   zero-pads the chunk up to chunk_size.

    Returns:
        List of 1D numpy arrays, each of length int(chunk_duration * sample_rate).
    """
    chunk_samples = int(chunk_duration * sample_rate)
    hop_samples = int(hop_duration * sample_rate) if hop_duration is not None else chunk_samples

    total_samples = len(audio)

    if total_samples == 0:
        if pad_short:
            return [np.zeros(chunk_samples, dtype=np.float32)]
        return []

    # Handle short audio less than 1 chunk
    if total_samples < chunk_samples:
        if pad_short:
            pad_width = chunk_samples - total_samples
            padded_chunk = np.pad(audio, (0, pad_width), mode="constant", constant_values=0.0)
            return [padded_chunk.astype(np.float32)]
        else:
            return []

    chunks = []
    start = 0
    while start + chunk_samples <= total_samples:
        chunk = audio[start : start + chunk_samples]
        chunks.append(np.ascontiguousarray(chunk, dtype=np.float32))
        start += hop_samples

    # Handle leftover tail if any and pad_short is True
    remaining_samples = total_samples - start
    if pad_short and remaining_samples > 0 and start < total_samples:
        tail = audio[start:]
        pad_width = chunk_samples - len(tail)
        padded_tail = np.pad(tail, (0, pad_width), mode="constant", constant_values=0.0)
        chunks.append(np.ascontiguousarray(padded_tail, dtype=np.float32))

    return chunks


def is_silence(audio_chunk: np.ndarray, threshold: float = SILENCE_RMS_THRESHOLD) -> bool:
    """
    Checks if an audio chunk is purely silent or negligible energy.
    """
    if len(audio_chunk) == 0:
        return True
    rms = np.sqrt(np.mean(np.square(audio_chunk)))
    return bool(rms < threshold)
