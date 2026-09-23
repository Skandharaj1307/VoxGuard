"""
modules/risk_engine/integration.py
==================================
VoxGuard Multi-Detector Pipeline Integration Layer.

Coordinates end-to-end inference across:
  - Module 2: AI / Synthetic Voice Detector (AASIST) -> P_ai
  - Module 3: Acoustic Replay Detector (13 acoustic features + LogisticRegression) -> P_replay
  - Module 4: Transmission Channel / Codec Detector (G.711 vs Clean) -> P_channel
  - Module 5: Risk Engine (IRS fusion + decision + explainable reasons)

Contract:
  run_full_detection(audio_filepath) -> Dict[str, Any]
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Union, Tuple
import numpy as np

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.risk_engine.engine import compute_risk


def load_audio_for_detectors(
    audio_filepath: Union[str, Path],
    target_sr: int = 16000
) -> Tuple[np.ndarray, int]:
    """
    Loads an audio file into a 1D float32 numpy array at target_sr (16 kHz) mono
    as expected by Module 2 and Module 3.

    Attempts loading via `soundfile` first; falls back to standard library `wave`
    module for standard uncompressed PCM WAV files.

    Args:
        audio_filepath: Path to the target audio file (.wav, .flac, etc.).
        target_sr: Desired sample rate in Hz (default: 16000).

    Returns:
        Tuple[np.ndarray, int]: (1D float32 audio waveform, sample_rate).
    """
    path = Path(audio_filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path.resolve()}")

    try:
        import soundfile as sf
        data, sr = sf.read(str(path))
        if data.ndim > 1:
            data = np.mean(data, axis=-1)
        data = data.astype(np.float32)

        if sr != target_sr:
            # Resample to target_sr using linear interpolation
            duration = len(data) / sr
            new_len = int(round(duration * target_sr))
            indices = np.linspace(0, len(data) - 1, new_len)
            data = np.interp(indices, np.arange(len(data)), data).astype(np.float32)
            sr = target_sr

        return data, sr

    except ImportError:
        # Fallback to standard library wave for .wav files
        if path.suffix.lower() == ".wav":
            import wave
            with wave.open(str(path), "rb") as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                raw_bytes = wf.readframes(n_frames)

            if sampwidth == 2:
                data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 1:
                data = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            elif sampwidth == 4:
                data = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

            if n_channels > 1:
                data = data.reshape(-1, n_channels).mean(axis=-1)

            if framerate != target_sr:
                duration = len(data) / framerate
                new_len = int(round(duration * target_sr))
                indices = np.linspace(0, len(data) - 1, new_len)
                data = np.interp(indices, np.arange(len(data)), data).astype(np.float32)
                framerate = target_sr

            return data, framerate
        else:
            raise ImportError(
                "Audio loading failed: 'soundfile' package is required to decode "
                f"'{path.suffix}' files. Please install soundfile."
            )


def run_full_detection(audio_filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Executes the full end-to-end VoxGuard detection pipeline:
      1. Validates the input audio filepath.
      2. Passes the original audio filepath directly to Module 4 (Channel Detector).
      3. Loads/normalizes the audio waveform (16 kHz mono float32) for Modules 2 and 3.
      4. Invokes Module 2 (AI Voice Detector) to obtain real P_ai.
      5. Invokes Module 3 (Replay Detector) to obtain real P_replay.
      6. Invokes Module 5 (Risk Engine) to compute the Integrated Risk Score (IRS).

    Strict partial failure policy:
      If ANY detector module fails to initialize, load its model, or produce
      a real numeric score, a clear RuntimeError is raised identifying precisely
      which module failed and why. No fallback, dummy, or partial scores are used.

    Args:
        audio_filepath: Path to the target audio file.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'audio_filepath': str path to input audio.
            - 'scores': {'p_ai': float, 'p_replay': float, 'p_channel': float}
            - 'detector_results': {
                'ai_voice': Dict[str, Any],
                'replay': Dict[str, Any],
                'channel': Dict[str, Any]
              }
            - 'risk_result': {'irs': float, 'decision': str, 'reasons': List[str]}
    """
    audio_path = Path(audio_filepath).resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Input audio file not found: {audio_path}")

    # -------------------------------------------------------------------------
    # 1. Module 4 — Transmission Channel / Codec Detector
    # (Passes original audio filepath directly to Module 4 per interface contract)
    # -------------------------------------------------------------------------
    try:
        from modules.detector_channel import predict as predict_channel
        channel_result = predict_channel(str(audio_path))
        if not isinstance(channel_result, dict) or "score" not in channel_result:
            raise ValueError(f"Module 4 returned unexpected result structure: {channel_result}")
        p_channel = float(channel_result["score"])
        if not (0.0 <= p_channel <= 1.0):
            raise ValueError(f"Module 4 produced out-of-range score: {p_channel}")
    except Exception as e:
        raise RuntimeError(f"Module 4 integration failed:\n{e}") from e

    # -------------------------------------------------------------------------
    # Prepare audio chunk for Modules 2 and 3 (16 kHz mono float32)
    # -------------------------------------------------------------------------
    audio_chunk, sr = load_audio_for_detectors(audio_path, target_sr=16000)

    # -------------------------------------------------------------------------
    # 2. Module 2 — AI / Synthetic Voice Detector (AASIST)
    # -------------------------------------------------------------------------
    try:
        from modules.detector_ai_voice import predict as predict_ai
        ai_result = predict_ai(audio_chunk)
        if not isinstance(ai_result, dict) or "score" not in ai_result:
            raise ValueError(f"Module 2 returned unexpected result structure: {ai_result}")
        p_ai = float(ai_result["score"])
        if not (0.0 <= p_ai <= 1.0):
            raise ValueError(f"Module 2 produced out-of-range score: {p_ai}")
    except Exception as e:
        raise RuntimeError(f"Module 2 integration failed:\n{e}") from e

    # -------------------------------------------------------------------------
    # 3. Module 3 — Acoustic Replay Detector
    # -------------------------------------------------------------------------
    try:
        from modules.detector_replay import predict as predict_replay
        replay_result = predict_replay(audio_chunk)
        if not isinstance(replay_result, dict) or "score" not in replay_result:
            raise ValueError(f"Module 3 returned unexpected result structure: {replay_result}")
        p_replay = float(replay_result["score"])
        if not (0.0 <= p_replay <= 1.0):
            raise ValueError(f"Module 3 produced out-of-range score: {p_replay}")
    except Exception as e:
        raise RuntimeError(f"Module 3 integration failed:\n{e}") from e

    # -------------------------------------------------------------------------
    # 4. Module 5 — Risk Engine Fusion
    # -------------------------------------------------------------------------
    risk_result = compute_risk(p_ai=p_ai, p_replay=p_replay, p_channel=p_channel)

    return {
        "audio_filepath": str(audio_path),
        "scores": {
            "p_ai": p_ai,
            "p_replay": p_replay,
            "p_channel": p_channel,
        },
        "detector_results": {
            "ai_voice": ai_result,
            "replay": replay_result,
            "channel": channel_result,
        },
        "risk_result": risk_result,
    }
