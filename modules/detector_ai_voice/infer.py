"""
VoxGuard Module 2: AI / Synthetic Voice Detector (AASIST-based)

NOTE ON MODEL PERFORMANCE & ACCURACY:

This module wraps an AASIST checkpoint fine-tuned on a 1200-sample balanced subset
of ASVspoof2019 (LA + PA, clean + G.711 mu-law), after fixing a label-convention bug
that had earlier training runs stuck near chance level (~50-53% accuracy).

Test set results (151 samples, 17 corrupted source files excluded):

    - Overall EER: 12.58%

    - LA (synthetic TTS / voice-conversion) EER: 2.11%
      <- this detector's core job, strong

    - PA (replay attacks) EER: 33.85%
      <- weak, BY DESIGN: replay detection is owned by the
         separate detector_replay module (handcrafted features, 0.35 fusion weight).
         AASIST's architecture targets synthesis artifacts, not recording-channel
         artifacts, so this gap reflects a task boundary between modules, not a bug
         in this one.

    - clean vs g711_8khz EER: 14.96% vs 15.45%
      (near-identical -> robust to telephone codec)
"""

# ============================================================
# WINDOWS / PYTORCH STABILITY SETTINGS
# These MUST be set before numpy/torch are imported.
# ============================================================

import os

# Limit CPU thread usage to avoid Windows native OpenMP/MKL
# memory-access conflicts during PyTorch inference.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import soundfile as sf
import torch

# Limit PyTorch CPU threads for more stable inference on Windows.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# ============================================================
# ADD REPOSITORY ROOT TO PYTHON PATH
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ============================================================
# IMPORT AASIST MODEL
# ============================================================

try:
    from models.AASIST import Model
except ImportError:
    raise ImportError(
        "Could not import Model from models.AASIST. "
        "Ensure models/AASIST.py exists."
    )


# ============================================================
# FIXED-LENGTH AUDIO INPUT
# ============================================================

# AASIST expects 64600 samples at 16 kHz
# 64600 / 16000 = approximately 4.0375 seconds

NUM_EVAL_SAMPLES = 64600


# ============================================================
# AI VOICE DETECTOR
# ============================================================

class AIVoiceDetector:
    """AASIST-based synthetic voice / AI audio detector wrapper."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):

        # ----------------------------------------------------
        # Resolve device
        # ----------------------------------------------------
        #
        # For Windows stability, CPU is used by default.
        #
        # You can still explicitly pass:
        # device="cuda"
        # if you later want to test CUDA.
        #
        if device is None:
            self.device = torch.device("cpu")

        elif isinstance(device, str):
            self.device = torch.device(device)

        else:
            self.device = device


        # ----------------------------------------------------
        # Determine checkpoint path
        # ----------------------------------------------------

        if checkpoint_path is None:

            default_ckpt = (
                REPO_ROOT
                / "modules"
                / "detector_ai_voice"
                / "checkpoints"
                / "best_model.pt"
            )

            fallback_ckpt = (
                REPO_ROOT
                / "models"
                / "best_AASIST_VoxGuard_zero_pad.pth"
            )

            if default_ckpt.exists():

                checkpoint_path = default_ckpt

            elif fallback_ckpt.exists():

                print(
                    f"[detector_ai_voice] WARNING: {default_ckpt} not found, "
                    f"falling back to {fallback_ckpt}. "
                    f"Verify this is the corrected checkpoint before a demo."
                )

                checkpoint_path = fallback_ckpt

            else:

                checkpoint_path = default_ckpt


        self.checkpoint_path = Path(checkpoint_path)


        if not self.checkpoint_path.exists():

            raise FileNotFoundError(
                f"AASIST checkpoint file not found at expected path: "
                f"{self.checkpoint_path}"
            )


        # ----------------------------------------------------
        # Determine config path
        # ----------------------------------------------------

        if config_path is None:

            default_config = (
                REPO_ROOT
                / "config"
                / "AASIST.conf"
            )

            alt_config = (
                REPO_ROOT
                / "modules"
                / "detector_ai_voice"
                / "config"
                / "AASIST.conf"
            )

            if default_config.exists():

                config_path = default_config

            elif alt_config.exists():

                config_path = alt_config

            else:

                config_path = default_config


        self.config_path = Path(config_path)


        if not self.config_path.exists():

            raise FileNotFoundError(
                f"AASIST config file not found at expected path: "
                f"{self.config_path}"
            )


        # ----------------------------------------------------
        # Label information
        # ----------------------------------------------------

        self.label_map: Dict[str, int] = {}

        self.spoof_index: int = -1


        # ----------------------------------------------------
        # Load AASIST model
        # ----------------------------------------------------

        self.model = self._load_model()


    # ========================================================
    # LOAD MODEL
    # ========================================================

    def _load_model(self) -> Model:
        """Loads and initializes the AASIST PyTorch model."""

        # ----------------------------------------------------
        # Load configuration
        # ----------------------------------------------------

        with open(self.config_path, "r", encoding="utf-8") as f:

            config = json.load(f)


        if "model_config" not in config:

            raise KeyError(
                f"Expected 'model_config' key in config file "
                f"{self.config_path}"
            )


        model_config = config["model_config"]


        # ----------------------------------------------------
        # Create model architecture
        # ----------------------------------------------------

        model = Model(model_config)

        print("[detector_ai_voice] AASIST model architecture created.")


        # ----------------------------------------------------
        # Load checkpoint
        # ----------------------------------------------------

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        print(
            f"[detector_ai_voice] Checkpoint loaded from "
            f"{self.checkpoint_path}"
        )


        # ----------------------------------------------------
        # CHECKPOINT FORMAT
        # ----------------------------------------------------

        # The corrected checkpoint contains:
        #
        # {
        #     "state_dict": ...,
        #     "label_map": ...
        # }
        #
        # The label_map is important because:
        #
        # bonafide = 1
        # spoof    = 0
        #
        # This is why we MUST NOT hardcode probs[0, 1].
        #

        if (
            isinstance(checkpoint, dict)
            and "state_dict" in checkpoint
            and "label_map" in checkpoint
        ):

            state_dict = checkpoint["state_dict"]

            self.label_map = checkpoint["label_map"]

        else:

            raise ValueError(
                f"Checkpoint at {self.checkpoint_path} is missing "
                f"'state_dict' and/or 'label_map'. "
                f"This wrapper requires the checkpoint format "
                f"produced by train_aasist_kaggle.py: "
                f"{{'state_dict': ..., 'label_map': {{...}}}}. "
                f"Loading a raw state_dict here risks silently using "
                f"the wrong spoof/bonafide index -- re-export the "
                f"checkpoint in the correct format instead of bypassing "
                f"this check."
            )


        # ----------------------------------------------------
        # Validate label map
        # ----------------------------------------------------

        if "spoof" not in self.label_map:

            raise ValueError(
                f"Checkpoint label_map {self.label_map} has no "
                f"'spoof' key. Expected something like "
                f"{{'bonafide': 1, 'spoof': 0}}."
            )


        self.spoof_index = self.label_map["spoof"]


        # ----------------------------------------------------
        # Load model weights
        # ----------------------------------------------------

        # strict=False allows SincConv dynamically-computed buffers to initialize naturally
        model.load_state_dict(
            state_dict,
            strict=False
        )

        print("[detector_ai_voice] Model weights loaded successfully.")


        # ----------------------------------------------------
        # Move model to device
        # ----------------------------------------------------

        model.to(self.device)

        model.eval()


        print(
            f"[detector_ai_voice] Loaded checkpoint from "
            f"{self.checkpoint_path}"
        )

        print(
            f"[detector_ai_voice] label_map={self.label_map} "
            f"-> using spoof_index={self.spoof_index}"
        )

        print(
            f"[detector_ai_voice] Device: {self.device}"
        )

        print(
            "[detector_ai_voice] Model ready for inference."
        )


        return model


    # ========================================================
    # PREPROCESS AUDIO
    # ========================================================

    def preprocess(self, audio_chunk: np.ndarray) -> torch.Tensor:
        """
        Preprocesses variable-length 1D numpy array into
        a 64600-sample PyTorch tensor.
        """

        # ----------------------------------------------------
        # Convert to numpy if necessary
        # ----------------------------------------------------

        if not isinstance(audio_chunk, np.ndarray):

            audio_chunk = np.array(
                audio_chunk,
                dtype=np.float32
            )


        # ----------------------------------------------------
        # Convert stereo/multichannel to mono
        # ----------------------------------------------------

        if audio_chunk.ndim > 1:

            audio_chunk = audio_chunk.mean(axis=-1)


        # ----------------------------------------------------
        # Ensure float32
        # ----------------------------------------------------

        audio_chunk = audio_chunk.astype(
            np.float32
        )


        # ----------------------------------------------------
        # Get length
        # ----------------------------------------------------

        length = len(audio_chunk)


        # ----------------------------------------------------
        # Empty audio
        # ----------------------------------------------------

        if length == 0:

            audio_chunk = np.zeros(
                NUM_EVAL_SAMPLES,
                dtype=np.float32
            )


        # ----------------------------------------------------
        # Short audio
        # ----------------------------------------------------

        elif length < NUM_EVAL_SAMPLES:

            # Repeat padding / tiling

            n_repeats = (
                NUM_EVAL_SAMPLES // length
            ) + 1

            audio_chunk = np.tile(
                audio_chunk,
                n_repeats
            )[:NUM_EVAL_SAMPLES]


        # ----------------------------------------------------
        # Long audio
        # ----------------------------------------------------

        elif length > NUM_EVAL_SAMPLES:

            # Center crop

            start = (
                length - NUM_EVAL_SAMPLES
            ) // 2

            audio_chunk = audio_chunk[
                start : start + NUM_EVAL_SAMPLES
            ]


        # ----------------------------------------------------
        # Convert to PyTorch tensor
        # ----------------------------------------------------

        wav_tensor = (
            torch.from_numpy(audio_chunk)
            .float()
            .unsqueeze(0)
            .to(self.device)
        )


        return wav_tensor


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        audio_chunk: np.ndarray
    ) -> Dict[str, Any]:

        """
        Contract:
            predict(audio_chunk)
            -> {"score": float, "top_feature": str}

        Args:
            audio_chunk:
                1D numpy float32 array,
                16kHz mono,
                variable length.

        Returns:
            Dict containing:

                "score":
                    float in [0.0, 1.0],
                    representing P(spoof).

                "top_feature":
                    str describing the feature or confidence driver.
        """

        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        print(
            "[detector_ai_voice] Starting preprocessing..."
        )

        wav_tensor = self.preprocess(
            audio_chunk
        )


        print(
            f"[detector_ai_voice] Tensor shape: "
            f"{wav_tensor.shape}"
        )

        print(
            f"[detector_ai_voice] Tensor dtype: "
            f"{wav_tensor.dtype}"
        )

        print(
            f"[detector_ai_voice] Tensor device: "
            f"{wav_tensor.device}"
        )


        # ----------------------------------------------------
        # AASIST inference
        # ----------------------------------------------------

        print(
            "[detector_ai_voice] Starting AASIST inference..."
        )


        with torch.no_grad():

            _, logits = self.model(
                wav_tensor
            )


        print(
            "[detector_ai_voice] AASIST inference completed."
        )


        # ----------------------------------------------------
        # Convert logits to probabilities
        # ----------------------------------------------------

        probs = torch.softmax(
            logits,
            dim=-1
        )


        # ----------------------------------------------------
        # Get spoof probability
        # ----------------------------------------------------

        # IMPORTANT:
        #
        # Do NOT hardcode:
        #
        # probs[0, 1]
        #
        # Instead use the spoof index stored in the checkpoint.

        spoof_prob = float(
            probs[
                0,
                self.spoof_index
            ].item()
        )


        # ----------------------------------------------------
        # Determine top feature
        # ----------------------------------------------------

        if spoof_prob >= 0.75:

            top_feature = (
                "high-confidence synthetic artifact "
                "(spectro-temporal graph attention)"
            )

        elif spoof_prob >= 0.50:

            top_feature = (
                "low-confidence synthetic artifact "
                "(spectro-temporal graph attention)"
            )

        elif spoof_prob <= 0.25:

            top_feature = (
                "high-confidence bonafide speech "
                "(spectro-temporal graph attention)"
            )

        else:

            top_feature = (
                "low-confidence bonafide speech "
                "(spectro-temporal graph attention)"
            )


        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return {
            "score": spoof_prob,
            "top_feature": top_feature
        }


# ============================================================
# LAZY SINGLETON
# ============================================================

_GLOBAL_DETECTOR: Optional[AIVoiceDetector] = None


# ============================================================
# MODULE-LEVEL PREDICT FUNCTION
# ============================================================

def predict(
    audio_chunk: np.ndarray
) -> Dict[str, Any]:

    """
    Module-level inference function implementing
    shared detector contract.

    predict(audio_chunk)
        -> {"score": float, "top_feature": str}
    """

    global _GLOBAL_DETECTOR


    if _GLOBAL_DETECTOR is None:

        _GLOBAL_DETECTOR = AIVoiceDetector()


    return _GLOBAL_DETECTOR.predict(
        audio_chunk
    )


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Check command-line argument
    # --------------------------------------------------------

    if len(sys.argv) < 2:

        print(
            "Usage: python "
            "modules/detector_ai_voice/infer.py "
            "<path_to_wav_file>"
        )

        sys.exit(1)


    # --------------------------------------------------------
    # Get WAV file
    # --------------------------------------------------------

    wav_file = sys.argv[1]


    if not os.path.exists(wav_file):

        raise FileNotFoundError(
            f"Input test audio file not found: "
            f"{wav_file}"
        )


    # --------------------------------------------------------
    # Read WAV
    # --------------------------------------------------------

    data, sr = sf.read(
        wav_file
    )


    print(
        f"Loaded '{wav_file}': "
        f"{len(data)} samples at {sr} Hz"
    )


    # --------------------------------------------------------
    # Sample-rate warning
    # --------------------------------------------------------

    if sr != 16000:

        print(
            f"WARNING: file is {sr}Hz, "
            f"expected 16kHz. "
            f"Results may be unreliable."
        )


    # --------------------------------------------------------
    # Create detector
    # --------------------------------------------------------

    detector = AIVoiceDetector()


    # --------------------------------------------------------
    # Run prediction
    # --------------------------------------------------------

    result = detector.predict(
        data
    )


    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print(
        "\n--- Inference Result ---"
    )

    print(
        json.dumps(
            result,
            indent=2
        )
    )