"""
test_replay_detector.py

Comprehensive Test Suite for VoxGuard Replay Attack Detector Module.

Verifies:
 1. Clean module import: `from modules.detector_replay import predict, predict_replay, ReplayDetector`
 2. Model artifact loading (replay_model.pkl, scaler.pkl, feature_config.json)
 3. Feature extraction correctness and shape
 4. Feature-column order consistency
 5. Prediction output type and dictionary structure
 6. P_replay numerical range: 0.0 <= P_replay <= 1.0
 7. Stereo audio handling (auto-downmixed to mono)
 8. Variable sample rates (8kHz, 44.1kHz resampled to 16kHz)
 9. Short audio handling (<2.0s padded)
10. Invalid audio & silence handling
11. Speaker-disjoint split zero-overlap verification
12. Full-file inference and multi-chunk aggregation
"""

import os
import sys
import unittest
from pathlib import Path
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Test 1: Clean module imports
from modules.detector_replay import (
    predict,
    predict_replay,
    predict_audio_file,
    ReplayDetector,
    get_replay_detector
)
from modules.detector_replay.features import (
    extract_chunk_features,
    extract_chunk_features_dict,
    REPLAY_FEATURE_NAMES,
    compute_reverb_proxy
)
from modules.detector_replay.preprocessing import (
    load_and_preprocess,
    chunk_waveform,
    is_silence
)
from dataset.split_replay_data import get_replay_splits


class TestReplayDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = get_replay_detector()
        cls.sr = 16000
        cls.chunk_samples = 32000  # 2.0s @ 16kHz
        # Synthetic speech-like test chunk
        t = np.linspace(0, 2.0, cls.chunk_samples, endpoint=False)
        cls.sample_chunk = (
            0.4 * np.sin(2 * np.pi * 250 * t) +
            0.2 * np.sin(2 * np.pi * 750 * t) +
            0.1 * np.random.normal(0, 0.05, cls.chunk_samples)
        ).astype(np.float32)

    def test_01_model_and_scaler_loading(self):
        """Test that model, scaler, and config load properly."""
        self.assertIsNotNone(self.detector.model)
        self.assertIsNotNone(self.detector.scaler)
        self.assertIsNotNone(self.detector.config)
        self.assertEqual(len(self.detector.feature_names), 13)

    def test_02_feature_extraction_shape_and_values(self):
        """Test feature extraction output shape, length, and numerical validity."""
        feats = extract_chunk_features(self.sample_chunk, sample_rate=self.sr)
        self.assertEqual(feats.shape, (13,))
        self.assertEqual(feats.dtype, np.float32)
        self.assertFalse(np.isnan(feats).any(), "NaN in features")
        self.assertFalse(np.isinf(feats).any(), "Inf in features")

    def test_03_feature_column_order_consistency(self):
        """Test that dictionary and array feature ordering match exactly."""
        feat_dict = extract_chunk_features_dict(self.sample_chunk, sample_rate=self.sr)
        feat_arr = extract_chunk_features(self.sample_chunk, sample_rate=self.sr)
        for i, name in enumerate(REPLAY_FEATURE_NAMES):
            self.assertAlmostEqual(feat_dict[name], float(feat_arr[i]), places=5)

    def test_04_prediction_output_type_and_schema(self):
        """Test standard predict() shared contract schema."""
        result = predict(self.sample_chunk)
        self.assertIsInstance(result, dict)
        self.assertIn("score", result)
        self.assertIn("top_feature", result)
        self.assertIsInstance(result["score"], (float, np.floating))
        self.assertIsInstance(result["top_feature"], str)

    def test_05_predict_replay_range(self):
        """Test that scalar predict_replay returns float in [0.0, 1.0]."""
        p_replay = predict_replay(self.sample_chunk, sample_rate=self.sr)
        self.assertIsInstance(p_replay, (float, np.floating))
        self.assertGreaterEqual(p_replay, 0.0)
        self.assertLessEqual(p_replay, 1.0)

    def test_06_stereo_audio_handling(self):
        """Test multi-channel (stereo) input is averaged to mono without error."""
        stereo_chunk = np.stack([self.sample_chunk, self.sample_chunk * 0.8], axis=1)
        self.assertEqual(stereo_chunk.shape, (32000, 2))
        p_stereo = predict_replay(stereo_chunk, sample_rate=self.sr)
        self.assertGreaterEqual(p_stereo, 0.0)
        self.assertLessEqual(p_stereo, 1.0)

    def test_07_different_sample_rates(self):
        """Test non-16kHz audio inputs (8kHz and 44.1kHz)."""
        # 8kHz audio
        t_8k = np.linspace(0, 2.0, 16000, endpoint=False)
        audio_8k = (0.5 * np.sin(2 * np.pi * 300 * t_8k)).astype(np.float32)
        p_8k = predict_replay(audio_8k, sample_rate=8000)
        self.assertGreaterEqual(p_8k, 0.0)
        self.assertLessEqual(p_8k, 1.0)

        # 44.1kHz audio
        t_44k = np.linspace(0, 2.0, int(44100 * 2.0), endpoint=False)
        audio_44k = (0.5 * np.sin(2 * np.pi * 300 * t_44k)).astype(np.float32)
        p_44k = predict_replay(audio_44k, sample_rate=44100)
        self.assertGreaterEqual(p_44k, 0.0)
        self.assertLessEqual(p_44k, 1.0)

    def test_08_short_audio_handling(self):
        """Test audio shorter than 2.0 seconds (e.g. 0.4s)."""
        short_audio = (0.3 * np.sin(2 * np.pi * 400 * np.linspace(0, 0.4, 6400))).astype(np.float32)
        p_short = predict_replay(short_audio, sample_rate=self.sr)
        self.assertGreaterEqual(p_short, 0.0)
        self.assertLessEqual(p_short, 1.0)

    def test_09_silence_handling(self):
        """Test silent audio array returns baseline safe score."""
        silent = np.zeros(32000, dtype=np.float32)
        res = predict(silent)
        self.assertLessEqual(res["score"], 0.15)
        self.assertTrue("silence" in res["top_feature"].lower() or "silent" in res["top_feature"].lower())

    def test_10_speaker_disjoint_split(self):
        """Test that data splits have strictly 0 speaker overlap."""
        train_df, val_df, test_df = get_replay_splits()
        tr_spk = set(train_df["speaker_id"].unique())
        val_spk = set(val_df["speaker_id"].unique())
        te_spk = set(test_df["speaker_id"].unique())

        self.assertEqual(len(tr_spk & val_spk), 0, "Train & Val speaker overlap detected")
        self.assertEqual(len(tr_spk & te_spk), 0, "Train & Test speaker overlap detected")
        self.assertEqual(len(val_spk & te_spk), 0, "Val & Test speaker overlap detected")

    def test_11_full_file_aggregation(self):
        """Test multi-chunk full file inference with different aggregations."""
        long_audio = np.tile(self.sample_chunk, 3)  # 6.0s = 3 chunks
        res_mean = self.detector.predict_file(long_audio, sample_rate=self.sr, aggregation="mean")
        res_max = self.detector.predict_file(long_audio, sample_rate=self.sr, aggregation="max")
        res_med = self.detector.predict_file(long_audio, sample_rate=self.sr, aggregation="median")

        self.assertEqual(res_mean["chunk_count"], 3)
        self.assertGreaterEqual(res_mean["score"], 0.0)
        self.assertLessEqual(res_mean["score"], 1.0)
        self.assertGreaterEqual(res_max["score"], res_mean["score"] - 1e-4)

    def test_12_reverb_proxy_naming_compliance(self):
        """Test that no RT60 or reverberation time variable names exist in feature names."""
        for feat in REPLAY_FEATURE_NAMES:
            self.assertNotIn("rt60", feat.lower())
            self.assertNotIn("reverberation_time", feat.lower())
        self.assertIn("reverb_proxy_decay_ratio", REPLAY_FEATURE_NAMES)


if __name__ == "__main__":
    print("=" * 75)
    print(" Running VoxGuard Replay Attack Detector Unit & Integration Tests")
    print("=" * 75)
    unittest.main(verbosity=2)
