"""
Unit tests for model inference, streaming buffer consistency, configuration validation,
and input boundary protection:
- tests/test_inference.py
"""

import unittest
from pathlib import Path
import numpy as np
import torch

from src.models.final_model_inference import FinalViewportPredictor


class TestFinalModelInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictor = FinalViewportPredictor()

    def setUp(self):
        self.predictor.reset_streaming()

    def test_model_loading_and_config_parameters(self):
        """Validates that predictor loads successfully and configuration parameters match expected constants."""
        self.assertEqual(self.predictor.history_len, 10)
        self.assertEqual(self.predictor.horizon, 5)
        self.assertEqual(len(self.predictor.norm_mean), 7)
        self.assertEqual(len(self.predictor.norm_std), 7)
        self.assertTrue(np.all(self.predictor.norm_std > 0.0))
        self.assertTrue(np.isfinite(self.predictor.pitch_mean))
        self.assertGreater(self.predictor.pitch_std, 0.0)

    def test_batch_prediction_output_validation(self):
        """Batch predict must output valid finite yaw in [-180, 180] and pitch in [-90, 90]."""
        synthetic_yaw = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]
        synthetic_pitch = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        pred_yaw, pred_pitch = self.predictor.predict(synthetic_yaw, synthetic_pitch)

        self.assertIsInstance(pred_yaw, float)
        self.assertIsInstance(pred_pitch, float)
        self.assertTrue(np.isfinite(pred_yaw))
        self.assertTrue(np.isfinite(pred_pitch))
        self.assertGreaterEqual(pred_yaw, -180.0)
        self.assertLessEqual(pred_yaw, 180.0)
        self.assertGreaterEqual(pred_pitch, -90.0)
        self.assertLessEqual(pred_pitch, 90.0)

    def test_insufficient_history_length_rejected(self):
        """Sequences with length < 10 must raise ValueError."""
        short_yaw = [0.0, 1.0, 2.0, 3.0]
        short_pitch = [0.0, 1.0, 2.0, 3.0]
        with self.assertRaises(ValueError):
            self.predictor.predict(short_yaw, short_pitch)

    def test_streaming_mode_consistency_enforced(self):
        """Streaming mode must strictly enforce either timestamped mode or fixed-interval mode without mixing."""
        # Start in timestamped mode
        res1 = self.predictor.push_and_predict_streaming(0.0, 0.0, timestamp=0.0)
        self.assertIsNone(res1)
        self.assertEqual(self.predictor.streaming_mode, "timestamped")

        # Pushing without timestamp must fail in timestamped mode
        with self.assertRaises(ValueError):
            self.predictor.push_and_predict_streaming(1.0, 0.0, timestamp=None)

        # Reset and test fixed-interval mode
        self.predictor.reset_streaming()
        res2 = self.predictor.push_and_predict_streaming(0.0, 0.0, timestamp=None)
        self.assertIsNone(res2)
        self.assertEqual(self.predictor.streaming_mode, "fixed_interval")

        # Pushing with timestamp must fail in fixed-interval mode
        with self.assertRaises(ValueError):
            self.predictor.push_and_predict_streaming(1.0, 0.0, timestamp=0.1)

    def test_streaming_prediction_generation(self):
        """Streaming predictor generates predictions once 10 frames are accumulated."""
        for i in range(9):
            res = self.predictor.push_and_predict_streaming(float(i), 0.0)
            self.assertIsNone(res)

        # 10th frame triggers prediction
        res_10 = self.predictor.push_and_predict_streaming(9.0, 0.0)
        self.assertIsNotNone(res_10)
        pred_yaw, pred_pitch = res_10
        self.assertTrue(np.isfinite(pred_yaw))
        self.assertTrue(np.isfinite(pred_pitch))


if __name__ == "__main__":
    unittest.main()

