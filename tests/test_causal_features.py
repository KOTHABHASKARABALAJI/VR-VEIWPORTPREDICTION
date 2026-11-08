"""
Unit and Regression Test Suite: test_causal_features.py
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming

Tests causal feature engineering in src/feature_engineering/features.py:
1. Strict Causality: Changing future trajectory values never alters historical features.
2. Prefix Invariance: Features computed on trajectory prefix [:k] match the slice [:k] of full features.
3. Circular Yaw Wrap-around: Proper shortest angular difference across +/-180° boundary.
4. Boundary Handling: Proper behavior on empty trajectories, single-frame trajectories, and small inputs.
5. Irregular Timestamps: Resilient to jitter, zero-interval duplicates, and non-positive dt.
6. Dimension and Order Integrity: Strictly preserves (T, 7) shape and column ordering.
7. Streaming Inference Equivalence: Online single-step updates match offline batch slices.
"""

import unittest
import numpy as np
from src.feature_engineering.features import (
    extract_7_features,
    compute_derivatives,
    compute_circular_difference,
    compute_sin_cos_yaw,
    unwrap_yaw_degrees,
    standardize_features,
)


class TestCausalFeatures(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # Create a synthetic realistic viewport trajectory (10 seconds at ~10 Hz = 100 frames)
        self.T = 100
        self.dt = 0.10091
        self.timestamps = np.cumsum(np.full(self.T, self.dt))
        # Smooth random walk in yaw [-180, 180] and pitch [-90, 90]
        yaw_steps = np.random.normal(0, 3.0, size=self.T)
        pitch_steps = np.random.normal(0, 1.5, size=self.T)
        self.yaw = ((np.cumsum(yaw_steps) + 180) % 360) - 180
        self.pitch = np.clip(np.cumsum(pitch_steps), -85.0, 85.0)

    def test_strict_causality_under_future_mutation(self):
        """
        Proof of Causality: Corrupting future frames at indices > k must NOT
        change any calculated feature at indices <= k.
        """
        clean_features = extract_7_features(self.yaw, self.pitch, dt=self.dt)

        # Test at various historical cutoffs k
        cutoffs = [1, 5, 10, 25, 50, 75, 90]
        for k in cutoffs:
            yaw_mutated = self.yaw.copy()
            pitch_mutated = self.pitch.copy()

            # Inject radical future perturbations (spikes, reversals, extreme values)
            yaw_mutated[k + 1 :] = np.random.uniform(-180, 180, size=self.T - k - 1) * 5.0
            pitch_mutated[k + 1 :] = 999.0

            mutated_features = extract_7_features(yaw_mutated, pitch_mutated, dt=self.dt)

            # Historical features up to index k must remain bitwise identical
            np.testing.assert_allclose(
                clean_features[: k + 1],
                mutated_features[: k + 1],
                atol=1e-12,
                rtol=1e-12,
                err_msg=f"Future mutation at >{k} leaked into historical features <= {k}!",
            )

    def test_prefix_invariance(self):
        """
        Evaluating a trajectory prefix of length k must yield identical features
        to the first k rows of the full trajectory evaluation.
        """
        full_features = extract_7_features(self.yaw, self.pitch, dt=self.dt)

        for k in [2, 10, 20, 50]:
            prefix_features = extract_7_features(self.yaw[:k], self.pitch[:k], dt=self.dt)
            np.testing.assert_allclose(
                prefix_features,
                full_features[:k],
                atol=1e-12,
                rtol=1e-12,
                err_msg=f"Prefix of length {k} diverged from full trajectory evaluation!",
            )

    def test_circular_yaw_wraparound(self):
        """
        Yaw velocity must reflect shortest circular path without spurious 360° spikes.
        """
        # Case A: Turning clockwise across +180° boundary (+179° to -179° is +2° step)
        yaw_cw = np.array([178.0, 179.0, -179.0, -178.0])
        dt = 0.1
        vel_cw, _ = compute_derivatives(yaw_cw, dt=dt, is_angular=True)
        # Expected step from 179 to -179: +2 deg in 0.1s -> +20 deg/s
        self.assertAlmostEqual(vel_cw[2], 20.0, places=5)
        self.assertAlmostEqual(vel_cw[1], 10.0, places=5)
        self.assertAlmostEqual(vel_cw[3], 10.0, places=5)

        # Case B: Turning counter-clockwise across -180° boundary (-179° to +179° is -2° step)
        yaw_ccw = np.array([-178.0, -179.0, 179.0, 178.0])
        vel_ccw, _ = compute_derivatives(yaw_ccw, dt=dt, is_angular=True)
        # Expected step from -179 to 179: -2 deg in 0.1s -> -20 deg/s
        self.assertAlmostEqual(vel_ccw[2], -20.0, places=5)
        self.assertAlmostEqual(vel_ccw[1], -10.0, places=5)
        self.assertAlmostEqual(vel_ccw[3], -10.0, places=5)

    def test_boundary_conditions_and_shapes(self):
        """
        Handles empty arrays, single-frame trajectories, and preserves (T, 7) shape.
        """
        # Empty input
        empty_feat = extract_7_features(np.array([]), np.array([]))
        self.assertEqual(empty_feat.shape, (0, 7))
        self.assertEqual(empty_feat.dtype, np.float32)

        # Single-frame input
        single_feat = extract_7_features(np.array([45.0]), np.array([10.0]))
        self.assertEqual(single_feat.shape, (1, 7))
        self.assertEqual(single_feat.dtype, np.float32)
        # Velocities and accelerations must be zero for t=0
        np.testing.assert_allclose(single_feat[0, 3:], [0.0, 0.0, 0.0, 0.0])

        # Standard trajectory
        feat = extract_7_features(self.yaw, self.pitch)
        self.assertEqual(feat.shape, (self.T, 7))
        self.assertEqual(feat.dtype, np.float32)
        # Initial boundary t=0 must be zero
        np.testing.assert_allclose(feat[0, 3:], [0.0, 0.0, 0.0, 0.0])

    def test_irregular_timestamps_and_zero_interval(self):
        """
        Safe handling of timestamps with zero or negative intervals (duplicates).
        """
        # Inject duplicate timestamps (dt = 0) and small negative jitter
        ts_jitter = self.timestamps.copy()
        ts_jitter[10] = ts_jitter[9]  # dt = 0
        ts_jitter[20] = ts_jitter[19] - 0.05  # negative dt

        features = extract_7_features(self.yaw, self.pitch, timestamps=ts_jitter)
        self.assertEqual(features.shape, (self.T, 7))
        self.assertFalse(np.isnan(features).any(), "NaN detected in feature matrix!")
        self.assertFalse(np.isinf(features).any(), "Inf detected in feature matrix!")

    def test_feature_order_and_encoding_integrity(self):
        """
        Verifies exact column order: [sin_yaw, cos_yaw, pitch, yaw_vel, pitch_vel, yaw_acc, pitch_acc].
        """
        features = extract_7_features(self.yaw, self.pitch, dt=self.dt)

        # Feature 0 & 1: sin and cos of yaw
        expected_sin = np.sin(np.radians(self.yaw))
        expected_cos = np.cos(np.radians(self.yaw))
        np.testing.assert_allclose(features[:, 0], expected_sin, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(features[:, 1], expected_cos, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(features[:, 0] ** 2 + features[:, 1] ** 2, 1.0, rtol=1e-5, atol=1e-5)

        # Feature 2: pitch in degrees
        np.testing.assert_allclose(features[:, 2], self.pitch, rtol=1e-5, atol=1e-5)

        # Feature 3: yaw velocity
        d_yaw = compute_circular_difference(self.yaw)
        np.testing.assert_allclose(features[1:, 3], d_yaw / self.dt, rtol=1e-5, atol=1e-5)

        # Feature 4: pitch velocity
        d_pitch = self.pitch[1:] - self.pitch[:-1]
        np.testing.assert_allclose(features[1:, 4], d_pitch / self.dt, rtol=1e-5, atol=1e-5)

        # Feature 5 & 6: accelerations
        d_yvel = features[1:, 3] - features[:-1, 3]
        np.testing.assert_allclose(features[1:, 5], d_yvel / self.dt, rtol=1e-5, atol=1e-5)
        d_pvel = features[1:, 4] - features[:-1, 4]
        np.testing.assert_allclose(features[1:, 6], d_pvel / self.dt, rtol=1e-5, atol=1e-5)

    def test_streaming_inference_equivalence(self):
        """
        Simulates an online streaming client where each new incoming viewport frame
        is processed causally. The streaming window at time t must match the offline batch.
        """
        seq_len = 10
        offline_features = extract_7_features(self.yaw, self.pitch, dt=self.dt)

        # Simulate streaming from t=10 to t=25
        for t in range(seq_len, 25):
            # Online client only has access to observations up to time t
            online_window = extract_7_features(self.yaw[: t + 1], self.pitch[: t + 1], dt=self.dt)[-seq_len:]
            offline_window = offline_features[t - seq_len + 1 : t + 1]

            np.testing.assert_allclose(
                online_window,
                offline_window,
                atol=1e-12,
                rtol=1e-12,
                err_msg=f"Online streaming window at t={t} differs from offline batch window!",
            )


if __name__ == "__main__":
    unittest.main()

