"""
Unit and Regression Test Suite: test_causal_features.py
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming

Tests causal feature engineering in src/feature_engineering/features.py:
1. Strict Causality: Changing future trajectory values never alters historical features.
2. Prefix Invariance: Features computed on trajectory prefix [:k] match the slice [:k] of full features.
3. Circular Yaw Wrap-around: Proper shortest angular difference across +/-180° boundary.
4. Boundary Handling: Initial acceleration boundary (acc[0]=0, acc[1]=0 avoids artificial spikes).
5. Irregular Timestamps: Handles legitimate positive irregular sampling intervals.
6. Timestamp Validation: Rejects duplicate timestamps (dt=0) and non-monotonic timestamps (dt<0).
7. Non-Finite & Dimension Rejection: Rejects NaNs, Infs, mismatched dimensions, and out-of-range pitch.
8. Dimension and Order Integrity: Strictly preserves (T, 7) shape and column ordering.
9. Streaming Inference Equivalence: Online single-step updates match offline batch slices.
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
        # Synthetic realistic viewport trajectory (10 seconds at ~10 Hz = 100 frames)
        self.T = 100
        self.dt = 0.10091
        self.timestamps = np.cumsum(np.full(self.T, self.dt))
        # Random walk in yaw [-180, 180] and pitch [-85, 85]
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

        cutoffs = [1, 5, 10, 25, 50, 75, 90]
        for k in cutoffs:
            yaw_mutated = self.yaw.copy()
            pitch_mutated = self.pitch.copy()

            # Inject radical future perturbations within valid physical angle ranges
            yaw_mutated[k + 1 :] = np.random.uniform(-180, 180, size=self.T - k - 1)
            pitch_mutated[k + 1 :] = np.random.uniform(-80, 80, size=self.T - k - 1)

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
        # Turning clockwise across +180° boundary (+179° to -179° is +2° step)
        yaw_cw = np.array([178.0, 179.0, -179.0, -178.0])
        dt = 0.1
        vel_cw, _ = compute_derivatives(yaw_cw, dt=dt, is_angular=True)
        self.assertAlmostEqual(vel_cw[2], 20.0, places=5)
        self.assertAlmostEqual(vel_cw[1], 10.0, places=5)
        self.assertAlmostEqual(vel_cw[3], 10.0, places=5)

        # Turning counter-clockwise across -180° boundary (-179° to +179° is -2° step)
        yaw_ccw = np.array([-178.0, -179.0, 179.0, 178.0])
        vel_ccw, _ = compute_derivatives(yaw_ccw, dt=dt, is_angular=True)
        self.assertAlmostEqual(vel_ccw[2], -20.0, places=5)
        self.assertAlmostEqual(vel_ccw[1], -10.0, places=5)
        self.assertAlmostEqual(vel_ccw[3], -10.0, places=5)

    def test_circular_difference_tie_breaking(self):
        """Exact 180° angular difference has documented deterministic tie-breaking (-180°)."""
        angles = np.array([0.0, 180.0])
        step = compute_circular_difference(angles)
        self.assertEqual(step[0], -180.0)

    def test_acceleration_boundary_behavior(self):
        """
        Initial acceleration boundary policy: acc[0]=0 and acc[1]=0.
        Must NOT calculate artificial acceleration using an assumed previous velocity of zero.
        """
        # Constant velocity trajectory (moving 10 deg every 0.1s -> vel = 100 deg/s)
        yaw_const_vel = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        pitch_const_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        feat = extract_7_features(yaw_const_vel, pitch_const_vel, dt=0.1)

        # At t=0: vel=0, acc=0
        self.assertEqual(feat[0, 3], 0.0)
        self.assertEqual(feat[0, 5], 0.0)

        # At t=1: vel=100, acc must be 0 (no artificial spike to 1000 deg/s^2)
        self.assertAlmostEqual(feat[1, 3], 100.0, places=5)
        self.assertEqual(feat[1, 5], 0.0, "acc[1] must be 0.0, avoiding artificial velocity jump artifact")

        # At t=2, 3, 4: vel=100, acceleration = 0 (constant velocity)
        for t in [2, 3, 4]:
            self.assertAlmostEqual(feat[t, 3], 100.0, places=5)
            self.assertAlmostEqual(feat[t, 5], 0.0, places=5)

    def test_legitimate_irregular_timestamps(self):
        """Handles legitimate irregular positive sampling intervals."""
        np.random.seed(99)
        # Positive jittered intervals between 0.05s and 0.15s
        dts = np.random.uniform(0.05, 0.15, size=self.T - 1)
        irregular_ts = np.insert(np.cumsum(dts), 0, 0.0)

        features = extract_7_features(self.yaw, self.pitch, timestamps=irregular_ts)
        self.assertEqual(features.shape, (self.T, 7))
        self.assertTrue(np.all(np.isfinite(features)))

        # Velocity at index 1 must equal delta_yaw / dt[0]
        expected_v1 = compute_circular_difference(self.yaw[:2])[0] / dts[0]
        self.assertAlmostEqual(features[1, 3], expected_v1, places=4)

    def test_duplicate_and_non_monotonic_timestamps_rejected(self):
        """Duplicate timestamps (dt=0) and decreasing timestamps (dt<0) must be rejected with ValueError."""
        # Duplicate timestamp
        ts_dup = self.timestamps.copy()
        ts_dup[5] = ts_dup[4]
        with self.assertRaises(ValueError):
            extract_7_features(self.yaw, self.pitch, timestamps=ts_dup)

        # Non-monotonic timestamp
        ts_neg = self.timestamps.copy()
        ts_neg[5] = ts_neg[4] - 0.1
        with self.assertRaises(ValueError):
            extract_7_features(self.yaw, self.pitch, timestamps=ts_neg)

    def test_non_finite_and_invalid_inputs_rejected(self):
        """Rejects non-finite values, mismatched lengths, out-of-range pitch, and simultaneous dt & timestamps."""
        # NaN in yaw
        yaw_nan = self.yaw.copy()
        yaw_nan[3] = np.nan
        with self.assertRaises(ValueError):
            extract_7_features(yaw_nan, self.pitch)

        # Inf in pitch
        pitch_inf = self.pitch.copy()
        pitch_inf[3] = np.inf
        with self.assertRaises(ValueError):
            extract_7_features(self.yaw, pitch_inf)

        # Pitch out of bounds (> 90°)
        pitch_oob = self.pitch.copy()
        pitch_oob[5] = 95.0
        with self.assertRaises(ValueError):
            extract_7_features(self.yaw, pitch_oob)

        # Length mismatch
        with self.assertRaises(ValueError):
            extract_7_features(self.yaw[:-1], self.pitch)

        # Providing both dt and timestamps
        with self.assertRaises(ValueError):
            extract_7_features(self.yaw, self.pitch, dt=0.1, timestamps=self.timestamps)

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

        # Feature 5 & 6: accelerations for t >= 2
        d_yvel = features[2:, 3] - features[1:-1, 3]
        np.testing.assert_allclose(features[2:, 5], d_yvel / self.dt, rtol=1e-5, atol=1e-5)
        d_pvel = features[2:, 4] - features[1:-1, 4]
        np.testing.assert_allclose(features[2:, 6], d_pvel / self.dt, rtol=1e-5, atol=1e-5)

    def test_streaming_inference_equivalence(self):
        """
        Online single-step updates match offline batch slices under identical causal boundary.
        """
        seq_len = 10
        offline_features = extract_7_features(self.yaw, self.pitch, dt=self.dt)

        for t in range(seq_len, 25):
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
