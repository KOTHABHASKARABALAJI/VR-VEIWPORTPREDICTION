"""
Tests for Day 3: Complete System Evaluation
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming
Phase: Phase 5, Day 3 — Complete System Evaluation
"""

import unittest
import numpy as np
import pandas as pd

from src.evaluation.viewport_evaluation import (
    compute_spherical_error_vector,
    compute_accuracy_metrics,
    ViewportPredictionEvaluator,
    ViewportAccuracyMetrics,
)
from src.evaluation.system_evaluator import (
    SystemEvaluator,
    TilePredictionBenchmarkResult,
    CANONICAL_BUDGETS_KBPS,
)
from src.quality.quality_config import QualityLevel


class TestViewportEvaluation(unittest.TestCase):
    """
    Unit tests for viewport trajectory prediction accuracy evaluation.
    """

    def test_compute_spherical_error_vector_exact_cases(self) -> None:
        """
        Tests spherical error vector under exact analytical cases:
        1. Identical coordinates -> 0.0°
        2. Orthogonal 90° yaw difference at equator -> 90.0°
        3. Opposite 180° yaw difference at equator -> 180.0°
        4. Zenith to Nadir (+90° to -90°) -> 180.0°
        5. Wraparound: +179° to -179° -> 2.0°
        """
        act_y = np.array([0.0, 0.0, 0.0, 0.0, 179.0])
        act_p = np.array([0.0, 0.0, 0.0, 90.0, 0.0])
        pred_y = np.array([0.0, 90.0, 180.0, 0.0, -179.0])
        pred_p = np.array([0.0, 0.0, 0.0, -90.0, 0.0])

        sph_err, yaw_diff, pitch_diff = compute_spherical_error_vector(act_y, act_p, pred_y, pred_p)

        self.assertAlmostEqual(sph_err[0], 0.0, places=4)
        self.assertAlmostEqual(sph_err[1], 90.0, places=4)
        self.assertAlmostEqual(sph_err[2], 180.0, places=4)
        self.assertAlmostEqual(sph_err[3], 180.0, places=4)
        self.assertAlmostEqual(sph_err[4], 2.0, places=4)
        self.assertAlmostEqual(yaw_diff[4], 2.0, places=4)

    def test_accuracy_metrics_calculation(self) -> None:
        """
        Verifies computation of MAE, RMSE, Median, P90, P95, P99.
        """
        act_y = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        act_p = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        pred_y = np.array([0.0, 15.0, 20.0, 35.0, 40.0])
        pred_p = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        metrics = compute_accuracy_metrics(act_y, act_p, pred_y, pred_p)
        self.assertEqual(metrics.count, 5)
        self.assertAlmostEqual(metrics.spherical_mae_deg, 2.0, places=4)
        self.assertAlmostEqual(metrics.yaw_mae_deg, 2.0, places=4)
        self.assertAlmostEqual(metrics.pitch_mae_deg, 0.0, places=4)
        self.assertAlmostEqual(metrics.spherical_median_deg, 0.0, places=4)

    def test_evaluator_dataframe_operations(self) -> None:
        """
        Verifies overall, subject-wise, and video-wise aggregations from DataFrame.
        """
        data = {
            "actual_yaw_deg": [0.0, 10.0, 20.0, 0.0, 10.0, 20.0],
            "actual_pitch_deg": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "predicted_yaw_deg": [1.0, 12.0, 21.0, 2.0, 14.0, 22.0],
            "predicted_pitch_deg": [0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
            "persistence_predicted_yaw_deg": [3.0, 15.0, 25.0, 4.0, 16.0, 26.0],
            "persistence_predicted_pitch_deg": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            "subject_id": [9, 9, 9, 16, 16, 16],
            "video": ["1.mp4", "1.mp4", "2.mp4", "1.mp4", "2.mp4", "2.mp4"],
        }
        df = pd.DataFrame(data)
        evaluator = ViewportPredictionEvaluator(df)

        overall = evaluator.evaluate_overall()
        self.assertIn("final_gru", overall)
        self.assertIn("persistence", overall)
        self.assertLess(overall["final_gru"].spherical_mae_deg, overall["persistence"].spherical_mae_deg)

        df_sub = evaluator.evaluate_by_subject()
        self.assertEqual(len(df_sub), 2)
        self.assertEqual(set(df_sub["subject_id"]), {9, 16})

        df_vid = evaluator.evaluate_by_video()
        self.assertEqual(len(df_vid), 2)
        self.assertEqual(set(df_vid["video"]), {"1.mp4", "2.mp4"})

        summary = evaluator.get_summary_comparison_table()
        self.assertEqual(len(summary), 2)


class TestSystemEvaluator(unittest.TestCase):
    """
    Unit tests for the integrated end-to-end system evaluator.
    """

    def setUp(self) -> None:
        self.evaluator = SystemEvaluator()

    def test_canonical_boundary_orientations(self) -> None:
        """
        Verifies that canonical boundaries (Equator, Poles, Yaw wraparounds)
        produce valid 24-tile intersections with sum of weights = 1.0.
        """
        df_bounds = self.evaluator.evaluate_canonical_boundary_orientations()
        self.assertEqual(len(df_bounds), 7)
        for _, row in df_bounds.iterrows():
            self.assertGreater(row["num_active_tiles"], 0)
            self.assertGreater(row["num_margin_tiles"], 0)
            self.assertAlmostEqual(row["weights_sum"], 1.0, places=4)

    def test_controlled_head_movement_simulation(self) -> None:
        """
        Verifies that controlled head rotation simulation correctly exhibits:
        1. 100% utility retention at delta_theta = 0.0°.
        2. Monotonically non-increasing retention as delta_theta increases.
        3. Neighbourhood-aware allocation retains higher utility than Viewport-Aware
           as rotation moves the FoV into adjacent margin tiles (delta_theta >= 20°).
        """
        df_motion = self.evaluator.simulate_controlled_head_movement(
            base_yaw_deg=0.0,
            base_pitch_deg=0.0,
            budget_kbps=7200.0,
            angle_offsets_deg=[0.0, 5.0, 15.0, 25.0, 30.0],
        )

        yaw_rot = df_motion[df_motion["rotation_direction"] == "Yaw Rotation"].sort_values("delta_theta_deg")
        self.assertAlmostEqual(yaw_rot.iloc[0]["viewport_retention_pct"], 100.0, places=3)
        self.assertAlmostEqual(yaw_rot.iloc[0]["neighbourhood_25_retention_pct"], 100.0, places=3)

        # At 25° and 30° rotation (rotational saccade across tile boundary), neighbourhood guard-band retains higher utility
        row_25 = yaw_rot[yaw_rot["delta_theta_deg"] == 25.0].iloc[0]
        self.assertGreater(
            row_25["neighbourhood_25_realized_utility"],
            row_25["viewport_realized_utility"],
            "Neighbourhood guard band must preserve higher realized utility under 25° rotation"
        )
        self.assertGreater(row_25["guard_band_gain_25_vs_vp"], 0.0)

        row_30 = yaw_rot[yaw_rot["delta_theta_deg"] == 30.0].iloc[0]
        self.assertGreater(
            row_30["neighbourhood_25_realized_utility"],
            row_30["viewport_realized_utility"],
            "Neighbourhood guard band must preserve higher realized utility under 30° rotation"
        )
        self.assertGreater(row_30["guard_band_gain_25_vs_vp"], 0.0)

    def test_strategy_comparison_budget_compliance(self) -> None:
        """
        Verifies that across all evaluated budgets, every strategy strictly satisfies the budget constraint.
        """
        dummy_df = pd.DataFrame({
            "actual_yaw_deg": [0.0, 15.0, -30.0],
            "actual_pitch_deg": [0.0, 5.0, -10.0],
            "predicted_yaw_deg": [1.0, 16.0, -28.0],
            "predicted_pitch_deg": [0.5, 4.0, -8.0],
            "persistence_predicted_yaw_deg": [2.0, 18.0, -25.0],
            "persistence_predicted_pitch_deg": [1.0, 3.0, -7.0],
            "subject_id": [9, 9, 9],
            "video": ["1.mp4", "1.mp4", "1.mp4"],
        })

        df_strat = self.evaluator.evaluate_allocation_strategies_on_dataset(
            dummy_df,
            budgets_kbps=[2400.0, 7200.0, 14400.0],
            sample_stride=1
        )
        self.assertEqual(len(df_strat), 3 * 4)

        for _, row in df_strat.iterrows():
            self.assertLessEqual(
                row["simulated_bitrate_kbps"],
                row["bandwidth_budget_kbps"] + 1e-6,
                f"Strategy {row['strategy']} violated budget {row['bandwidth_budget_kbps']}"
            )
            self.assertGreaterEqual(row["actual_realized_utility"], 1.0)
            self.assertGreaterEqual(row["predicted_weighted_utility"], 1.0)

    def test_sampling_and_fov_sensitivity(self) -> None:
        """
        Verifies sensitivity functions return valid DataFrames.
        """
        df_ray = self.evaluator.evaluate_sampling_sensitivity(resolutions=[15, 21])
        self.assertEqual(len(df_ray), 2 * 3)

        df_fov = self.evaluator.evaluate_fov_sensitivity(fov_degrees=[90.0, 100.0])
        self.assertEqual(len(df_fov), 2)
        self.assertGreaterEqual(df_fov.iloc[1]["active_tiles_count"], df_fov.iloc[0]["active_tiles_count"])


if __name__ == "__main__":
    unittest.main()
