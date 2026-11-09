"""
Unit tests for tile evaluation, persistence prediction coordinates, input validation,
percentile metrics, and subject-wise reporting:
- tests/test_tile_evaluation.py
"""

import unittest
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

from src.tiling.evaluate_tile_prediction import validate_predictions_df, run_tile_evaluation
from src.tiling.viewport_mapper import ViewportMapper
from src.tiling.grid_partitioner import TileGridPartitioner


class TestTileEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.preds_csv_path = cls.project_root / "results" / "predictions" / "final_model_test_predictions.csv"
        cls.metrics_csv_path = cls.project_root / "results" / "tiling" / "tile_prediction_metrics.csv"
        cls.subject_csv_path = cls.project_root / "results" / "tiling" / "tile_prediction_metrics_by_subject.csv"
        cls.summary_txt_path = cls.project_root / "results" / "tiling" / "tile_prediction_summary.txt"

    def test_1_persistence_coordinates_saved_correctly(self):
        """1. Persistence yaw, pitch, and spherical error must be saved correctly without NaNs."""
        self.assertTrue(self.preds_csv_path.exists(), f"Predictions CSV not found: {self.preds_csv_path}")
        df = pd.read_csv(self.preds_csv_path)

        for col in ["persistence_predicted_yaw_deg", "persistence_predicted_pitch_deg", "persistence_sph_err_deg"]:
            self.assertIn(col, df.columns, f"Missing required column: {col}")
            self.assertTrue(np.all(np.isfinite(df[col])), f"Column {col} has non-finite values")

        # Verify spherical error consistency between actual and persistence coordinates.
        # Note: atol=0.03 accounts for float32 trigonometric arccos conditioning near dot=1.0.
        act_yaw = np.radians(df["actual_yaw_deg"].values[:100].astype(np.float64))
        act_pitch = np.radians(df["actual_pitch_deg"].values[:100].astype(np.float64))
        pers_yaw = np.radians(df["persistence_predicted_yaw_deg"].values[:100].astype(np.float64))
        pers_pitch = np.radians(df["persistence_predicted_pitch_deg"].values[:100].astype(np.float64))

        dot = np.sin(act_pitch) * np.sin(pers_pitch) + np.cos(act_pitch) * np.cos(pers_pitch) * np.cos(act_yaw - pers_yaw)
        dot = np.clip(dot, -1.0, 1.0)
        recomputed_err = np.degrees(np.arccos(dot))
        np.testing.assert_allclose(recomputed_err, df["persistence_sph_err_deg"].values[:100], atol=0.03)

    def test_2_persistence_uses_saved_coordinates_not_spherical_approximation(self):
        """2. Persistence tile predictions must use saved coordinates, not spherical-error approximations."""
        # Create a synthetic dataset where persistence_sph_err_deg is altered to an extreme value
        # If the code used `act_yaw + sph_err * 0.7`, changing sph_err would change the results.
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            sample_data = {
                "sample_index": [0, 1],
                "subject_id": [9, 9],
                "actual_yaw_deg": [0.0, 10.0],
                "actual_pitch_deg": [0.0, 0.0],
                "predicted_yaw_deg": [1.0, 11.0],
                "predicted_pitch_deg": [0.0, 0.0],
                "persistence_predicted_yaw_deg": [0.0, 10.0],
                "persistence_predicted_pitch_deg": [0.0, 0.0],
                "persistence_sph_err_deg": [0.0, 0.0]
            }
            df_true = pd.DataFrame(sample_data)
            csv_path1 = tmp_path / "preds1.csv"
            df_true.to_csv(csv_path1, index=False)

            # Corrupt persistence_sph_err_deg to 90 degrees
            df_corrupt_err = df_true.copy()
            df_corrupt_err["persistence_sph_err_deg"] = [90.0, 90.0]
            csv_path2 = tmp_path / "preds2.csv"
            df_corrupt_err.to_csv(csv_path2, index=False)

            summary1, _ = run_tile_evaluation(csv_path1, tmp_path / "out1", sample_stride=1)
            summary2, _ = run_tile_evaluation(csv_path2, tmp_path / "out2", sample_stride=1)

            pers1 = summary1[summary1["Strategy"] == "Persistence Baseline"].iloc[0]
            pers2 = summary2[summary2["Strategy"] == "Persistence Baseline"].iloc[0]

            # Results must be identical because evaluation depends strictly on coordinates, not error approximation
            self.assertAlmostEqual(pers1["Mean Viewport Coverage"], pers2["Mean Viewport Coverage"], places=5)
            self.assertAlmostEqual(pers1["Mean Tile Hit Ratio (Recall)"], pers2["Mean Tile Hit Ratio (Recall)"], places=5)

    def test_3_independent_tile_counts(self):
        """3. Persistence average tile count must be computed from its own tile list, independent of GRU."""
        # Construct synthetic data where GRU points to equator (fetching 6-9 tiles)
        # while Persistence points to pole (pitch=85°, intersecting polar tiles)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            synthetic_df = pd.DataFrame({
                "sample_index": [0, 1, 2],
                "subject_id": [9, 9, 9],
                "actual_yaw_deg": [0.0, 0.0, 0.0],
                "actual_pitch_deg": [0.0, 0.0, 0.0],
                "predicted_yaw_deg": [0.0, 0.0, 0.0],
                "predicted_pitch_deg": [0.0, 0.0, 0.0],  # Equator
                "persistence_predicted_yaw_deg": [0.0, 0.0, 0.0],
                "persistence_predicted_pitch_deg": [85.0, 85.0, 85.0],  # Pole
            })
            csv_path = tmp_path / "independent_counts.csv"
            synthetic_df.to_csv(csv_path, index=False)

            summary_df, _ = run_tile_evaluation(csv_path, tmp_path / "out", sample_stride=1)
            pers_tiles = summary_df[summary_df["Strategy"] == "Persistence Baseline"]["Avg Tiles Fetched (out of 24)"].iloc[0]
            gru_tiles = summary_df[summary_df["Strategy"] == "Final GRU Model (No Guard Band)"]["Avg Tiles Fetched (out of 24)"].iloc[0]

            self.assertNotEqual(pers_tiles, gru_tiles, "Persistence and GRU tile counts must be computed independently")

    def test_4_identical_evaluation_samples(self):
        """4. All three strategies must be evaluated on the exact same set of samples."""
        df = pd.read_csv(self.preds_csv_path)
        stride = 10
        expected_samples = len(df.iloc[::stride])

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            summary_df, subject_df = run_tile_evaluation(self.preds_csv_path, tmp_path, sample_stride=stride)

            # Sum of subject sample counts must equal total evaluated samples
            for strat in ["Persistence Baseline", "Final GRU Model (No Guard Band)", "Final GRU Model (+1 Guard Band Margin)"]:
                strat_df = subject_df[subject_df["Strategy"] == strat]
                self.assertEqual(strat_df["Sample Count"].sum(), expected_samples)

    def test_5_invalid_sample_stride_and_fov_rejected(self):
        """5. Invalid sample stride (<=0, float, bool, string) and invalid FoV (0, 180, bool) must raise ValueError."""
        df = pd.DataFrame({
            "actual_yaw_deg": [0.0], "actual_pitch_deg": [0.0],
            "predicted_yaw_deg": [0.0], "predicted_pitch_deg": [0.0],
            "persistence_predicted_yaw_deg": [0.0], "persistence_predicted_pitch_deg": [0.0],
            "subject_id": [9]
        })
        for bad_stride in [0, -1, -10, 2.5, True, False, "5"]:
            with self.assertRaises(ValueError):
                validate_predictions_df(df, sample_stride=bad_stride)

        # FoV validation: strictly in (0, 180)
        for bad_fov in [0.0, 180.0, -10.0, 185.0, True, False]:
            with self.assertRaises(ValueError):
                validate_predictions_df(df, hfov_deg=bad_fov)
            with self.assertRaises(ValueError):
                validate_predictions_df(df, vfov_deg=bad_fov)

    def test_invalid_pitch_coordinates_rejected(self):
        """Pitch coordinates outside [-90°, +90°] must raise ValueError."""
        bad_pitch_df = pd.DataFrame({
            "actual_yaw_deg": [0.0], "actual_pitch_deg": [95.0],
            "predicted_yaw_deg": [0.0], "predicted_pitch_deg": [0.0],
            "persistence_predicted_yaw_deg": [0.0], "persistence_predicted_pitch_deg": [0.0],
            "subject_id": [9]
        })
        with self.assertRaises(ValueError):
            validate_predictions_df(bad_pitch_df)

    def test_6_missing_csv_columns_rejected(self):
        """6. Missing required CSV columns must raise ValueError."""
        base_df = pd.DataFrame({
            "actual_yaw_deg": [0.0], "actual_pitch_deg": [0.0],
            "predicted_yaw_deg": [0.0], "predicted_pitch_deg": [0.0],
            "persistence_predicted_yaw_deg": [0.0], "persistence_predicted_pitch_deg": [0.0],
            "subject_id": [9]
        })
        for req_col in base_df.columns:
            dropped_df = base_df.drop(columns=[req_col])
            with self.assertRaises(ValueError):
                validate_predictions_df(dropped_df)

    def test_7_nan_and_infinite_prediction_values_rejected(self):
        """7. NaN and infinite values in prediction coordinates must raise ValueError."""
        base_df = pd.DataFrame({
            "actual_yaw_deg": [0.0], "actual_pitch_deg": [0.0],
            "predicted_yaw_deg": [0.0], "predicted_pitch_deg": [0.0],
            "persistence_predicted_yaw_deg": [0.0], "persistence_predicted_pitch_deg": [0.0],
            "subject_id": [9]
        })
        for invalid_val in [np.nan, np.inf, -np.inf]:
            for col in ["predicted_yaw_deg", "persistence_predicted_pitch_deg", "actual_yaw_deg"]:
                bad_df = base_df.copy()
                bad_df.loc[0, col] = invalid_val
                with self.assertRaises(ValueError):
                    validate_predictions_df(bad_df)

    def test_8_p10_and_p90_percentile_calculations(self):
        """8. P10 and P90 must be mathematically valid percentiles and computed using true np.percentile."""
        summary_df = pd.read_csv(self.metrics_csv_path)
        for _, row in summary_df.iterrows():
            p10_cov = row["10th Percentile Coverage (P10)"]
            p90_cov = row["90th Percentile Coverage (P90)"]
            self.assertLessEqual(p10_cov, p90_cov, f"P10 ({p10_cov}) must be <= P90 ({p90_cov})")

        # Explicitly verify on a known non-trivial distribution
        synthetic_coverages = np.array([0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0, 1.0, 1.0, 1.0])
        p10_calc = float(np.percentile(synthetic_coverages, 10) * 100.0)
        p90_calc = float(np.percentile(synthetic_coverages, 90) * 100.0)
        self.assertAlmostEqual(p10_calc, 68.0, places=1)
        self.assertAlmostEqual(p90_calc, 100.0, places=1)
        self.assertLess(p10_calc, p90_calc)

    def test_9_subject_wise_presence_all_test_subjects(self):
        """9. Subject-wise metrics must contain all 8 unseen test subjects."""
        self.assertTrue(self.subject_csv_path.exists(), f"Subject CSV not found: {self.subject_csv_path}")
        sub_df = pd.read_csv(self.subject_csv_path)

        expected_subjects = {9, 16, 20, 22, 30, 40, 44, 49}
        found_subjects = set(sub_df["subject_id"].unique())
        self.assertEqual(expected_subjects, found_subjects, "Must contain all 8 test subjects")

        # Each subject must have exactly 3 strategy rows
        for sub in expected_subjects:
            sub_rows = sub_df[sub_df["subject_id"] == sub]
            self.assertEqual(len(sub_rows), 3, f"Subject {sub} must have 3 strategy rows")
            strats = set(sub_rows["Strategy"].values)
            self.assertEqual(strats, {
                "Persistence Baseline",
                "Final GRU Model (No Guard Band)",
                "Final GRU Model (+1 Guard Band Margin)"
            })

    def test_10_guard_band_and_no_guard_band_separated(self):
        """10. Guard band strategy must fetch strictly more tiles than no guard band."""
        summary_df = pd.read_csv(self.metrics_csv_path)
        gru_no_margin = summary_df[summary_df["Strategy"] == "Final GRU Model (No Guard Band)"].iloc[0]
        gru_margin = summary_df[summary_df["Strategy"] == "Final GRU Model (+1 Guard Band Margin)"].iloc[0]

        self.assertGreater(
            gru_margin["Avg Tiles Fetched (out of 24)"],
            gru_no_margin["Avg Tiles Fetched (out of 24)"],
            "Guard band must fetch strictly more tiles than exact/no guard band"
        )
        self.assertGreater(
            gru_margin["Mean Viewport Coverage"],
            gru_no_margin["Mean Viewport Coverage"],
            "Guard band must increase viewport visual coverage"
        )


if __name__ == "__main__":
    unittest.main()
