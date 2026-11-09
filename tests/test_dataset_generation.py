"""
Unit tests for dataset generation, subject partitioning, timestamp validation,
target indexing, and metadata alignment:
- tests/test_dataset_generation.py
"""

import unittest
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

from src.dataset_generation.sequence_generator import (
    validate_raw_trajectories_df,
    get_subject_split_mapping,
    validate_subject_splits,
    preprocess_and_validate_trajectories,
    create_sliding_windows_with_metadata,
)


class TestDatasetGeneration(unittest.TestCase):

    def setUp(self):
        self.project_root = Path(__file__).resolve().parents[1]
        self.traj_csv_path = self.project_root / "data" / "processed" / "viewport_trajectories.csv"

    def test_subject_split_mapping_and_disjointness(self):
        """Subject split must contain 33 train, 7 val, 8 test subjects with ZERO overlap."""
        splits = get_subject_split_mapping()
        validate_subject_splits(splits)

        train_set = set(splits["train"])
        val_set = set(splits["validation"])
        test_set = set(splits["test"])

        self.assertEqual(len(train_set), 33)
        self.assertEqual(len(val_set), 7)
        self.assertEqual(len(test_set), 8)
        self.assertEqual(len(train_set | val_set | test_set), 48)

        # Zero subject leakage
        self.assertEqual(len(train_set & val_set), 0)
        self.assertEqual(len(train_set & test_set), 0)
        self.assertEqual(len(val_set & test_set), 0)

    def test_target_indexing_mathematical_correctness(self):
        """
        Mathematical verification of target indexing:
        - Window i has history [i : i + seq_len]
        - Last input index is (i + seq_len - 1) = i + 9
        - Future target index is (i + seq_len + horizon - 1) = i + 14
        - Future lead time is target_timestamp - last_input_timestamp
        """
        T = 50
        seq_len = 10
        horizon = 5
        dt = 0.1
        timestamps = np.arange(T) * dt
        yaw = np.linspace(0, 50, T)
        pitch = np.linspace(-10, 10, T)
        features = np.zeros((T, 7), dtype=np.float32)

        win_dict = create_sliding_windows_with_metadata(
            features=features,
            yaw_deg=yaw,
            pitch_deg=pitch,
            timestamps=timestamps,
            subject_id=9,
            video_name="test.mp4",
            trajectory_id="sub_9_vid_test",
            seq_len=seq_len,
            horizon=horizon
        )

        expected_samples = T - (seq_len + horizon - 1)  # 50 - 14 = 36
        self.assertEqual(len(win_dict["X"]), expected_samples)

        for i in range(expected_samples):
            self.assertEqual(win_dict["window_start_indices"][i], i)
            self.assertEqual(win_dict["target_indices"][i], i + 14)
            self.assertAlmostEqual(win_dict["target_timestamps"][i], timestamps[i + 14])
            self.assertAlmostEqual(win_dict["last_input_timestamps"][i], timestamps[i + 9])
            self.assertAlmostEqual(
                win_dict["target_timestamps"][i] - win_dict["last_input_timestamps"][i],
                5 * dt,
                places=5
            )
            self.assertAlmostEqual(float(win_dict["target_yaws"][i]), float(yaw[i + 14]), places=4)
            self.assertAlmostEqual(float(win_dict["target_pitches"][i]), float(pitch[i + 14]), places=4)

    def test_raw_dataset_validation_rejects_corrupted_data(self):
        """Validates that corrupt datasets (missing columns, NaNs, missing subjects) are rejected."""
        good_df = pd.DataFrame({
            "subject_id": np.repeat(np.arange(2, 50), 10),
            "video": ["1.mp4"] * 480,
            "timestamp": np.tile(np.arange(10) * 0.1, 48),
            "pitch": [0.0] * 480,
            "yaw": [0.0] * 480,
            "roll": [0.0] * 480
        })
        # Good df should pass
        validate_raw_trajectories_df(good_df)

        # Missing required column
        bad_df = good_df.drop(columns=["yaw"])
        with self.assertRaises(ValueError):
            validate_raw_trajectories_df(bad_df)

        # NaN in timestamp
        bad_df_nan = good_df.copy()
        bad_df_nan.loc[0, "timestamp"] = np.nan
        with self.assertRaises(ValueError):
            validate_raw_trajectories_df(bad_df_nan)

        # Missing subject (only 47 subjects)
        bad_df_subs = good_df[good_df["subject_id"] != 2]
        with self.assertRaises(ValueError):
            validate_raw_trajectories_df(bad_df_subs)

    def test_duplicate_timestamp_handling_and_audit_logging(self):
        """Duplicate timestamps must be recorded in audit log and deduplicated to strictly positive dt."""
        raw_df = pd.DataFrame({
            "subject_id": [9, 9, 9, 9] + [s for s in range(2, 50) if s != 9 for _ in range(2)],
            "video": ["1.mp4"] * (4 + 47 * 2),
            "timestamp": [0.0, 0.0, 0.1, 0.2] + [0.0, 0.1] * 47,
            "pitch": [0.0] * (4 + 47 * 2),
            "yaw": [0.0] * (4 + 47 * 2),
            "roll": [0.0] * (4 + 47 * 2)
        })

        clean_df, audit = preprocess_and_validate_trajectories(raw_df, handle_initial_duplicates=True)
        self.assertEqual(audit["duplicate_timestamps_corrected"], 1)
        self.assertEqual(len(clean_df), len(raw_df) - 1)
        self.assertGreater(audit["dt_min_s"], 0.0)

        # If duplicate handling is disabled, must raise ValueError
        with self.assertRaises(ValueError):
            preprocess_and_validate_trajectories(raw_df, handle_initial_duplicates=False)


if __name__ == "__main__":
    unittest.main()

