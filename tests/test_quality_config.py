"""
Unit tests for Tile Quality and Bitrate Configuration Module:
- tests/test_quality_config.py
Phase: Phase 5, Day 2 — Module 1
"""

import unittest
import numpy as np

from src.quality.quality_config import (
    QualityLevel,
    QualityTierSpec,
    TileQualityConfig,
    TileQualityAllocation,
)


class TestTileQualityConfig(unittest.TestCase):

    def setUp(self):
        self.config = TileQualityConfig()

    def test_all_four_quality_levels_and_aliases(self):
        """Verifies parsing and properties of all four quality tiers and their aliases."""
        self.assertEqual(QualityLevel.from_string("Q1"), QualityLevel.Q1)
        self.assertEqual(QualityLevel.from_string("low"), QualityLevel.Q1)
        self.assertEqual(QualityLevel.from_string("Q1_LOW"), QualityLevel.Q1)

        self.assertEqual(QualityLevel.from_string("Q2"), QualityLevel.Q2)
        self.assertEqual(QualityLevel.from_string("medium"), QualityLevel.Q2)
        self.assertEqual(QualityLevel.from_string("med"), QualityLevel.Q2)

        self.assertEqual(QualityLevel.from_string("Q3"), QualityLevel.Q3)
        self.assertEqual(QualityLevel.from_string("high"), QualityLevel.Q3)

        self.assertEqual(QualityLevel.from_string("Q4"), QualityLevel.Q4)
        self.assertEqual(QualityLevel.from_string("ultra"), QualityLevel.Q4)

        # Rejects invalid strings
        with self.assertRaises(ValueError):
            QualityLevel.from_string("Q5")

        with self.assertRaises(ValueError):
            QualityLevel.from_string("UNKNOWN")

        # Rejects non-string types
        with self.assertRaises(TypeError):
            QualityLevel.from_string(123)

        with self.assertRaises(TypeError):
            QualityLevel.from_string(None)

    def test_default_bitrates_and_utilities(self):
        """Verifies canonical simulated bitrates and concave diminishing-returns utilities."""
        self.assertEqual(self.config.get_bitrate(QualityLevel.Q1), 100.0)
        self.assertEqual(self.config.get_bitrate(QualityLevel.Q2), 300.0)
        self.assertEqual(self.config.get_bitrate(QualityLevel.Q3), 600.0)
        self.assertEqual(self.config.get_bitrate(QualityLevel.Q4), 1000.0)

        self.assertEqual(self.config.get_utility(QualityLevel.Q1), 1.00)
        self.assertEqual(self.config.get_utility(QualityLevel.Q2), 2.00)
        self.assertEqual(self.config.get_utility(QualityLevel.Q3), 2.80)
        self.assertEqual(self.config.get_utility(QualityLevel.Q4), 3.40)

        # Monotonicity checks
        b_vals = [self.config.get_bitrate(lvl) for lvl in QualityLevel]
        u_vals = [self.config.get_utility(lvl) for lvl in QualityLevel]
        self.assertTrue(all(b_vals[i] < b_vals[i + 1] for i in range(len(b_vals) - 1)))
        self.assertTrue(all(u_vals[i] < u_vals[i + 1] for i in range(len(u_vals) - 1)))

    def test_uniform_allocation_bitrate_for_24_tiles(self):
        """Tests simulated bitrate calculation across all 24 tiles for uniform quality levels."""
        # Q1: 24 * 100 kbps = 2,400 kbps (2.4 Mbps)
        alloc_q1 = self.config.create_uniform_allocation("Q1")
        self.assertEqual(alloc_q1.total_tiles_assigned, 24)
        self.assertTrue(alloc_q1.is_complete)
        self.assertEqual(alloc_q1.total_bitrate_kbps, 2400.0)
        self.assertEqual(alloc_q1.total_bitrate_mbps, 2.4)
        self.assertAlmostEqual(alloc_q1.mean_utility, 1.00, places=5)

        # Q2: 24 * 300 kbps = 7,200 kbps (7.2 Mbps)
        alloc_q2 = self.config.create_uniform_allocation("Q2")
        self.assertEqual(alloc_q2.total_bitrate_kbps, 7200.0)
        self.assertEqual(alloc_q2.total_bitrate_mbps, 7.2)
        self.assertAlmostEqual(alloc_q2.mean_utility, 2.00, places=5)

        # Q3: 24 * 600 kbps = 14,400 kbps (14.4 Mbps)
        alloc_q3 = self.config.create_uniform_allocation("Q3")
        self.assertEqual(alloc_q3.total_bitrate_kbps, 14400.0)
        self.assertEqual(alloc_q3.total_bitrate_mbps, 14.4)
        self.assertAlmostEqual(alloc_q3.mean_utility, 2.80, places=5)

        # Q4: 24 * 1000 kbps = 24,000 kbps (24.0 Mbps)
        alloc_q4 = self.config.create_uniform_allocation("Q4")
        self.assertEqual(alloc_q4.total_bitrate_kbps, 24000.0)
        self.assertEqual(alloc_q4.total_bitrate_mbps, 24.0)
        self.assertAlmostEqual(alloc_q4.mean_utility, 3.40, places=5)

    def test_mixed_quality_assignment(self):
        """Tests mixed per-tile quality assignments, bitrate sum, and tier breakdown."""
        # 12 Low (12*100=1200), 6 Med (6*300=1800), 4 High (4*600=2400), 2 Ultra (2*1000=2000)
        # Total expected bitrate: 1200 + 1800 + 2400 + 2000 = 7400 kbps (7.4 Mbps)
        assignments = {}
        for t in range(0, 12):
            assignments[t] = QualityLevel.Q1
        for t in range(12, 18):
            assignments[t] = QualityLevel.Q2
        for t in range(18, 22):
            assignments[t] = QualityLevel.Q3
        for t in range(22, 24):
            assignments[t] = QualityLevel.Q4

        alloc = TileQualityAllocation(assignments=assignments, config=self.config)
        self.assertTrue(alloc.is_complete)
        self.assertEqual(alloc.total_tiles_assigned, 24)
        self.assertEqual(alloc.total_bitrate_kbps, 7400.0)
        self.assertEqual(alloc.total_bitrate_mbps, 7.4)

        dist = alloc.tier_distribution
        self.assertEqual(dist[QualityLevel.Q1], 12)
        self.assertEqual(dist[QualityLevel.Q2], 6)
        self.assertEqual(dist[QualityLevel.Q3], 4)
        self.assertEqual(dist[QualityLevel.Q4], 2)

        expected_mean_util = (12 * 1.0 + 6 * 2.0 + 4 * 2.8 + 2 * 3.4) / 24.0
        self.assertAlmostEqual(alloc.mean_utility, expected_mean_util, places=5)

    def test_budget_satisfied_and_exceeded(self):
        """Verifies bandwidth budget satisfaction, deficit, and headroom calculations."""
        # Allocation bitrate = 7400 kbps
        assignments = {t: QualityLevel.Q1 for t in range(24)}
        for t in range(20, 24):
            assignments[t] = QualityLevel.Q4
        # 20 * 100 + 4 * 1000 = 6000 kbps
        alloc = TileQualityAllocation(assignments=assignments, config=self.config)
        self.assertEqual(alloc.total_bitrate_kbps, 6000.0)

        # Budget strictly greater
        self.assertTrue(alloc.satisfies_budget(8000.0))
        self.assertEqual(alloc.remaining_budget(8000.0), 2000.0)

        # Budget exactly equal
        self.assertTrue(alloc.satisfies_budget(6000.0))
        self.assertEqual(alloc.remaining_budget(6000.0), 0.0)

        # Budget exceeded
        self.assertFalse(alloc.satisfies_budget(5000.0))
        self.assertEqual(alloc.remaining_budget(5000.0), -1000.0)

        # Configuration direct method
        self.assertTrue(self.config.satisfies_budget(assignments, 6000.0))
        self.assertFalse(self.config.satisfies_budget(assignments, 5999.0))

        # Rejects negative budget or invalid budget types
        with self.assertRaises(ValueError):
            alloc.satisfies_budget(-100.0)
        with self.assertRaises(TypeError):
            alloc.satisfies_budget(True)
        with self.assertRaises(TypeError):
            alloc.satisfies_budget("5000")

    def test_invalid_tile_ids_rejected(self):
        """Rejects negative tile IDs, out-of-bounds tile IDs, and non-integer types."""
        # Negative tile ID
        with self.assertRaises(ValueError):
            self.config.validate_tile_id(-1)

        # Out of bounds for 24 tiles (max valid index is 23)
        with self.assertRaises(ValueError):
            self.config.validate_tile_id(24)

        # Non-integer types
        with self.assertRaises(TypeError):
            self.config.validate_tile_id(1.5)
        with self.assertRaises(TypeError):
            self.config.validate_tile_id("0")
        with self.assertRaises(TypeError):
            self.config.validate_tile_id(True)  # Booleans are not tile indices

    def test_invalid_quality_levels_in_allocation_rejected(self):
        """Rejects invalid tier names inside an allocation dictionary."""
        bad_alloc = {0: "Q1", 1: "INVALID_TIER"}
        with self.assertRaises(ValueError):
            self.config.validate_allocation(bad_alloc, allow_incomplete=True)

    def test_invalid_bitrate_or_utility_configuration_rejected(self):
        """Rejects custom configurations with non-positive, non-finite, non-monotonic values or invalid grids."""
        # Negative bitrate
        with self.assertRaises(ValueError):
            TileQualityConfig(custom_bitrates_kbps={QualityLevel.Q1: -50.0})

        # Zero bitrate
        with self.assertRaises(ValueError):
            TileQualityConfig(custom_bitrates_kbps={QualityLevel.Q1: 0.0})

        # NaN and Infinite bitrates
        with self.assertRaises(ValueError):
            TileQualityConfig(custom_bitrates_kbps={QualityLevel.Q1: float("nan")})
        with self.assertRaises(ValueError):
            TileQualityConfig(custom_bitrates_kbps={QualityLevel.Q4: float("inf")})

        # Non-monotonic bitrates (Q2 < Q1)
        with self.assertRaises(ValueError):
            TileQualityConfig(custom_bitrates_kbps={QualityLevel.Q1: 500.0, QualityLevel.Q2: 400.0})

        # Negative and zero utility
        with self.assertRaises(ValueError):
            TileQualityConfig(custom_utilities={QualityLevel.Q1: -1.0})
        with self.assertRaises(ValueError):
            TileQualityConfig(custom_utilities={QualityLevel.Q1: 0.0})

        # NaN and Infinite utilities
        with self.assertRaises(ValueError):
            TileQualityConfig(custom_utilities={QualityLevel.Q1: float("nan")})
        with self.assertRaises(ValueError):
            TileQualityConfig(custom_utilities={QualityLevel.Q4: float("inf")})

        # Non-monotonic utilities (Q3 < Q2)
        with self.assertRaises(ValueError):
            TileQualityConfig(custom_utilities={QualityLevel.Q2: 2.5, QualityLevel.Q3: 2.0})

        # Non-numeric parameters
        with self.assertRaises(TypeError):
            TileQualityConfig(custom_bitrates_kbps={QualityLevel.Q1: "100"})
        with self.assertRaises(TypeError):
            TileQualityConfig(custom_utilities={QualityLevel.Q1: True})

        # Contradicting canonical project grid without explicit flag
        with self.assertRaises(ValueError):
            TileQualityConfig(grid_rows=3, grid_cols=5)
        with self.assertRaises(ValueError):
            TileQualityConfig(grid_rows=4, grid_cols=8)

        # Experimental non-canonical grid permitted when explicitly flagged
        cfg_exp = TileQualityConfig(grid_rows=3, grid_cols=5, allow_non_canonical_grid=True)
        self.assertEqual(cfg_exp.total_tiles, 15)
        self.assertFalse(cfg_exp.is_canonical_grid)

        # Non-positive or boolean grid dimensions
        with self.assertRaises(ValueError):
            TileQualityConfig(grid_rows=0)
        with self.assertRaises(ValueError):
            TileQualityConfig(grid_cols=-1)
        with self.assertRaises(ValueError):
            TileQualityConfig(grid_rows=True)

    def test_empty_and_incomplete_allocations(self):
        """Verifies handling of empty and partial tile allocations."""
        # Empty allocation rejected when full grid required
        with self.assertRaises(ValueError):
            self.config.validate_allocation({}, allow_incomplete=False)

        # Incomplete allocation (only 5 tiles)
        partial = {0: "Q1", 1: "Q2", 2: "Q3", 3: "Q4", 4: "Q1"}
        with self.assertRaises(ValueError):
            self.config.validate_allocation(partial, allow_incomplete=False)

        # Permitted when allow_incomplete=True
        cleaned = self.config.validate_allocation(partial, allow_incomplete=True)
        self.assertEqual(len(cleaned), 5)

        # TileQualityAllocation tracking missing tiles
        alloc = TileQualityAllocation(assignments=partial, config=self.config)
        self.assertFalse(alloc.is_complete)
        self.assertEqual(alloc.total_tiles_assigned, 5)
        self.assertEqual(len(alloc.missing_tiles), 19)
        self.assertIn(5, alloc.missing_tiles)
        self.assertNotIn(0, alloc.missing_tiles)

    def test_weighted_utility_calculation(self):
        """Verifies spatial importance weighting for viewport-aware utility calculation and strict validation."""
        # Assign tiles 0..23
        assignments = {t: QualityLevel.Q1 for t in range(24)}
        assignments[10] = QualityLevel.Q4  # Ultra: utility 3.4
        assignments[11] = QualityLevel.Q3  # High: utility 2.8

        alloc = TileQualityAllocation(assignments=assignments, config=self.config)

        # Weights concentrated on focal tile 10 (0.7) and tile 11 (0.3)
        weights = {10: 0.7, 11: 0.3}
        # Expected: 0.7 * 3.4 + 0.3 * 2.8 = 2.38 + 0.84 = 3.22
        w_util = alloc.weighted_utility(weights)
        self.assertAlmostEqual(w_util, 3.22, places=5)

        # Empty weights rejected
        with self.assertRaises(ValueError):
            alloc.weighted_utility({})

        # Zero-sum weights rejected
        with self.assertRaises(ValueError):
            alloc.weighted_utility({10: 0.0, 11: 0.0})

        # Weights not summing to 1.0 within tolerance rejected
        with self.assertRaises(ValueError):
            alloc.weighted_utility({10: 0.4, 11: 0.3})  # sum = 0.7
        with self.assertRaises(ValueError):
            alloc.weighted_utility({10: 0.7, 11: 0.6})  # sum = 1.3

        # Rejects negative weights
        with self.assertRaises(ValueError):
            alloc.weighted_utility({10: -0.5, 11: 1.5})

        # Rejects NaN and infinite weights
        with self.assertRaises(ValueError):
            alloc.weighted_utility({10: float("nan"), 11: 0.5})
        with self.assertRaises(ValueError):
            alloc.weighted_utility({10: float("inf"), 11: 0.5})

        # Rejects boolean weights
        with self.assertRaises(TypeError):
            alloc.weighted_utility({10: True, 11: 0.0})

        # Rejects non-dict weights
        with self.assertRaises(TypeError):
            alloc.weighted_utility([0.7, 0.3])

        # Missing tiles with positive viewport weights rejected (do not silently ignore missing tiles)
        partial_alloc = TileQualityAllocation(
            assignments={t: QualityLevel.Q1 for t in range(10)},
            config=self.config
        )
        with self.assertRaises(ValueError):
            partial_alloc.weighted_utility({0: 0.5, 15: 0.5})  # Tile 15 unassigned but has weight 0.5

    def test_tile_getter_setter_and_serialization(self):
        """Verifies querying, updating individual tiles, and dictionary serialization."""
        alloc = self.config.create_uniform_allocation("Q1")
        self.assertEqual(alloc.get_tile_quality(0), QualityLevel.Q1)

        # Update tile 0 to Ultra
        alloc.set_tile_quality(0, "Ultra")
        self.assertEqual(alloc.get_tile_quality(0), QualityLevel.Q4)
        self.assertEqual(alloc.total_bitrate_kbps, 23 * 100.0 + 1000.0)  # 3300 kbps

        # Serialization to dict
        summary = alloc.to_dict()
        self.assertEqual(summary["total_tiles_assigned"], 24)
        self.assertTrue(summary["is_complete"])
        self.assertEqual(summary["total_bitrate_kbps"], 3300.0)
        self.assertEqual(summary["total_bitrate_mbps"], 3.3)
        self.assertEqual(summary["tier_distribution"]["Q4"], 1)
        self.assertEqual(summary["tier_distribution"]["Q1"], 23)
        self.assertEqual(summary["assignments"][0], "Q4")


if __name__ == "__main__":
    unittest.main()
