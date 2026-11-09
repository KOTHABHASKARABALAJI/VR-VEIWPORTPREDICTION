"""
Unit tests for 360° video spatial grid tiling, polar coordinates, and viewport mapping:
- tests/test_tiling.py
"""

import unittest
import numpy as np

from src.tiling.grid_partitioner import TileGridPartitioner
from src.tiling.viewport_mapper import ViewportMapper


class TestTileGridPartitioner(unittest.TestCase):
    def setUp(self):
        self.grid = TileGridPartitioner(frame_width=3840, frame_height=1920, num_rows=4, num_cols=6)

    def test_grid_initialization(self):
        self.assertEqual(len(self.grid.tiles), 24)
        self.assertEqual(self.grid.tile_width, 640)
        self.assertEqual(self.grid.tile_height, 480)

    def test_invalid_parameters_rejected(self):
        """Invalid dimensions, negative values, and non-divisible dimensions must raise ValueError."""
        with self.assertRaises(ValueError):
            TileGridPartitioner(frame_width=3841, frame_height=1920)
        with self.assertRaises(ValueError):
            TileGridPartitioner(frame_width=3840, frame_height=1920, num_rows=0)
        with self.assertRaises(ValueError):
            TileGridPartitioner(frame_width=3840, frame_height=1920, num_cols=-6)

    def test_solid_angle_sum(self):
        """Sum of all 24 tile solid angles must equal total sphere area (4*pi steradians)."""
        total_sa = sum(t.solid_angle_sr for t in self.grid.tiles.values())
        self.assertAlmostEqual(total_sa, 4.0 * np.pi, places=4)

    def test_pixel_to_spherical_roundtrip(self):
        """Converting spherical -> pixel -> spherical should preserve angles."""
        test_points = [(0.0, 0.0), (60.0, 30.0), (-120.0, -45.0), (179.0, 85.0)]
        for yaw, pitch in test_points:
            px, py = self.grid.spherical_to_pixel(yaw, pitch)
            ryaw, rpitch = self.grid.pixel_to_spherical(px, py)
            self.assertAlmostEqual(yaw, ryaw, delta=0.5)
            self.assertAlmostEqual(pitch, rpitch, delta=0.5)

    def test_polar_singularity_mapping(self):
        """At exact poles (+90° and -90° pitch), pixel Y must be clamped to boundaries (0 and frame_height)."""
        px_top, py_top = self.grid.spherical_to_pixel(0.0, 90.0)
        self.assertEqual(py_top, 0.0)

        px_bot, py_bot = self.grid.spherical_to_pixel(0.0, -90.0)
        self.assertEqual(py_bot, float(self.grid.frame_height))

    def test_circular_neighbors(self):
        """Column 0 (-180°) must wrap to Column 5 (+180°)."""
        tile_0_neighbors = self.grid.get_adjacent_tiles(0, include_diagonals=False)
        self.assertIn(1, tile_0_neighbors)
        self.assertIn(5, tile_0_neighbors)
        self.assertIn(6, tile_0_neighbors)


class TestViewportMapper(unittest.TestCase):
    def setUp(self):
        self.grid = TileGridPartitioner()
        self.mapper = ViewportMapper(grid_partitioner=self.grid, hfov_deg=100.0, vfov_deg=100.0)

    def test_invalid_fov_rejected(self):
        """FoV <= 0 or >= 180 or boolean must raise ValueError."""
        with self.assertRaises(ValueError):
            ViewportMapper(hfov_deg=0.0)
        with self.assertRaises(ValueError):
            ViewportMapper(hfov_deg=180.0)
        with self.assertRaises(ValueError):
            ViewportMapper(vfov_deg=-10.0)
        with self.assertRaises(ValueError):
            ViewportMapper(hfov_deg=True)

    def test_intersection_weights_sum_to_one(self):
        """Sum of fractional ray weights across active tiles must equal 1.0."""
        inter = self.mapper.get_viewport_intersection(0.0, 0.0)
        total_weight = sum(inter.tile_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=5)
        self.assertIn(inter.primary_tile, inter.active_tiles)
        for w in inter.tile_weights.values():
            self.assertGreaterEqual(w, 0.0)

    def test_perfect_prediction_recall(self):
        """When predicted viewport equals ground truth, recall and coverage must be 1.0."""
        res = self.mapper.evaluate_tile_prediction(15.0, -10.0, 15.0, -10.0)
        self.assertAlmostEqual(res["tile_hit_ratio_recall"], 1.0, places=5)
        self.assertAlmostEqual(res["tile_precision"], 1.0, places=5)
        self.assertAlmostEqual(res["viewport_coverage"], 1.0, places=5)

    def test_sampling_sensitivity_experiment(self):
        """Ray sampling sensitivity experiment must execute and report stability across resolutions."""
        df_sens = ViewportMapper.run_sampling_sensitivity_experiment(resolutions=[21, 41])
        self.assertFalse(df_sens.empty)
        self.assertIn("max_weight_diff_vs_21x21", df_sens.columns)
        # Weight difference between 21x21 and 41x41 should be small (< 0.05)
        max_diff = df_sens["max_weight_diff_vs_21x21"].max()
        self.assertLess(max_diff, 0.05)


if __name__ == "__main__":
    unittest.main()
