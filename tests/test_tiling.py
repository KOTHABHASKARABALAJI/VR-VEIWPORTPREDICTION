"""
Unit tests for 360° video spatial grid tiling and viewport mapping:
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

    def test_circular_neighbors(self):
        """Column 0 (-180°) must wrap to Column 5 (+180°)."""
        tile_0_neighbors = self.grid.get_adjacent_tiles(0, include_diagonals=False)
        # Tile 0 is row 0, col 0. Right neighbor is 1, Bottom is 6, Left wrap is 5
        self.assertIn(1, tile_0_neighbors)
        self.assertIn(5, tile_0_neighbors)
        self.assertIn(6, tile_0_neighbors)


class TestViewportMapper(unittest.TestCase):
    def setUp(self):
        self.grid = TileGridPartitioner()
        self.mapper = ViewportMapper(grid_partitioner=self.grid, hfov_deg=100.0, vfov_deg=100.0)

    def test_intersection_weights_sum_to_one(self):
        """Sum of fractional ray weights across active tiles must equal 1.0."""
        inter = self.mapper.get_viewport_intersection(0.0, 0.0)
        total_weight = sum(inter.tile_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=5)
        self.assertIn(inter.primary_tile, inter.active_tiles)

    def test_perfect_prediction_recall(self):
        """When predicted viewport equals ground truth, recall and coverage must be 1.0."""
        res = self.mapper.evaluate_tile_prediction(15.0, -10.0, 15.0, -10.0)
        self.assertAlmostEqual(res["tile_hit_ratio_recall"], 1.0, places=5)
        self.assertAlmostEqual(res["tile_precision"], 1.0, places=5)
        self.assertAlmostEqual(res["viewport_coverage"], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()

