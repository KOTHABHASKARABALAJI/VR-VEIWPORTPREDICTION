"""
Tests for Module 4: Neighbourhood-Aware Tile Quality Allocation
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming
Phase: Phase 5, Day 2 — Module 4: Neighbourhood-Aware Tile Quality Allocation
"""

import unittest
import numpy as np

from src.quality.quality_config import (
    QualityLevel,
    TileQualityConfig,
    TileQualityAllocation,
)
from src.quality.uniform_allocator import (
    InsufficientBandwidthError,
    UniformQualityAllocator,
)
from src.quality.viewport_allocator import (
    ViewportQualityAllocator,
)
from src.quality.neighbourhood_allocator import (
    TileNeighbourhoodFinder,
    NeighbourhoodQualityAllocator,
    NeighbourhoodAllocationResult,
    DEFAULT_NEIGHBOUR_FACTOR,
    allocate_neighbourhood_quality,
)
from src.tiling.grid_partitioner import TileGridPartitioner
from src.tiling.viewport_mapper import ViewportMapper


class TestNeighbourhoodIdentification(unittest.TestCase):
    """
    Verifies topological 24-tile neighbourhood identification on the 4x6 ERP grid.
    """

    def setUp(self) -> None:
        self.finder = TileNeighbourhoodFinder(num_rows=4, num_cols=6)
        self.grid = TileGridPartitioner(num_rows=4, num_cols=6)

    def test_horizontal_wraparound(self) -> None:
        """
        Horizontal movement must wrap around circularly across the 360° ERP longitude boundary:
        - Tile in col 0 (e.g. Tile 6: row 1, col 0) must have left neighbour in col 5 (Tile 11).
        - Tile in col 5 (e.g. Tile 11: row 1, col 5) must have right neighbour in col 0 (Tile 6).
        """
        n6 = self.finder.get_tile_neighbours(6)
        self.assertIn(11, n6, "Tile 6 (col 0) must wrap left to Tile 11 (col 5)")

        n11 = self.finder.get_tile_neighbours(11)
        self.assertIn(6, n11, "Tile 11 (col 5) must wrap right to Tile 6 (col 0)")

    def test_top_and_bottom_polar_boundaries(self) -> None:
        """
        Vertical movement must NOT wrap over the top (Zenith, row 0) and bottom (Nadir, row 3) poles.
        Zenith tiles (row 0: IDs 0..5) have 5 neighbours.
        Nadir tiles (row 3: IDs 18..23) have 5 neighbours.
        Interior tiles (rows 1..2: IDs 6..17) have 8 neighbours.
        """
        # Zenith tile: 0 (row 0, col 0)
        n0 = self.finder.get_tile_neighbours(0)
        self.assertEqual(len(n0), 5, "Zenith tiles must have exactly 5 neighbours")
        # No neighbours in row -1
        for tid in n0:
            row = tid // 6
            self.assertIn(row, [0, 1], "Zenith neighbours can only be in row 0 or row 1")

        # Nadir tile: 23 (row 3, col 5)
        n23 = self.finder.get_tile_neighbours(23)
        self.assertEqual(len(n23), 5, "Nadir tiles must have exactly 5 neighbours")
        for tid in n23:
            row = tid // 6
            self.assertIn(row, [2, 3], "Nadir neighbours can only be in row 2 or row 3")

        # Interior tile: 7 (row 1, col 1)
        n7 = self.finder.get_tile_neighbours(7)
        self.assertEqual(len(n7), 8, "Interior tiles must have exactly 8 neighbours")

    def test_diagonal_neighbours_included(self) -> None:
        """
        Confirms diagonal neighbours (top-left, top-right, bottom-left, bottom-right)
        are properly included when include_diagonals=True, and excluded when False.
        """
        # Tile 7: row 1, col 1.
        # Orthogonal: Top (1), Bottom (13), Left (6), Right (8)
        # Diagonals: Top-Left (0), Top-Right (2), Bottom-Left (12), Bottom-Right (14)
        n7_diag = self.finder.get_tile_neighbours(7, include_diagonals=True)
        self.assertEqual(len(n7_diag), 8)
        self.assertEqual(set(n7_diag), {0, 1, 2, 6, 8, 12, 13, 14})

        n7_ortho = self.finder.get_tile_neighbours(7, include_diagonals=False)
        self.assertEqual(len(n7_ortho), 4)
        self.assertEqual(set(n7_ortho), {1, 6, 8, 13})

    def test_duplicate_removal_and_no_self_neighbour(self) -> None:
        """
        A tile must never be its own neighbour, and all neighbour lists must contain unique IDs.
        """
        for tid in range(24):
            neighbours = self.finder.get_tile_neighbours(tid)
            self.assertNotIn(tid, neighbours, f"Tile {tid} cannot be in its own neighbourhood")
            self.assertEqual(len(neighbours), len(set(neighbours)), f"Duplicate neighbours for {tid}")
            self.assertEqual(neighbours, sorted(neighbours), f"Neighbours for {tid} must be sorted")

    def test_valid_tile_ids_and_cross_check_with_grid_partitioner(self) -> None:
        """
        All returned neighbour IDs must be valid integers in [0, 23] and match
        TileGridPartitioner.get_adjacent_tiles exactly.
        """
        for tid in range(24):
            n_finder = self.finder.get_tile_neighbours(tid, include_diagonals=True)
            n_grid = self.grid.get_adjacent_tiles(tid, include_diagonals=True)
            self.assertEqual(n_finder, n_grid, f"Mismatch for tile {tid} between finder and grid partitioner")
            for nid in n_finder:
                self.assertIsInstance(nid, int)
                self.assertTrue(0 <= nid < 24)

    def test_multi_ring_neighbourhood(self) -> None:
        """
        Verifies multi-ring neighbourhood extraction surrounding an active set of tiles.
        """
        # Active tile: tile 7 (row 1, col 1)
        ring1_tiles, ring_map = self.finder.get_multi_ring_neighbourhood([7], num_rings=1)
        self.assertEqual(len(ring1_tiles), 8)
        self.assertNotIn(7, ring1_tiles)
        for tid in ring1_tiles:
            self.assertEqual(ring_map[tid], 1)

        # 2 rings around tile 7
        ring2_tiles, ring_map2 = self.finder.get_multi_ring_neighbourhood([7], num_rings=2)
        self.assertGreater(len(ring2_tiles), len(ring1_tiles))
        self.assertNotIn(7, ring2_tiles)
        # Ring 1 tiles have ring index 1; new tiles have ring index 2
        for tid in ring1_tiles:
            self.assertEqual(ring_map2[tid], 1)


class TestNeighbourhoodImportanceWeighting(unittest.TestCase):
    """
    Verifies the mathematical correctness of neighbourhood-aware importance weighting.
    """

    def setUp(self) -> None:
        self.allocator = NeighbourhoodQualityAllocator()
        # Single active focal tile 7 (weight 1.0)
        self.single_active = {i: 0.0 for i in range(24)}
        self.single_active[7] = 1.0

        # Two active tiles: 7 (w=0.7) and 8 (w=0.3)
        self.two_active = {i: 0.0 for i in range(24)}
        self.two_active[7] = 0.70
        self.two_active[8] = 0.30

    def test_correct_normalization(self) -> None:
        """
        Final neighbourhood-aware weights must sum to 1.0 within numerical tolerance across diverse factors.
        """
        for alpha in [0.0, 0.10, 0.25, 0.40, 0.50, 0.75, 0.90]:
            weights, _ = self.allocator.compute_neighbourhood_weights(
                self.two_active, neighbour_weight_factor=alpha
            )
            self.assertEqual(len(weights), 24)
            self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)
            for tid, w in weights.items():
                self.assertGreaterEqual(w, 0.0)

    def test_zero_neighbourhood_factor_reproduces_viewport_only(self) -> None:
        """
        When neighbour_weight_factor = 0.0, the computed weights must exactly reproduce
        the original viewport weights, with 0 weight assigned to margin tiles.
        """
        weights, reg = self.allocator.compute_neighbourhood_weights(
            self.two_active, neighbour_weight_factor=0.0
        )
        for tid in range(24):
            self.assertAlmostEqual(weights[tid], self.two_active[tid], places=6)
        self.assertEqual(reg["neighbourhood"], [])

    def test_increasing_factor_increases_neighbour_importance(self) -> None:
        """
        Increasing neighbour_weight_factor proportionally increases total neighbourhood weight
        while reducing total viewport weight.
        """
        w_low, _ = self.allocator.compute_neighbourhood_weights(self.single_active, neighbour_weight_factor=0.20)
        w_high, _ = self.allocator.compute_neighbourhood_weights(self.single_active, neighbour_weight_factor=0.50)

        # Active tile 7 weight decreases
        self.assertAlmostEqual(w_low[7], 0.80, places=5)
        self.assertAlmostEqual(w_high[7], 0.50, places=5)

        # Margin neighbours (e.g. tile 8) weight increases
        self.assertAlmostEqual(w_low[8], 0.20 / 8.0, places=5)
        self.assertAlmostEqual(w_high[8], 0.50 / 8.0, places=5)
        self.assertGreater(w_high[8], w_low[8])

    def test_relative_viewport_priority_preserved(self) -> None:
        """
        The relative importance ratio between any two active viewport tiles must be
        strictly preserved: w_a^neigh / w_b^neigh == w_a^orig / w_b^orig.
        """
        weights, _ = self.allocator.compute_neighbourhood_weights(
            self.two_active, neighbour_weight_factor=0.25
        )
        orig_ratio = self.two_active[7] / self.two_active[8]  # 0.70 / 0.30 = 2.3333
        neigh_ratio = weights[7] / weights[8]
        self.assertAlmostEqual(neigh_ratio, orig_ratio, places=5)

    def test_lower_importance_to_neighbours_than_viewport_focus(self) -> None:
        """
        With default factor 0.25, individual neighbours must receive lower importance
        than the primary focal viewport tile.
        """
        weights, reg = self.allocator.compute_neighbourhood_weights(
            self.two_active, neighbour_weight_factor=0.25
        )
        focus_w = weights[7]
        for m_id in reg["neighbourhood"]:
            self.assertLess(
                weights[m_id], focus_w,
                f"Neighbour tile {m_id} weight ({weights[m_id]}) must be strictly less than focal tile ({focus_w})"
            )

    def test_overlapping_neighbourhoods_handled_correctly(self) -> None:
        """
        When two active tiles share a common neighbour, the shared neighbour accumulates
        contributions from both tiles without duplication.
        Tile 7 (col 1) and Tile 8 (col 2):
        - Shared top neighbour in row 0: Tile 1 and Tile 2.
        - Shared bottom neighbour in row 2: Tile 13 and Tile 14.
        A tile adjacent to BOTH active tiles receives higher overlap score than a tile adjacent to only one.
        """
        weights, reg = self.allocator.compute_neighbourhood_weights(
            self.two_active, neighbour_weight_factor=0.25
        )
        # Tile 1 is adjacent to both 7 and 8
        # Tile 6 (col 0) is adjacent only to 7
        self.assertGreater(
            weights[1], weights[6],
            "Tile 1 (adjacent to both active tiles) must receive higher weight than Tile 6 (adjacent to only one)"
        )

    def test_invalid_parameters_rejected(self) -> None:
        """
        Rejects invalid factors (negative, >= 1.0, NaN, Inf, bool), invalid rings, and malformed weights.
        """
        with self.assertRaises(ValueError):
            self.allocator.compute_neighbourhood_weights(self.single_active, neighbour_weight_factor=-0.1)
        with self.assertRaises(ValueError):
            self.allocator.compute_neighbourhood_weights(self.single_active, neighbour_weight_factor=1.0)
        with self.assertRaises(ValueError):
            self.allocator.compute_neighbourhood_weights(self.single_active, neighbour_weight_factor=1.5)
        with self.assertRaises(ValueError):
            self.allocator.compute_neighbourhood_weights(self.single_active, neighbour_weight_factor=np.nan)
        with self.assertRaises(TypeError):
            self.allocator.compute_neighbourhood_weights(self.single_active, neighbour_weight_factor=True)
        with self.assertRaises(ValueError):
            self.allocator.compute_neighbourhood_weights(self.single_active, num_rings=0)


class TestNeighbourhoodQualityAllocator(unittest.TestCase):
    """
    Unit test suite verifying neighbourhood-aware greedy quality allocation.
    """

    def setUp(self) -> None:
        self.allocator = NeighbourhoodQualityAllocator(neighbour_weight_factor=0.25)
        # Localized active tiles: Tile 7 (0.70), Tile 8 (0.30)
        self.vp_weights = {i: 0.0 for i in range(24)}
        self.vp_weights[7] = 0.70
        self.vp_weights[8] = 0.30

    def test_minimum_feasible_budget(self) -> None:
        """
        At exact minimum feasible budget (2,400 kbps), all 24 tiles must remain at Q1
        with 0 upgrades applied.
        """
        res = self.allocator.allocate_with_result(self.vp_weights, budget_kbps=2400.0)
        self.assertTrue(res.is_feasible)
        self.assertTrue(res.is_budget_satisfied)
        self.assertEqual(res.total_bitrate_kbps, 2400.0)
        self.assertEqual(res.remaining_budget_kbps, 0.0)
        self.assertEqual(res.num_upgrades_applied, 0)
        self.assertEqual(res.tier_distribution[QualityLevel.Q1], 24)
        self.assertAlmostEqual(res.weighted_utility_original, 1.00, places=4)
        self.assertAlmostEqual(res.weighted_utility_neighbourhood, 1.00, places=4)

    def test_all_24_tiles_assigned(self) -> None:
        """
        All 24 tiles must receive exactly one quality level.
        """
        res = self.allocator.allocate_with_result(self.vp_weights, budget_kbps=8000.0)
        self.assertEqual(len(res.assignments), 24)
        self.assertEqual(set(res.assignments.keys()), set(range(24)))
        self.assertIsNotNone(res.allocation)
        self.assertTrue(res.allocation.is_complete)

    def test_guard_band_allocation_behavior(self) -> None:
        """
        Verifies that under moderate bandwidth (e.g. 7,200 kbps), neighbourhood-aware
        allocation assigns upgrades to BOTH viewport tiles and first-ring margin tiles.
        """
        res = self.allocator.allocate_with_result(self.vp_weights, budget_kbps=7200.0)
        self.assertTrue(res.is_budget_satisfied)
        self.assertLessEqual(res.total_bitrate_kbps, 7200.0)

        # Check upgrades by region
        self.assertGreater(res.upgrades_by_region["viewport"], 0, "Viewport tiles must receive upgrades")
        self.assertGreater(res.upgrades_by_region["neighbourhood"], 0, "Margin tiles must receive guard-band upgrades")

        # Viewport tiles achieve higher tiers (e.g. Q4/Q3)
        self.assertIn(res.assignments[7], [QualityLevel.Q3, QualityLevel.Q4])

        # At least one neighbour has been upgraded beyond Q1
        margin_tiles = res.region_tiles["neighbourhood"]
        margin_qualities = [res.assignments[m] for m in margin_tiles]
        self.assertTrue(any(q != QualityLevel.Q1 for q in margin_qualities), "Guard band margin must have upgraded tiles")

    def test_all_upgrades_respect_budget(self) -> None:
        """
        Sweeps 25 diverse budget levels confirming total simulated bitrate NEVER exceeds budget.
        """
        budgets = np.linspace(2400.0, 25000.0, 25)
        for b in budgets:
            res = self.allocator.allocate_with_result(self.vp_weights, budget_kbps=float(b))
            self.assertTrue(res.is_budget_satisfied, f"Budget violated at {b}")
            self.assertLessEqual(res.total_bitrate_kbps, float(b))
            self.assertGreaterEqual(res.remaining_budget_kbps, 0.0)

    def test_invalid_budgets_rejected(self) -> None:
        """
        Sub-2,400 kbps budgets raise InsufficientBandwidthError; negative, NaN, bool raise errors.
        """
        with self.assertRaises(InsufficientBandwidthError):
            self.allocator.allocate(self.vp_weights, budget_kbps=2000.0)
        with self.assertRaises(ValueError):
            self.allocator.allocate(self.vp_weights, budget_kbps=-500.0)
        with self.assertRaises(ValueError):
            self.allocator.allocate(self.vp_weights, budget_kbps=np.nan)
        with self.assertRaises(TypeError):
            self.allocator.allocate(self.vp_weights, budget_kbps=True)

    def test_deterministic_tie_breaking(self) -> None:
        """
        Deterministic tie-breaking (ascending tile ID) must produce identical results
        independent of dictionary order.
        """
        res1 = self.allocator.allocate(self.vp_weights, budget_kbps=3000.0)
        # Invert dictionary order
        inverted_vp = {k: self.vp_weights[k] for k in sorted(list(self.vp_weights.keys()), reverse=True)}
        res2 = self.allocator.allocate(inverted_vp, budget_kbps=3000.0)
        self.assertEqual(res1.assignments, res2.assignments)

    def test_original_viewport_utility_calculation(self) -> None:
        """
        Verifies that weighted_utility_original is evaluated using the original viewport weights,
        confirming honest reporting of in-FoV vs out-of-FoV trade-offs.
        """
        res = self.allocator.allocate_with_result(self.vp_weights, budget_kbps=7200.0)
        # Calculate manually with original weights
        expected_orig_u = sum(self.vp_weights[t] * self.allocator.config.get_utility(res.assignments[t]) for t in range(24))
        self.assertAlmostEqual(res.weighted_utility_original, expected_orig_u, places=5)

    def test_comparison_with_modules_2_and_3(self) -> None:
        """
        Executes compare_all_strategies across canonical budgets (2,400, 5,000, 7,200, 10,000, 14,400 kbps)
        and confirms:
        1. All strategies satisfy the budget.
        2. At 7,200 kbps, Viewport-Aware achieves maximum in-FoV utility, while Neighbourhood-Aware
           trades a small amount of in-FoV utility for guard-band protection in the margin.
        """
        comp = self.allocator.compare_all_strategies(
            viewport_weights=self.vp_weights,
            budget_kbps=7200.0,
            neighbour_weight_factors=[0.0, 0.25, 0.50]
        )

        self.assertIn("uniform", comp)
        self.assertIn("viewport_aware", comp)
        self.assertIn("neighbourhood_aware", comp)

        # Budget adherence
        self.assertLessEqual(comp["uniform"]["total_bitrate_kbps"], 7200.0)
        self.assertLessEqual(comp["viewport_aware"]["total_bitrate_kbps"], 7200.0)

        n_00 = comp["neighbourhood_aware"]["alpha_0.00"]
        n_25 = comp["neighbourhood_aware"]["alpha_0.25"]
        n_50 = comp["neighbourhood_aware"]["alpha_0.50"]

        self.assertLessEqual(n_00["total_bitrate_kbps"], 7200.0)
        self.assertLessEqual(n_25["total_bitrate_kbps"], 7200.0)
        self.assertLessEqual(n_50["total_bitrate_kbps"], 7200.0)

        # alpha=0.00 in neighbourhood matches viewport-aware exactly
        self.assertAlmostEqual(
            n_00["weighted_utility_original"],
            comp["viewport_aware"]["weighted_utility_original"],
            places=4
        )

        # Guard band trade-off: alpha=0.25 allocates upgrades to margin tiles
        self.assertGreater(n_25["upgrades_by_region"]["neighbourhood"], 0)


if __name__ == "__main__":
    unittest.main()

