"""
Tests for Module 3: Viewport-Aware Quality Allocation
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming
Phase: Phase 5, Day 2 — Module 3: Viewport-Aware Quality Allocation
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
    ViewportAllocationResult,
    allocate_viewport_quality,
)
from src.tiling.viewport_mapper import ViewportMapper


class TestViewportQualityAllocator(unittest.TestCase):
    """
    Unit test suite verifying the deterministic greedy marginal-utility-per-bitrate
    viewport-aware tile quality allocator.
    """

    def setUp(self) -> None:
        self.config = TileQualityConfig()
        self.allocator = ViewportQualityAllocator(config=self.config)

        # Baseline uniform weights (1/24 per tile)
        self.uniform_weights = {i: 1.0 / 24.0 for i in range(24)}

        # Highly localized weights: tile 7 has 0.7, tile 8 has 0.3, remaining 22 tiles have 0.0
        self.localized_weights = {i: 0.0 for i in range(24)}
        self.localized_weights[7] = 0.7
        self.localized_weights[8] = 0.3

    def test_exact_minimum_feasible_budget(self) -> None:
        """
        At exact minimum feasible budget of 2,400 kbps (24 * 100 kbps), all 24 tiles
        must remain at baseline tier Q1 (Low) with zero upgrades applied.
        """
        res = self.allocator.allocate_with_result(self.localized_weights, budget_kbps=2400.0)
        self.assertTrue(res.is_feasible)
        self.assertTrue(res.is_budget_satisfied)
        self.assertEqual(res.total_bitrate_kbps, 2400.0)
        self.assertEqual(res.remaining_budget_kbps, 0.0)
        self.assertEqual(res.num_upgrades_applied, 0)
        self.assertEqual(res.tier_distribution[QualityLevel.Q1], 24)
        for t_id, q in res.assignments.items():
            self.assertEqual(q, QualityLevel.Q1)
        self.assertAlmostEqual(res.weighted_utility, 1.00, places=4)

    def test_priority_of_high_importance_tile(self) -> None:
        """
        One highly important tile must receive priority over an otherwise equivalent
        low-importance tile under constrained bandwidth.
        """
        # Tile 5 has weight 0.8, Tile 10 has weight 0.2, all others 0.0
        weights = {i: 0.0 for i in range(24)}
        weights[5] = 0.8
        weights[10] = 0.2

        # Budget allows exactly ONE single upgrade: 2400 + 200 = 2600 kbps
        res = self.allocator.allocate_with_result(weights, budget_kbps=2600.0)
        self.assertEqual(res.num_upgrades_applied, 1)
        self.assertEqual(res.assignments[5], QualityLevel.Q2, "High-weight tile 5 must upgrade to Q2")
        self.assertEqual(res.assignments[10], QualityLevel.Q1, "Lower-weight tile 10 must remain at Q1")
        self.assertEqual(res.total_bitrate_kbps, 2600.0)

    def test_monotonicity_of_importance_priority(self) -> None:
        """
        Higher viewport importance does not reduce a tile's priority when all other factors are equal.
        A tile with weight w_a > w_b must achieve >= quality tier than tile b.
        """
        weights = {i: 0.0 for i in range(24)}
        weights[2] = 0.50
        weights[3] = 0.30
        weights[4] = 0.20

        # Test across multiple intermediate budgets
        tier_ranks = {QualityLevel.Q1: 1, QualityLevel.Q2: 2, QualityLevel.Q3: 3, QualityLevel.Q4: 4}
        for budget in [2600.0, 3000.0, 3500.0, 4000.0, 5000.0]:
            res = self.allocator.allocate(weights, budget_kbps=budget)
            rank2 = tier_ranks[res.assignments[2]]
            rank3 = tier_ranks[res.assignments[3]]
            rank4 = tier_ranks[res.assignments[4]]
            self.assertGreaterEqual(
                rank2, rank3,
                f"Tile 2 (w=0.50) must have >= tier than Tile 3 (w=0.30) at budget {budget}"
            )
            self.assertGreaterEqual(
                rank3, rank4,
                f"Tile 3 (w=0.30) must have >= tier than Tile 4 (w=0.20) at budget {budget}"
            )

    def test_uniform_weights_produce_sensible_allocation(self) -> None:
        """
        Uniform weights (1/24 per tile) must produce a balanced allocation matching
        uniform quality at exact uniform tier cost boundaries.
        """
        # Exact Q2 uniform cost: 7,200 kbps
        res_7200 = self.allocator.allocate_with_result(self.uniform_weights, budget_kbps=7200.0)
        self.assertEqual(res_7200.tier_distribution[QualityLevel.Q2], 24)
        self.assertEqual(res_7200.total_bitrate_kbps, 7200.0)
        self.assertAlmostEqual(res_7200.weighted_utility, 2.00, places=4)

        # Exact Q3 uniform cost: 14,400 kbps
        res_14400 = self.allocator.allocate_with_result(self.uniform_weights, budget_kbps=14400.0)
        self.assertEqual(res_14400.tier_distribution[QualityLevel.Q3], 24)
        self.assertEqual(res_14400.total_bitrate_kbps, 14400.0)
        self.assertAlmostEqual(res_14400.weighted_utility, 2.80, places=4)

        # Partial intermediate budget: 5,000 kbps (headroom: 2,600 kbps = 13 upgrades of 200 kbps)
        res_5000 = self.allocator.allocate_with_result(self.uniform_weights, budget_kbps=5000.0)
        self.assertEqual(res_5000.tier_distribution[QualityLevel.Q2], 13)
        self.assertEqual(res_5000.tier_distribution[QualityLevel.Q1], 11)
        self.assertEqual(res_5000.total_bitrate_kbps, 5000.0)

    def test_tight_and_generous_bandwidth_budgets(self) -> None:
        """
        Verifies behavior under tight (constrained) and generous (unconstrained) budgets.
        """
        # Tight budget: 2,700 kbps (only 300 kbps headroom beyond Q1 baseline)
        # Allows 1 upgrade of 200 kbps (Q1->Q2), with 100 kbps remaining headroom
        res_tight = self.allocator.allocate_with_result(self.localized_weights, budget_kbps=2700.0)
        self.assertEqual(res_tight.num_upgrades_applied, 1)
        self.assertEqual(res_tight.total_bitrate_kbps, 2600.0)
        self.assertEqual(res_tight.remaining_budget_kbps, 100.0)
        self.assertTrue(res_tight.is_budget_satisfied)

        # Generous budget: 24,000 kbps (enough for all 24 tiles to reach Q4)
        res_generous = self.allocator.allocate_with_result(self.localized_weights, budget_kbps=24000.0)
        self.assertEqual(res_generous.tier_distribution[QualityLevel.Q4], 24)
        self.assertEqual(res_generous.total_bitrate_kbps, 24000.0)
        self.assertEqual(res_generous.remaining_budget_kbps, 0.0)
        self.assertAlmostEqual(res_generous.weighted_utility, 3.40, places=4)

    def test_all_upgrades_respect_budget(self) -> None:
        """
        Confirms that total simulated bitrate strictly never exceeds the supplied budget
        across a sweep of 25 diverse budget values.
        """
        budgets = np.linspace(2400.0, 26000.0, 25)
        for b in budgets:
            res = self.allocator.allocate_with_result(self.localized_weights, budget_kbps=float(b))
            self.assertTrue(
                res.is_budget_satisfied,
                f"Budget violated at {b}: bitrate {res.total_bitrate_kbps} > {b}"
            )
            self.assertLessEqual(res.total_bitrate_kbps, float(b))
            self.assertGreaterEqual(res.remaining_budget_kbps, 0.0)

    def test_all_24_tiles_assigned(self) -> None:
        """
        Every single tile from ID 0 to 23 must have exactly one quality assignment.
        """
        res = self.allocator.allocate_with_result(self.localized_weights, budget_kbps=8500.0)
        self.assertEqual(len(res.assignments), 24)
        self.assertEqual(set(res.assignments.keys()), set(range(24)))
        self.assertIsNotNone(res.allocation)
        self.assertTrue(res.allocation.is_complete)
        self.assertEqual(len(res.allocation.missing_tiles), 0)

    def test_invalid_weights_rejected(self) -> None:
        """
        Rejects invalid weights: wrong count, non-numeric, boolean, negative, NaN, infinite,
        zero-sum, or weights not summing to 1.0 within tolerance.
        """
        # Non-dict
        with self.assertRaises(TypeError):
            self.allocator.allocate([1.0 / 24.0] * 24, budget_kbps=5000.0)

        # Missing tiles (only 23 tiles)
        incomplete = {i: 1.0 / 23.0 for i in range(23)}
        with self.assertRaises(ValueError):
            self.allocator.allocate(incomplete, budget_kbps=5000.0)

        # Extra tiles (25 tiles)
        extra = {i: 1.0 / 25.0 for i in range(25)}
        with self.assertRaises(ValueError):
            self.allocator.allocate(extra, budget_kbps=5000.0)

        # Negative weight
        neg = dict(self.uniform_weights)
        neg[0] = -0.1
        neg[1] = neg[1] + 0.1 + (1.0 / 24.0)
        with self.assertRaises(ValueError):
            self.allocator.allocate(neg, budget_kbps=5000.0)

        # NaN and Infinite weights
        for bad_val in (np.nan, np.inf, -np.inf):
            bad_w = dict(self.uniform_weights)
            bad_w[0] = bad_val
            with self.assertRaises(ValueError):
                self.allocator.allocate(bad_w, budget_kbps=5000.0)

        # Boolean weight
        bool_w = dict(self.uniform_weights)
        bool_w[0] = True
        with self.assertRaises(TypeError):
            self.allocator.allocate(bool_w, budget_kbps=5000.0)

        # Non-unit sum (sums to 0.8)
        sum_small = {i: 0.8 / 24.0 for i in range(24)}
        with self.assertRaises(ValueError):
            self.allocator.allocate(sum_small, budget_kbps=5000.0)

        # Non-unit sum (sums to 1.2)
        sum_large = {i: 1.2 / 24.0 for i in range(24)}
        with self.assertRaises(ValueError):
            self.allocator.allocate(sum_large, budget_kbps=5000.0)

    def test_invalid_budgets_rejected(self) -> None:
        """
        Budgets below 2,400 kbps, negative, NaN, Inf, boolean, or non-numeric must be rejected.
        """
        # Sub-2,400 kbps budget
        with self.assertRaises(InsufficientBandwidthError):
            self.allocator.allocate(self.uniform_weights, budget_kbps=2000.0)

        with self.assertRaises(InsufficientBandwidthError):
            self.allocator.allocate(self.uniform_weights, budget_kbps=0.0)

        # Negative budget
        with self.assertRaises(ValueError):
            self.allocator.allocate(self.uniform_weights, budget_kbps=-1000.0)

        # NaN and Inf budgets
        for bad_b in (np.nan, np.inf, -np.inf):
            with self.assertRaises(ValueError):
                self.allocator.allocate(self.uniform_weights, budget_kbps=bad_b)

        # Boolean and string budgets
        with self.assertRaises(TypeError):
            self.allocator.allocate(self.uniform_weights, budget_kbps=True)
        with self.assertRaises(TypeError):
            self.allocator.allocate(self.uniform_weights, budget_kbps="5000")

    def test_deterministic_tie_breaking(self) -> None:
        """
        Deterministic tie-breaking (ascending tile ID) must produce identical results
        regardless of dictionary insertion order or execution iteration.
        """
        # With uniform weights at 2,600 kbps, all 24 tiles have identical scores for Q1->Q2.
        # Exactly one tile must be upgraded: tile ID 0!
        res1 = self.allocator.allocate(self.uniform_weights, budget_kbps=2600.0)
        self.assertEqual(res1.assignments[0], QualityLevel.Q2)
        for i in range(1, 24):
            self.assertEqual(res1.assignments[i], QualityLevel.Q1)

        # Shuffle dictionary insertion order and re-run
        import random
        keys = list(range(24))
        random.seed(42)
        random.shuffle(keys)
        shuffled_weights = {k: self.uniform_weights[k] for k in keys}

        res2 = self.allocator.allocate(shuffled_weights, budget_kbps=2600.0)
        self.assertEqual(res1.assignments, res2.assignments)

    def test_zero_weight_tiles_handling(self) -> None:
        """
        Verifies safe handling of zero-weight tiles:
        1. Zero-weight tiles never preempt positive-weight tiles.
        2. When allow_zero_weight_upgrades=False, zero-weight tiles remain at Q1 even under generous budgets.
        3. When allow_zero_weight_upgrades=True, surplus budget beyond positive tiles can upgrade zero-weight tiles.
        """
        weights = {i: 0.0 for i in range(24)}
        weights[10] = 1.0  # Only tile 10 has positive weight

        # Generous budget: 10,000 kbps
        # Max upgrade cost for tile 10: from Q1 to Q4 = +900 kbps (total 2400 + 900 = 3300 kbps)
        # If allow_zero_weight_upgrades=False, remaining 23 tiles must strictly remain at Q1!
        res_no_zero = self.allocator.allocate_with_result(
            weights, budget_kbps=10000.0, allow_zero_weight_upgrades=False
        )
        self.assertEqual(res_no_zero.assignments[10], QualityLevel.Q4)
        for i in range(24):
            if i != 10:
                self.assertEqual(res_no_zero.assignments[i], QualityLevel.Q1)
        self.assertEqual(res_no_zero.total_bitrate_kbps, 3300.0)
        self.assertEqual(res_no_zero.remaining_budget_kbps, 6700.0)

        # If allow_zero_weight_upgrades=True, remaining budget upgrades zero-weight tiles in ascending order
        res_with_zero = self.allocator.allocate_with_result(
            weights, budget_kbps=10000.0, allow_zero_weight_upgrades=True
        )
        self.assertEqual(res_with_zero.assignments[10], QualityLevel.Q4)
        # Some zero-weight tiles received upgrades
        self.assertGreater(res_with_zero.total_bitrate_kbps, 3300.0)
        self.assertLessEqual(res_with_zero.total_bitrate_kbps, 10000.0)

    def test_weighted_utility_calculation(self) -> None:
        """
        Mathematically verifies the weighted utility calculation:
            U_weighted = sum_{i} w_i * U(q_i)
        """
        weights = {i: 0.0 for i in range(24)}
        weights[0] = 0.60
        weights[1] = 0.40

        # Run allocation with budget 3,300 kbps (headroom: 900 kbps)
        # Upgrades:
        # Tile 0 (w=0.6): Q1->Q2 (+200, score 0.00300)
        # Tile 1 (w=0.4): Q1->Q2 (+200, score 0.00200)
        # Tile 0 (w=0.6): Q2->Q3 (+300, score 0.00160)
        # Tile 1 (w=0.4): Q2->Q3 (+300, score 0.00107) -> cannot fit (+300 > remaining 200)
        # Tile 0 or 1 cannot fit further. Headroom remaining: 900 - 700 = 200 kbps.
        # Tile 1 cannot upgrade Q2->Q3 (+300 kbps).
        res = self.allocator.allocate_with_result(weights, budget_kbps=3300.0, allow_zero_weight_upgrades=False)
        self.assertEqual(res.assignments[0], QualityLevel.Q3)  # Utility = 2.80
        self.assertEqual(res.assignments[1], QualityLevel.Q2)  # Utility = 2.00

        # Expected weighted utility = 0.60 * 2.80 + 0.40 * 2.00 = 1.68 + 0.80 = 2.48
        expected_u = 0.60 * 2.80 + 0.40 * 2.00
        self.assertAlmostEqual(res.weighted_utility, expected_u, places=5)

    def test_comparison_with_uniform_allocation(self) -> None:
        """
        Runs a direct side-by-side comparison between Uniform Allocation (Module 2)
        and Viewport-Aware Allocation (Module 3).
        At typical video budgets (e.g. 7,200 kbps), viewport-aware allocation must
        deliver substantially higher weighted utility within the user's line of sight.
        """
        comp = self.allocator.compare_with_uniform(
            self.localized_weights, budget_kbps=7200.0, allow_zero_weight_upgrades=True
        )
        self.assertIn("uniform", comp)
        self.assertIn("viewport_aware", comp)
        self.assertIn("metrics_delta", comp)

        # Uniform at 7,200 kbps gives Q2 (utility 2.00)
        self.assertAlmostEqual(comp["uniform"]["weighted_utility"], 2.00, places=4)

        # Viewport-aware at 7,200 kbps upgrades localized tiles (7 and 8) to Q4 (utility 3.40)
        self.assertGreater(comp["viewport_aware"]["weighted_utility"], 2.00)
        self.assertAlmostEqual(comp["viewport_aware"]["weighted_utility"], 3.40, places=4)

        # Weighted utility delta is strictly positive (+1.40)
        self.assertAlmostEqual(comp["metrics_delta"]["weighted_utility_diff"], 1.40, places=4)

        # Both methods satisfy the 7,200 kbps budget
        self.assertLessEqual(comp["uniform"]["total_bitrate_kbps"], 7200.0)
        self.assertLessEqual(comp["viewport_aware"]["total_bitrate_kbps"], 7200.0)

    def test_integration_with_day1_viewport_mapper(self) -> None:
        """
        End-to-end integration test using ViewportMapper to generate realistic
        ray-intersection tile weights from spherical coordinates (yaw=0°, pitch=0°).
        """
        mapper = ViewportMapper(hfov_deg=100.0, vfov_deg=100.0, ray_samples_per_axis=21)
        intersection = mapper.get_viewport_intersection(yaw_deg=0.0, pitch_deg=0.0)

        # Expand sparse active-tile weights to canonical 24-tile grid
        full_weights = ViewportQualityAllocator.expand_weights_to_grid(intersection.tile_weights)
        self.assertEqual(len(full_weights), 24)
        self.assertAlmostEqual(sum(full_weights.values()), 1.0, places=4)

        # Allocate with moderate budget: 8,000 kbps
        res = self.allocator.allocate_with_result(full_weights, budget_kbps=8000.0)
        self.assertTrue(res.is_budget_satisfied)
        self.assertLessEqual(res.total_bitrate_kbps, 8000.0)

        # Active tiles should have higher average quality than non-active tiles
        active_qualities = [res.assignments[t] for t in intersection.active_tiles]
        non_active_tiles = [t for t in range(24) if t not in intersection.active_tiles]
        non_active_qualities = [res.assignments[t] for t in non_active_tiles]

        tier_vals = {QualityLevel.Q1: 1, QualityLevel.Q2: 2, QualityLevel.Q3: 3, QualityLevel.Q4: 4}
        mean_active = np.mean([tier_vals[q] for q in active_qualities])
        mean_non_active = np.mean([tier_vals[q] for q in non_active_qualities])

        self.assertGreater(
            mean_active, mean_non_active,
            "Tiles within the predicted viewport must achieve higher quality on average than background tiles"
        )


if __name__ == "__main__":
    unittest.main()

