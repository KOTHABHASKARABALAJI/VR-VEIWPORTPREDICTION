"""
Unit tests for Uniform Quality Allocation Module:
- tests/test_uniform_allocator.py
Phase: Phase 5, Day 2 — Module 2
"""

import unittest
import numpy as np

from src.quality.quality_config import (
    QualityLevel,
    TileQualityConfig,
    TileQualityAllocation,
)
from src.quality.uniform_allocator import (
    UniformQualityAllocator,
    UniformAllocationResult,
    InsufficientBandwidthError,
    allocate_uniform_quality,
)


class TestUniformQualityAllocator(unittest.TestCase):

    def setUp(self):
        self.allocator = UniformQualityAllocator()

    def test_tier_costs_calculation(self):
        """Verifies the exact 24-tile simulated bitrate costs for each tier."""
        costs = self.allocator.get_all_tier_costs()
        self.assertEqual(costs[QualityLevel.Q1], 2400.0)   # 24 * 100 kbps
        self.assertEqual(costs[QualityLevel.Q2], 7200.0)   # 24 * 300 kbps
        self.assertEqual(costs[QualityLevel.Q3], 14400.0)  # 24 * 600 kbps
        self.assertEqual(costs[QualityLevel.Q4], 24000.0)  # 24 * 1000 kbps

    def test_budget_2400_selects_q1(self):
        """Budget 2,400 kbps must select Q1 (Low) and satisfy budget with 0 remaining headroom."""
        res = self.allocator.allocate_with_result(2400.0)
        self.assertTrue(res.is_feasible)
        self.assertTrue(res.is_budget_satisfied)
        self.assertEqual(res.selected_quality, QualityLevel.Q1)
        self.assertEqual(res.total_bitrate_kbps, 2400.0)
        self.assertEqual(res.total_bitrate_mbps, 2.4)
        self.assertEqual(res.remaining_budget_kbps, 0.0)
        self.assertIsNotNone(res.allocation)
        self.assertTrue(res.allocation.is_complete)
        self.assertEqual(res.allocation.total_tiles_assigned, 24)
        self.assertTrue(all(q == QualityLevel.Q1 for q in res.allocation.assignments.values()))

    def test_budget_5000_selects_q1(self):
        """Budget 5,000 kbps must select Q1 (7,200 kbps needed for Q2) with 2,600 kbps headroom."""
        res = self.allocator.allocate_with_result(5000.0)
        self.assertTrue(res.is_feasible)
        self.assertTrue(res.is_budget_satisfied)
        self.assertEqual(res.selected_quality, QualityLevel.Q1)
        self.assertEqual(res.total_bitrate_kbps, 2400.0)
        self.assertEqual(res.remaining_budget_kbps, 2600.0)
        self.assertTrue(res.allocation.is_complete)

    def test_budget_7200_selects_q2(self):
        """Budget 7,200 kbps must select Q2 (Medium) with 0 remaining headroom."""
        res = self.allocator.allocate_with_result(7200.0)
        self.assertTrue(res.is_feasible)
        self.assertTrue(res.is_budget_satisfied)
        self.assertEqual(res.selected_quality, QualityLevel.Q2)
        self.assertEqual(res.total_bitrate_kbps, 7200.0)
        self.assertEqual(res.total_bitrate_mbps, 7.2)
        self.assertEqual(res.remaining_budget_kbps, 0.0)
        self.assertTrue(all(q == QualityLevel.Q2 for q in res.allocation.assignments.values()))

    def test_budget_14400_selects_q3(self):
        """Budget 14,400 kbps must select Q3 (High) with 0 remaining headroom."""
        res = self.allocator.allocate_with_result(14400.0)
        self.assertTrue(res.is_feasible)
        self.assertTrue(res.is_budget_satisfied)
        self.assertEqual(res.selected_quality, QualityLevel.Q3)
        self.assertEqual(res.total_bitrate_kbps, 14400.0)
        self.assertEqual(res.total_bitrate_mbps, 14.4)
        self.assertEqual(res.remaining_budget_kbps, 0.0)
        self.assertTrue(all(q == QualityLevel.Q3 for q in res.allocation.assignments.values()))

    def test_budget_24000_selects_q4(self):
        """Budget 24,000 kbps must select Q4 (Ultra) with 0 remaining headroom."""
        res = self.allocator.allocate_with_result(24000.0)
        self.assertTrue(res.is_feasible)
        self.assertTrue(res.is_budget_satisfied)
        self.assertEqual(res.selected_quality, QualityLevel.Q4)
        self.assertEqual(res.total_bitrate_kbps, 24000.0)
        self.assertEqual(res.total_bitrate_mbps, 24.0)
        self.assertEqual(res.remaining_budget_kbps, 0.0)
        self.assertTrue(all(q == QualityLevel.Q4 for q in res.allocation.assignments.values()))

    def test_budget_above_24000_selects_q4(self):
        """Budget > 24,000 kbps (e.g. 30,000 kbps) caps at Q4 with surplus headroom."""
        res = self.allocator.allocate_with_result(30000.0)
        self.assertEqual(res.selected_quality, QualityLevel.Q4)
        self.assertEqual(res.total_bitrate_kbps, 24000.0)
        self.assertEqual(res.remaining_budget_kbps, 6000.0)
        self.assertTrue(res.is_budget_satisfied)

    def test_budget_below_2400_raises_insufficient_bandwidth(self):
        """Budgets below 2,400 kbps cannot satisfy uniform allocation without exceeding limits."""
        # allocate() raises InsufficientBandwidthError by default
        with self.assertRaises(InsufficientBandwidthError):
            self.allocator.allocate(2399.0)

        with self.assertRaises(InsufficientBandwidthError):
            self.allocator.allocate(1500.0)

        with self.assertRaises(InsufficientBandwidthError):
            self.allocator.allocate(0.0)

        # select_highest_feasible_tier returns None
        self.assertIsNone(self.allocator.select_highest_feasible_tier(2399.0))
        self.assertIsNone(self.allocator.select_highest_feasible_tier(1000.0))

        # allocate_with_result returns infeasible result without raising
        res = self.allocator.allocate_with_result(1500.0)
        self.assertFalse(res.is_feasible)
        self.assertFalse(res.is_budget_satisfied)
        self.assertIsNone(res.selected_quality)
        self.assertIsNone(res.allocation)
        self.assertEqual(res.total_bitrate_kbps, 0.0)
        self.assertEqual(res.remaining_budget_kbps, 1500.0)

        # Optional fallback explicitly enabled
        fallback_res = self.allocator.allocate_with_result(1500.0, allow_under_budget_fallback=True)
        self.assertEqual(fallback_res.selected_quality, QualityLevel.Q1)
        self.assertFalse(fallback_res.is_budget_satisfied)
        self.assertEqual(fallback_res.remaining_budget_kbps, -900.0)

    def test_budget_exactly_equal_to_each_tier_cost(self):
        """Verifies exact-match boundaries for all four quality tiers."""
        exact_budgets = [
            (2400.0, QualityLevel.Q1),
            (7200.0, QualityLevel.Q2),
            (14400.0, QualityLevel.Q3),
            (24000.0, QualityLevel.Q4),
        ]
        for budget, expected_tier in exact_budgets:
            alloc = self.allocator.allocate(budget)
            self.assertTrue(alloc.is_complete)
            self.assertEqual(alloc.total_bitrate_kbps, budget)
            self.assertTrue(alloc.satisfies_budget(budget))
            self.assertEqual(alloc.remaining_budget(budget), 0.0)
            self.assertTrue(all(q == expected_tier for q in alloc.assignments.values()))

    def test_invalid_negative_nan_infinite_budgets_rejected(self):
        """Rejects negative, NaN, infinite, boolean, and non-numeric budgets."""
        # Negative budget
        with self.assertRaises(ValueError):
            self.allocator.allocate(-100.0)
        with self.assertRaises(ValueError):
            self.allocator.select_highest_feasible_tier(-1.0)

        # NaN budget
        with self.assertRaises(ValueError):
            self.allocator.allocate(float("nan"))
        with self.assertRaises(ValueError):
            self.allocator.select_highest_feasible_tier(float("nan"))

        # Infinite budget
        with self.assertRaises(ValueError):
            self.allocator.allocate(float("inf"))
        with self.assertRaises(ValueError):
            self.allocator.allocate(float("-inf"))

        # Boolean budget
        with self.assertRaises(TypeError):
            self.allocator.allocate(True)

        # Non-numeric string
        with self.assertRaises(TypeError):
            self.allocator.allocate("5000")

    def test_complete_assignment_all_24_tiles(self):
        """Confirms that every tile from ID 0 to 23 is assigned a quality level."""
        for budget in [2400.0, 7200.0, 14400.0, 24000.0]:
            alloc = self.allocator.allocate(budget)
            self.assertTrue(alloc.is_complete)
            self.assertEqual(alloc.total_tiles_assigned, 24)
            self.assertEqual(alloc.missing_tiles, set())
            for t_id in range(24):
                self.assertIn(t_id, alloc.assignments)
                self.assertIsInstance(alloc.assignments[t_id], QualityLevel)

    def test_no_allocation_ever_exceeds_supplied_budget(self):
        """Sweeps diverse budget points confirming total bitrate NEVER exceeds budget."""
        test_budgets = [
            2400.0, 2500.0, 4000.0, 7199.9,
            7200.0, 8000.0, 12000.0, 14399.9,
            14400.0, 18000.0, 23999.9, 24000.0, 50000.0
        ]
        for b in test_budgets:
            alloc = self.allocator.allocate(b)
            self.assertLessEqual(
                alloc.total_bitrate_kbps,
                b,
                f"Allocation ({alloc.total_bitrate_kbps} kbps) exceeded budget ({b} kbps)!"
            )
            self.assertTrue(alloc.satisfies_budget(b))

    def test_convenience_function_and_serialization(self):
        """Tests allocate_uniform_quality convenience function and UniformAllocationResult serialization."""
        alloc = allocate_uniform_quality(7200.0)
        self.assertEqual(alloc.total_bitrate_kbps, 7200.0)
        self.assertTrue(alloc.is_complete)

        res = self.allocator.allocate_with_result(14400.0)
        res_dict = res.to_dict()
        self.assertEqual(res_dict["selected_quality"], "Q3")
        self.assertEqual(res_dict["total_bitrate_kbps"], 14400.0)
        self.assertEqual(res_dict["total_bitrate_mbps"], 14.4)
        self.assertEqual(res_dict["bandwidth_budget_kbps"], 14400.0)
        self.assertEqual(res_dict["remaining_budget_kbps"], 0.0)
        self.assertTrue(res_dict["is_budget_satisfied"])
        self.assertTrue(res_dict["is_feasible"])
        self.assertIn("allocation", res_dict)
        self.assertEqual(res_dict["allocation"]["total_tiles_assigned"], 24)


if __name__ == "__main__":
    unittest.main()

