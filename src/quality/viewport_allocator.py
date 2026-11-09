"""
Module: viewport_allocator.py
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming
Phase: Phase 5, Day 2 — Module 3: Viewport-Aware Quality Allocation

Implements the deterministic greedy marginal-utility-per-bitrate tile quality allocator
across the canonical 4x6 (24-tile) Equirectangular Projection (ERP) video canvas.

Allocation Strategy:
--------------------
1. Assign baseline quality Q1 (100 kbps) to all 24 spatial tiles (IDs 0..23).
   Initial baseline cost = 24 * 100 kbps = 2,400 kbps (2.4 Mbps).
2. For each tile i in 0..23, evaluate the marginal score of upgrading from its current tier
   q_curr to the next higher tier q_next:
       score_i = (w_i * (U(q_next) - U(q_curr))) / (R(q_next) - R(q_curr))
   where:
       w_i is the normalized viewport importance weight of tile i (sum_{i} w_i = 1.0).
       U(q) is the simulated numerical quality utility from TileQualityConfig.
       R(q) is the simulated per-tile bitrate in kbps from TileQualityConfig.
3. In each iteration, greedily select the feasible upgrade (where total_bitrate + delta_R <= budget)
   with the maximum marginal score.
4. Deterministic tie-breaking: when multiple candidate upgrades share the identical score
   (within 1e-12 numerical precision), ties are broken deterministically by ascending tile ID.
5. Repeat iteratively until no feasible upgrade can fit within the remaining bandwidth budget,
   or until all tiles reach the maximum tier Q4.
6. Return a complete, validated 24-tile TileQualityAllocation and ViewportAllocationResult.

Simulation Scope & Disclaimers:
-------------------------------
- Bitrates (100, 300, 600, 1000 kbps) are simulated transmission envelope assumptions.
- Numerical utilities (1.00, 2.00, 2.80, 3.40) are illustrative concave diminishing-returns abstractions.
- No claim is made of empirical PSNR, SSIM, VMAF, or measured video codec bitrates.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Any, Union, List, Tuple
import numpy as np

from src.quality.quality_config import (
    QualityLevel,
    TileQualityConfig,
    TileQualityAllocation,
)
from src.quality.uniform_allocator import (
    InsufficientBandwidthError,
    UniformQualityAllocator,
    UniformAllocationResult,
)


@dataclass(frozen=True)
class ViewportAllocationResult:
    """
    Immutable structured summary of a viewport-aware tile quality allocation decision.
    """
    assignments: Dict[int, QualityLevel]
    total_bitrate_kbps: float
    total_bitrate_mbps: float
    bandwidth_budget_kbps: float
    remaining_budget_kbps: float
    weighted_utility: float
    mean_utility: float
    tier_distribution: Dict[QualityLevel, int]
    is_budget_satisfied: bool
    is_feasible: bool
    tile_weights: Dict[int, float]
    allocation: Optional[TileQualityAllocation]
    num_upgrades_applied: int
    message: str

    def to_dict(self) -> Dict[str, Any]:
        """Serializes result into a JSON-compatible dictionary."""
        return {
            "assignments": {int(k): v.value for k, v in self.assignments.items()},
            "total_bitrate_kbps": float(self.total_bitrate_kbps),
            "total_bitrate_mbps": float(self.total_bitrate_mbps),
            "bandwidth_budget_kbps": float(self.bandwidth_budget_kbps),
            "remaining_budget_kbps": float(self.remaining_budget_kbps),
            "weighted_utility": float(self.weighted_utility),
            "mean_utility": float(self.mean_utility),
            "tier_distribution": {k.value: v for k, v in self.tier_distribution.items()},
            "is_budget_satisfied": bool(self.is_budget_satisfied),
            "is_feasible": bool(self.is_feasible),
            "num_upgrades_applied": int(self.num_upgrades_applied),
            "message": str(self.message),
        }


class ViewportQualityAllocator:
    """
    Deterministic greedy marginal-utility-per-bitrate tile quality allocator.

    Optimizes per-tile quality tier assignments to maximize viewport-weighted visual utility
    under a hard downlink bandwidth constraint.
    """

    # Tier progression order
    TIER_ORDER: Tuple[QualityLevel, ...] = (
        QualityLevel.Q1,
        QualityLevel.Q2,
        QualityLevel.Q3,
        QualityLevel.Q4,
    )

    def __init__(self, config: Optional[TileQualityConfig] = None) -> None:
        """
        Initializes the viewport-aware allocator.

        Parameters:
            config: Optional TileQualityConfig instance. If None, uses canonical 4x6 (24-tile) configuration.
        """
        self.config = config if config is not None else TileQualityConfig()

        # Build progression mappings
        self._next_tier: Dict[QualityLevel, Optional[QualityLevel]] = {
            QualityLevel.Q1: QualityLevel.Q2,
            QualityLevel.Q2: QualityLevel.Q3,
            QualityLevel.Q3: QualityLevel.Q4,
            QualityLevel.Q4: None,
        }

        # Cache step differences for efficiency
        self._delta_bitrate: Dict[QualityLevel, float] = {}
        self._delta_utility: Dict[QualityLevel, float] = {}
        for current_tier in (QualityLevel.Q1, QualityLevel.Q2, QualityLevel.Q3):
            next_t = self._next_tier[current_tier]
            assert next_t is not None
            self._delta_bitrate[current_tier] = (
                self.config.get_bitrate(next_t) - self.config.get_bitrate(current_tier)
            )
            self._delta_utility[current_tier] = (
                self.config.get_utility(next_t) - self.config.get_utility(current_tier)
            )

        # Baseline cost across all tiles at Q1
        self.baseline_bitrate_kbps: float = (
            self.config.total_tiles * self.config.get_bitrate(QualityLevel.Q1)
        )

    def validate_tile_weights(
        self,
        tile_weights: Dict[int, float],
        tolerance: float = 1e-4
    ) -> Dict[int, float]:
        """
        Validates that tile_weights contains exactly 24 valid, non-negative, finite numeric weights
        summing to 1.0 within numerical tolerance.

        Parameters:
            tile_weights: Dict mapping tile_id (int) -> importance weight.
            tolerance: Maximum allowable deviation from 1.0 (default: 1e-4).

        Returns:
            Dict[int, float]: Cleaned dictionary with integer keys and float values for all 24 tiles.

        Raises:
            TypeError: If tile_weights is not a dict, or if keys/values are invalid types (e.g. bool).
            ValueError: If key count != 24, tile IDs out of range, weights negative/NaN/Inf,
                        or weight sum diverges from 1.0.
        """
        if not isinstance(tile_weights, dict):
            raise TypeError(f"tile_weights must be a dict, got {type(tile_weights).__name__}")

        if len(tile_weights) != self.config.total_tiles:
            raise ValueError(
                f"tile_weights must contain exactly {self.config.total_tiles} tile entries (IDs 0..{self.config.total_tiles - 1}), "
                f"got {len(tile_weights)} entries."
            )

        cleaned: Dict[int, float] = {}
        for raw_id, raw_w in tile_weights.items():
            t_id = self.config.validate_tile_id(raw_id)

            if isinstance(raw_w, bool) or not isinstance(raw_w, (int, float, np.floating, np.integer)):
                raise TypeError(f"Weight for tile {t_id} must be numeric, got {type(raw_w).__name__}")

            w = float(raw_w)
            if not np.isfinite(w):
                raise ValueError(f"Weight for tile {t_id} must be finite, got {w}")
            if w < 0.0:
                raise ValueError(f"Weight for tile {t_id} cannot be negative, got {w}")

            cleaned[t_id] = w

        # Ensure all expected tile IDs from 0 to total_tiles-1 are present
        missing = set(range(self.config.total_tiles)) - set(cleaned.keys())
        if missing:
            raise ValueError(f"Missing required tile IDs in tile_weights: {sorted(list(missing))}")

        sum_w = sum(cleaned.values())
        if sum_w <= 0.0 or np.isclose(sum_w, 0.0):
            raise ValueError(f"Zero-sum weights encountered (sum={sum_w}); weights must sum to 1.0.")

        if abs(sum_w - 1.0) > tolerance:
            raise ValueError(
                f"Tile weights must sum to 1.0 within numerical tolerance {tolerance}, "
                f"got sum={sum_w:.6f} (absolute deviation={abs(sum_w - 1.0):.6f})."
            )

        return cleaned

    @classmethod
    def expand_weights_to_grid(
        cls,
        sparse_weights: Dict[int, float],
        total_tiles: int = 24,
        tolerance: float = 1e-4
    ) -> Dict[int, float]:
        """
        Convenience helper converting sparse active-tile weights (such as ViewportIntersection.tile_weights)
        into a complete 24-tile weight dictionary by assigning 0.0 to all non-active tiles.

        Parameters:
            sparse_weights: Dict mapping active tile IDs to their non-negative weights.
            total_tiles: Total number of canvas tiles (default: 24).
            tolerance: Allowable deviation of sparse weight sum from 1.0.

        Returns:
            Dict[int, float]: Complete dictionary with keys 0..total_tiles-1 summing to 1.0.
        """
        if not isinstance(sparse_weights, dict):
            raise TypeError(f"sparse_weights must be a dict, got {type(sparse_weights).__name__}")

        sum_w = sum(float(w) for w in sparse_weights.values() if not isinstance(w, bool))
        if abs(sum_w - 1.0) > tolerance:
            raise ValueError(
                f"Sparse weights must sum to 1.0 within tolerance {tolerance}, got {sum_w:.6f}"
            )

        full_weights: Dict[int, float] = {}
        for t_id in range(total_tiles):
            full_weights[t_id] = float(sparse_weights.get(t_id, 0.0))

        return full_weights

    def allocate(
        self,
        tile_weights: Dict[int, float],
        budget_kbps: float,
        allow_zero_weight_upgrades: bool = True,
        tolerance: float = 1e-4,
    ) -> TileQualityAllocation:
        """
        Executes the greedy marginal-utility-per-bitrate allocation algorithm.

        Parameters:
            tile_weights: Dict mapping each tile ID (0..23) to its viewport importance weight.
            budget_kbps: Client downlink bandwidth budget in kbps.
            allow_zero_weight_upgrades: If True (default), tiles with weight 0.0 may be upgraded
                using remaining bandwidth after all positive-weight upgrades are exhausted.
                If False, tiles with weight 0.0 strictly remain at Q1.
            tolerance: Tolerance for unit-sum weight validation.

        Returns:
            TileQualityAllocation: Complete 24-tile quality allocation.

        Raises:
            InsufficientBandwidthError: If budget_kbps < 2,400 kbps (minimum Q1 baseline cost).
            TypeError / ValueError: On invalid weights or invalid budget parameters.
        """
        valid_weights = self.validate_tile_weights(tile_weights, tolerance=tolerance)
        valid_budget = self.config.validate_bandwidth_budget(budget_kbps)

        if valid_budget < self.baseline_bitrate_kbps:
            raise InsufficientBandwidthError(
                f"Bandwidth budget ({valid_budget:.1f} kbps) is below the minimum required "
                f"({self.baseline_bitrate_kbps:.1f} kbps) to transmit all {self.config.total_tiles} "
                f"tiles at the lowest quality level (Q1, {self.config.get_bitrate(QualityLevel.Q1):.1f} kbps)."
            )

        # Step 1: Assign baseline Q1 to all tiles
        assignments: Dict[int, QualityLevel] = {
            t_id: QualityLevel.Q1 for t_id in range(self.config.total_tiles)
        }
        current_bitrate = float(self.baseline_bitrate_kbps)

        # Step 3-6: Iterative greedy upgrade loop
        while True:
            best_upgrade_found = False
            best_score = -1.0
            best_tile_id = -1
            best_delta_r = 0.0
            best_next_tier: Optional[QualityLevel] = None

            for t_id in range(self.config.total_tiles):
                curr_tier = assignments[t_id]
                next_tier = self._next_tier[curr_tier]
                if next_tier is None:
                    continue  # Already at Q4 (Ultra), cannot upgrade further

                delta_r = self._delta_bitrate[curr_tier]
                if current_bitrate + delta_r > valid_budget:
                    continue  # Feasibility constraint violated

                w_i = valid_weights[t_id]
                if not allow_zero_weight_upgrades and w_i <= 0.0:
                    continue  # Zero-weight upgrades disabled

                delta_u = self._delta_utility[curr_tier]
                # Marginal score: tile_weight * utility_gain / additional_bitrate
                score = w_i * (delta_u / delta_r)

                if not best_upgrade_found:
                    best_score = score
                    best_tile_id = t_id
                    best_delta_r = delta_r
                    best_next_tier = next_tier
                    best_upgrade_found = True
                else:
                    diff = score - best_score
                    if diff > 1e-12:
                        # Strictly higher marginal score
                        best_score = score
                        best_tile_id = t_id
                        best_delta_r = delta_r
                        best_next_tier = next_tier
                    elif abs(diff) <= 1e-12:
                        # Deterministic tie-breaking: lower tile ID wins
                        if t_id < best_tile_id:
                            best_score = score
                            best_tile_id = t_id
                            best_delta_r = delta_r
                            best_next_tier = next_tier

            if not best_upgrade_found:
                # No remaining upgrades fit within the bandwidth budget
                break

            # Apply the selected upgrade
            assert best_next_tier is not None
            assignments[best_tile_id] = best_next_tier
            current_bitrate += best_delta_r

        return TileQualityAllocation(assignments=assignments, config=self.config)

    def allocate_with_result(
        self,
        tile_weights: Dict[int, float],
        budget_kbps: float,
        allow_zero_weight_upgrades: bool = True,
        allow_under_budget_fallback: bool = False,
        tolerance: float = 1e-4,
    ) -> ViewportAllocationResult:
        """
        Executes viewport-aware allocation and returns a structured audit result.

        Parameters:
            tile_weights: Dict mapping tile_id (0..23) to importance weight.
            budget_kbps: Downlink bandwidth budget in kbps.
            allow_zero_weight_upgrades: Whether zero-weight tiles can use surplus budget.
            allow_under_budget_fallback: If True, returns fallback allocation when budget < 2,400 kbps
                instead of raising InsufficientBandwidthError.
            tolerance: Allowable deviation of weight sum from 1.0.

        Returns:
            ViewportAllocationResult: Detailed audit dataclass.
        """
        valid_weights = self.validate_tile_weights(tile_weights, tolerance=tolerance)
        valid_budget = self.config.validate_bandwidth_budget(budget_kbps)

        if valid_budget < self.baseline_bitrate_kbps:
            if allow_under_budget_fallback:
                fallback_alloc = self.config.create_uniform_allocation(QualityLevel.Q1)
                total_b = fallback_alloc.total_bitrate_kbps
                w_util = fallback_alloc.weighted_utility(valid_weights, tolerance=tolerance)
                return ViewportAllocationResult(
                    assignments=dict(fallback_alloc.assignments),
                    total_bitrate_kbps=total_b,
                    total_bitrate_mbps=total_b / 1000.0,
                    bandwidth_budget_kbps=valid_budget,
                    remaining_budget_kbps=valid_budget - total_b,
                    weighted_utility=w_util,
                    mean_utility=fallback_alloc.mean_utility,
                    tier_distribution=fallback_alloc.tier_distribution,
                    is_budget_satisfied=False,
                    is_feasible=False,
                    tile_weights=valid_weights,
                    allocation=fallback_alloc,
                    num_upgrades_applied=0,
                    message=f"Budget ({valid_budget:.1f} kbps) insufficient for Q1 baseline ({self.baseline_bitrate_kbps:.1f} kbps). Fallback Q1 exceeds budget.",
                )
            else:
                return ViewportAllocationResult(
                    assignments={},
                    total_bitrate_kbps=0.0,
                    total_bitrate_mbps=0.0,
                    bandwidth_budget_kbps=valid_budget,
                    remaining_budget_kbps=valid_budget,
                    weighted_utility=0.0,
                    mean_utility=0.0,
                    tier_distribution={lvl: 0 for lvl in QualityLevel},
                    is_budget_satisfied=False,
                    is_feasible=False,
                    tile_weights=valid_weights,
                    allocation=None,
                    num_upgrades_applied=0,
                    message=f"Budget ({valid_budget:.1f} kbps) insufficient for Q1 baseline ({self.baseline_bitrate_kbps:.1f} kbps). No allocation made.",
                )

        allocation = self.allocate(
            tile_weights=valid_weights,
            budget_kbps=valid_budget,
            allow_zero_weight_upgrades=allow_zero_weight_upgrades,
            tolerance=tolerance,
        )

        total_b = allocation.total_bitrate_kbps
        remaining_b = valid_budget - total_b
        w_util = allocation.weighted_utility(valid_weights, tolerance=tolerance)
        m_util = allocation.mean_utility
        t_dist = allocation.tier_distribution

        # Count total tier upgrades applied beyond Q1 baseline
        tier_indices = {QualityLevel.Q1: 0, QualityLevel.Q2: 1, QualityLevel.Q3: 2, QualityLevel.Q4: 3}
        total_upgrades = sum(tier_indices[lvl] for lvl in allocation.assignments.values())

        return ViewportAllocationResult(
            assignments=dict(allocation.assignments),
            total_bitrate_kbps=total_b,
            total_bitrate_mbps=total_b / 1000.0,
            bandwidth_budget_kbps=valid_budget,
            remaining_budget_kbps=remaining_b,
            weighted_utility=w_util,
            mean_utility=m_util,
            tier_distribution=t_dist,
            is_budget_satisfied=True,
            is_feasible=True,
            tile_weights=valid_weights,
            allocation=allocation,
            num_upgrades_applied=total_upgrades,
            message=(
                f"Viewport-aware allocation successful: {total_upgrades} upgrades applied. "
                f"Bitrate {total_b:.1f} kbps <= budget {valid_budget:.1f} kbps. "
                f"Weighted utility: {w_util:.4f}."
            ),
        )

    def compare_with_uniform(
        self,
        tile_weights: Dict[int, float],
        budget_kbps: float,
        allow_zero_weight_upgrades: bool = True,
        tolerance: float = 1e-4,
    ) -> Dict[str, Any]:
        """
        Runs a direct side-by-side benchmark comparison between Uniform Allocation (Module 2)
        and Viewport-Aware Allocation (Module 3) under the identical bandwidth budget and tile weights.

        Parameters:
            tile_weights: Dict mapping tile_id (0..23) to importance weight.
            budget_kbps: Downlink bandwidth budget in kbps.
            allow_zero_weight_upgrades: Whether zero-weight tiles can use surplus budget in viewport allocator.
            tolerance: Weight tolerance.

        Returns:
            Dict containing detailed comparison metrics and human-readable comparative summary.
        """
        valid_weights = self.validate_tile_weights(tile_weights, tolerance=tolerance)
        valid_budget = self.config.validate_bandwidth_budget(budget_kbps)

        # 1. Uniform allocation
        uniform_allocator = UniformQualityAllocator(config=self.config)
        uniform_res: UniformAllocationResult = uniform_allocator.allocate_with_result(valid_budget)

        # 2. Viewport-aware allocation
        vp_res: ViewportAllocationResult = self.allocate_with_result(
            tile_weights=valid_weights,
            budget_kbps=valid_budget,
            allow_zero_weight_upgrades=allow_zero_weight_upgrades,
            tolerance=tolerance,
        )

        # Calculate weighted utility for uniform allocation
        uniform_weighted_util = 0.0
        uniform_mean_util = 0.0
        uniform_tier_dist = {lvl.value: 0 for lvl in QualityLevel}
        if uniform_res.allocation is not None:
            uniform_weighted_util = uniform_res.allocation.weighted_utility(valid_weights, tolerance=tolerance)
            uniform_mean_util = uniform_res.allocation.mean_utility
            uniform_tier_dist = {k.value: v for k, v in uniform_res.allocation.tier_distribution.items()}

        comparison = {
            "bandwidth_budget_kbps": float(valid_budget),
            "uniform": {
                "selected_tier": uniform_res.selected_quality.value if uniform_res.selected_quality else None,
                "total_bitrate_kbps": float(uniform_res.total_bitrate_kbps),
                "remaining_budget_kbps": float(uniform_res.remaining_budget_kbps),
                "weighted_utility": float(uniform_weighted_util),
                "mean_utility": float(uniform_mean_util),
                "tier_distribution": uniform_tier_dist,
                "is_feasible": bool(uniform_res.is_feasible),
            },
            "viewport_aware": {
                "total_bitrate_kbps": float(vp_res.total_bitrate_kbps),
                "remaining_budget_kbps": float(vp_res.remaining_budget_kbps),
                "weighted_utility": float(vp_res.weighted_utility),
                "mean_utility": float(vp_res.mean_utility),
                "tier_distribution": {k.value: v for k, v in vp_res.tier_distribution.items()},
                "num_upgrades_applied": int(vp_res.num_upgrades_applied),
                "is_feasible": bool(vp_res.is_feasible),
            },
            "metrics_delta": {
                "weighted_utility_diff": float(vp_res.weighted_utility - uniform_weighted_util),
                "mean_utility_diff": float(vp_res.mean_utility - uniform_mean_util),
                "bitrate_diff_kbps": float(vp_res.total_bitrate_kbps - uniform_res.total_bitrate_kbps),
            },
            "scientific_disclaimer": (
                "Bitrates and utilities are simulated modeling assumptions. "
                "They do not represent measured video codec encodings, PSNR, SSIM, or real-world QoE."
            ),
        }
        return comparison


def allocate_viewport_quality(
    tile_weights: Dict[int, float],
    budget_kbps: float,
    config: Optional[TileQualityConfig] = None,
    allow_zero_weight_upgrades: bool = True,
    tolerance: float = 1e-4,
) -> TileQualityAllocation:
    """
    Convenience functional interface for viewport-aware quality allocation.

    Parameters:
        tile_weights: Dict mapping tile IDs (0..23) to importance weights summing to 1.0.
        budget_kbps: Downlink bandwidth budget in kbps.
        config: Optional TileQualityConfig instance.
        allow_zero_weight_upgrades: Whether zero-weight tiles can be upgraded with surplus budget.
        tolerance: Numerical tolerance for weight sum.

    Returns:
        TileQualityAllocation: Complete 24-tile allocation.
    """
    allocator = ViewportQualityAllocator(config=config)
    return allocator.allocate(
        tile_weights=tile_weights,
        budget_kbps=budget_kbps,
        allow_zero_weight_upgrades=allow_zero_weight_upgrades,
        tolerance=tolerance,
    )

