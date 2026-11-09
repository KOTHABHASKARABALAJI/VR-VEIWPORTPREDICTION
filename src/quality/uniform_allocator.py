"""
Module: uniform_allocator.py
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming
Phase: Phase 5, Day 2 — Module 2: Uniform Quality Allocation

Implements the uniform tile-quality allocation strategy where all 24 spatial tiles
on the Equirectangular Projection (ERP) canvas are assigned the identical quality tier,
serving as the non-adaptive baseline benchmark under simulated bandwidth budgets.

Simulation Scope & Modeling Assumptions:
-----------------------------------------
1. Uniform Allocation Strategy:
   - Assigns the same quality level q in {Q1, Q2, Q3, Q4} across all 24 tiles.
   - Total bitrate costs:
     * Q1 (Low, 100 kbps/tile):    24 * 100  =  2,400 kbps (2.4 Mbps)
     * Q2 (Medium, 300 kbps/tile): 24 * 300  =  7,200 kbps (7.2 Mbps)
     * Q3 (High, 600 kbps/tile):   24 * 600  = 14,400 kbps (14.4 Mbps)
     * Q4 (Ultra, 1000 kbps/tile): 24 * 1000 = 24,000 kbps (24.0 Mbps)
   - Selects the highest quality tier whose total cost <= bandwidth_budget_kbps.
   - If budget < 2,400 kbps, the budget cannot be satisfied by uniform allocation without
     exceeding bandwidth limits; raises InsufficientBandwidthError unless fallback is enabled.

2. Explicit Metric Disclaimers:
   - Bitrates are simulated transmission envelope assumptions, NOT measured video encodings.
   - Numerical quality utilities are illustrative abstractions (Q1=1.00, Q2=2.00, Q3=2.80, Q4=3.40),
     NOT empirical PSNR, SSIM, or VMAF measurements.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Any, Union
import numpy as np

from src.quality.quality_config import (
    QualityLevel,
    TileQualityConfig,
    TileQualityAllocation,
)


class InsufficientBandwidthError(ValueError):
    """
    Raised when the available bandwidth budget is insufficient to satisfy
    even the lowest uniform quality tier (Q1, 2,400 kbps for 24 tiles).
    """
    pass


@dataclass(frozen=True)
class UniformAllocationResult:
    """
    Structured outcome of a uniform quality allocation decision.
    """
    selected_quality: Optional[QualityLevel]
    total_bitrate_kbps: float
    total_bitrate_mbps: float
    bandwidth_budget_kbps: float
    remaining_budget_kbps: float
    is_budget_satisfied: bool
    is_feasible: bool
    allocation: Optional[TileQualityAllocation]
    message: str

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the uniform allocation result to a Python dictionary."""
        return {
            "selected_quality": self.selected_quality.value if self.selected_quality else None,
            "total_bitrate_kbps": self.total_bitrate_kbps,
            "total_bitrate_mbps": self.total_bitrate_mbps,
            "bandwidth_budget_kbps": self.bandwidth_budget_kbps,
            "remaining_budget_kbps": self.remaining_budget_kbps,
            "is_budget_satisfied": self.is_budget_satisfied,
            "is_feasible": self.is_feasible,
            "message": self.message,
            "allocation": self.allocation.to_dict() if self.allocation else None,
        }


class UniformQualityAllocator:
    """
    Allocates uniform tile quality across the 24-tile ERP canvas subject to a bandwidth budget.
    """

    def __init__(self, config: Optional[TileQualityConfig] = None):
        """
        Initializes the uniform allocator with a tile quality configuration.

        Parameters:
            config: Optional TileQualityConfig instance (defaults to canonical 4x6 grid).
        """
        self.config = config if config is not None else TileQualityConfig()

    def get_tier_cost_kbps(self, level: Union[QualityLevel, str]) -> float:
        """
        Calculates the total simulated bitrate in kbps required to deliver
        all tiles in the grid at the specified quality level.
        """
        parsed = QualityLevel.from_string(level)
        return float(self.config.total_tiles * self.config.get_bitrate(parsed))

    def get_all_tier_costs(self) -> Dict[QualityLevel, float]:
        """
        Returns a dictionary mapping each QualityLevel to its total grid bitrate cost in kbps.
        """
        return {lvl: self.get_tier_cost_kbps(lvl) for lvl in QualityLevel}

    def select_highest_feasible_tier(self, budget_kbps: float) -> Optional[QualityLevel]:
        """
        Selects the highest quality level whose total 24-tile cost does not exceed budget_kbps.

        Parameters:
            budget_kbps: Available downlink bandwidth budget in kbps.

        Returns:
            QualityLevel if a tier satisfies the budget, or None if budget < Q1 total cost.

        Raises:
            TypeError: If budget_kbps is a boolean or non-numeric.
            ValueError: If budget_kbps is negative or non-finite.
        """
        valid_budget = self.config.validate_bandwidth_budget(budget_kbps)

        # Inspect tiers in descending order of quality: Q4 -> Q3 -> Q2 -> Q1
        descending_tiers = [QualityLevel.Q4, QualityLevel.Q3, QualityLevel.Q2, QualityLevel.Q1]
        for tier in descending_tiers:
            tier_cost = self.get_tier_cost_kbps(tier)
            if tier_cost <= valid_budget:
                return tier

        return None

    def allocate(
        self,
        budget_kbps: float,
        allow_under_budget_fallback: bool = False
    ) -> TileQualityAllocation:
        """
        Assigns the highest feasible uniform quality level to all 24 tiles.

        Parameters:
            budget_kbps: Available downlink bandwidth budget in kbps.
            allow_under_budget_fallback: If True and budget is below Q1 cost, falls back
                to assigning Q1 rather than raising an exception. Defaults to False to
                strictly prevent allocations from exceeding the supplied budget.

        Returns:
            TileQualityAllocation: A complete 24-tile allocation.

        Raises:
            InsufficientBandwidthError: If budget < Q1 total cost and fallback is False.
            TypeError: If budget_kbps is invalid type.
            ValueError: If budget_kbps is negative or non-finite.
        """
        valid_budget = self.config.validate_bandwidth_budget(budget_kbps)
        selected_tier = self.select_highest_feasible_tier(valid_budget)

        if selected_tier is None:
            min_cost = self.get_tier_cost_kbps(QualityLevel.Q1)
            if not allow_under_budget_fallback:
                raise InsufficientBandwidthError(
                    f"Bandwidth budget ({valid_budget:.1f} kbps) is insufficient for uniform allocation. "
                    f"Minimum required is {min_cost:.1f} kbps (Q1 across {self.config.total_tiles} tiles). "
                    f"Allocation would exceed budget by {min_cost - valid_budget:.1f} kbps."
                )
            selected_tier = QualityLevel.Q1

        return self.config.create_uniform_allocation(selected_tier)

    def allocate_with_result(
        self,
        budget_kbps: float,
        allow_under_budget_fallback: bool = False
    ) -> UniformAllocationResult:
        """
        Performs uniform allocation and returns a structured result including metadata,
        headroom, and feasibility status.

        Parameters:
            budget_kbps: Available downlink bandwidth budget in kbps.
            allow_under_budget_fallback: If True, assigns Q1 when budget is insufficient.

        Returns:
            UniformAllocationResult: Structured outcome with all budget metrics.
        """
        valid_budget = self.config.validate_bandwidth_budget(budget_kbps)
        selected_tier = self.select_highest_feasible_tier(valid_budget)

        if selected_tier is None:
            min_cost = self.get_tier_cost_kbps(QualityLevel.Q1)
            if allow_under_budget_fallback:
                alloc = self.config.create_uniform_allocation(QualityLevel.Q1)
                total_b = alloc.total_bitrate_kbps
                return UniformAllocationResult(
                    selected_quality=QualityLevel.Q1,
                    total_bitrate_kbps=total_b,
                    total_bitrate_mbps=total_b / 1000.0,
                    bandwidth_budget_kbps=valid_budget,
                    remaining_budget_kbps=valid_budget - total_b,
                    is_budget_satisfied=False,
                    is_feasible=False,
                    allocation=alloc,
                    message=f"Budget ({valid_budget:.1f} kbps) insufficient. Fallback to Q1 ({total_b:.1f} kbps) exceeds budget.",
                )
            else:
                return UniformAllocationResult(
                    selected_quality=None,
                    total_bitrate_kbps=0.0,
                    total_bitrate_mbps=0.0,
                    bandwidth_budget_kbps=valid_budget,
                    remaining_budget_kbps=valid_budget,
                    is_budget_satisfied=False,
                    is_feasible=False,
                    allocation=None,
                    message=f"Budget ({valid_budget:.1f} kbps) insufficient for uniform Q1 ({min_cost:.1f} kbps). No allocation made.",
                )

        alloc = self.config.create_uniform_allocation(selected_tier)
        total_b = alloc.total_bitrate_kbps
        remaining = valid_budget - total_b

        return UniformAllocationResult(
            selected_quality=selected_tier,
            total_bitrate_kbps=total_b,
            total_bitrate_mbps=total_b / 1000.0,
            bandwidth_budget_kbps=valid_budget,
            remaining_budget_kbps=remaining,
            is_budget_satisfied=True,
            is_feasible=True,
            allocation=alloc,
            message=f"Uniform allocation successful: tier {selected_tier.value} selected ({total_b:.1f} kbps <= {valid_budget:.1f} kbps).",
        )


def allocate_uniform_quality(
    budget_kbps: float,
    config: Optional[TileQualityConfig] = None,
    allow_under_budget_fallback: bool = False
) -> TileQualityAllocation:
    """
    Convenience functional interface for uniform quality allocation.

    Parameters:
        budget_kbps: Downlink bandwidth budget in kbps.
        config: Optional TileQualityConfig instance.
        allow_under_budget_fallback: If True, falls back to Q1 when budget < 2,400 kbps.

    Returns:
        TileQualityAllocation: Complete 24-tile allocation.
    """
    allocator = UniformQualityAllocator(config=config)
    return allocator.allocate(budget_kbps=budget_kbps, allow_under_budget_fallback=allow_under_budget_fallback)

