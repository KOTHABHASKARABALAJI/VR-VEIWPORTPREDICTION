"""
Module: neighbourhood_allocator.py
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming
Phase: Phase 5, Day 2 — Module 4: Neighbourhood-Aware Tile Quality Allocation

Implements the neighbourhood-aware tile-quality allocation strategy across the canonical
4x6 (24-tile) Equirectangular Projection (ERP) video canvas (3840 x 1920 resolution).

Motivation & Guard-Band Heuristic:
----------------------------------
In 360° video streaming, viewers frequently execute small head rotations and saccades (~5° to 20°).
Tiles immediately adjacent to the predicted viewport frustum (first-ring neighbours) are at high
risk of entering the field of view before the next network adaptation cycle.
By allocating intermediate quality (e.g. Q2/Q3) to these margin tiles in advance, the streaming
system creates a spatial guard band (protective buffer) that mitigates sudden visual stalling
or jarring quality degradation during unexpected head movement.

Priority Regions:
-----------------
1. Region 1 (Predicted Viewport, V): Active tiles intersecting the predicted FoV (highest priority).
2. Region 2 (Neighbourhood Margin, M): First-ring adjacent tiles not in V (guard band priority).
3. Region 3 (Background Tiles, B): Remaining canvas tiles (baseline fallback Q1, lowest priority).

Weighting Formulation:
----------------------
Given original normalized viewport weights w_i^vp (sum = 1.0) and neighbour_weight_factor alpha in [0, 1):
    w_i^neigh = (1 - alpha) * w_i^vp                       for i in Region 1 (Viewport)
    w_j^neigh = alpha * (s_j / sum_{k in M} s_k)            for j in Region 2 (Neighbourhood)
    w_k^neigh = 0.0                                         for k in Region 3 (Background)
where s_j = sum_{i in V and adjacent to j} w_i^vp represents the adjacency overlap score.
This formulation mathematically guarantees:
    - sum_{i=0..23} w_i^neigh = 1.0
    - Exact preservation of relative viewport priority: w_a^neigh / w_b^neigh = w_a^vp / w_b^vp
    - alpha = 0.0 exactly reproduces pure Viewport-Aware Allocation (Module 3).

Simulation Scope & Disclaimers:
-------------------------------
- Bitrates (100, 300, 600, 1000 kbps) and utilities (1.00, 2.00, 2.80, 3.40) are simulated modeling assumptions.
- No claim of empirical PSNR, SSIM, VMAF, or measured video codec bitrates is made.
- This is a heuristic greedy allocation algorithm, not a proof of global mathematical optimality.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, Any, Union
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
from src.quality.viewport_allocator import (
    ViewportQualityAllocator,
    ViewportAllocationResult,
)
from src.tiling.grid_partitioner import TileGridPartitioner


DEFAULT_NEIGHBOUR_FACTOR: float = 0.25


@dataclass(frozen=True)
class NeighbourhoodAllocationResult:
    """
    Immutable structured summary of a neighbourhood-aware tile quality allocation decision.
    """
    assignments: Dict[int, QualityLevel]
    total_bitrate_kbps: float
    total_bitrate_mbps: float
    bandwidth_budget_kbps: float
    remaining_budget_kbps: float
    original_viewport_weights: Dict[int, float]
    neighbourhood_weights: Dict[int, float]
    weighted_utility_original: float
    weighted_utility_neighbourhood: float
    mean_utility: float
    tier_distribution: Dict[QualityLevel, int]
    num_upgrades_applied: int
    is_budget_satisfied: bool
    is_feasible: bool
    region_tiles: Dict[str, List[int]]
    upgrades_by_region: Dict[str, int]
    neighbour_weight_factor: float
    num_rings: int
    allocation: Optional[TileQualityAllocation]
    message: str

    def to_dict(self) -> Dict[str, Any]:
        """Serializes result into a JSON-compatible dictionary."""
        return {
            "assignments": {int(k): v.value for k, v in self.assignments.items()},
            "total_bitrate_kbps": float(self.total_bitrate_kbps),
            "total_bitrate_mbps": float(self.total_bitrate_mbps),
            "bandwidth_budget_kbps": float(self.bandwidth_budget_kbps),
            "remaining_budget_kbps": float(self.remaining_budget_kbps),
            "weighted_utility_original": float(self.weighted_utility_original),
            "weighted_utility_neighbourhood": float(self.weighted_utility_neighbourhood),
            "mean_utility": float(self.mean_utility),
            "tier_distribution": {k.value: v for k, v in self.tier_distribution.items()},
            "num_upgrades_applied": int(self.num_upgrades_applied),
            "is_budget_satisfied": bool(self.is_budget_satisfied),
            "is_feasible": bool(self.is_feasible),
            "region_tiles": {k: [int(tid) for tid in v] for k, v in self.region_tiles.items()},
            "upgrades_by_region": dict(self.upgrades_by_region),
            "neighbour_weight_factor": float(self.neighbour_weight_factor),
            "num_rings": int(self.num_rings),
            "message": str(self.message),
        }


class TileNeighbourhoodFinder:
    """
    Computes topological tile neighbourhoods on a 4x6 (24-tile) Equirectangular Projection grid.
    Handles circular 360° horizontal wrapping and non-wrapping polar boundaries.
    """

    def __init__(self, num_rows: int = 4, num_cols: int = 6) -> None:
        if isinstance(num_rows, bool) or not isinstance(num_rows, int) or num_rows <= 0:
            raise ValueError(f"num_rows must be a positive integer, got {num_rows}")
        if isinstance(num_cols, bool) or not isinstance(num_cols, int) or num_cols <= 0:
            raise ValueError(f"num_cols must be a positive integer, got {num_cols}")

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.total_tiles = num_rows * num_cols

    def validate_tile_id(self, tile_id: Any) -> int:
        """Validates that tile_id is a valid integer within [0, total_tiles - 1]."""
        if isinstance(tile_id, bool) or not isinstance(tile_id, (int, np.integer)):
            raise TypeError(f"tile_id must be an integer, got {type(tile_id).__name__}")
        t = int(tile_id)
        if not (0 <= t < self.total_tiles):
            raise ValueError(f"tile_id must be in [0, {self.total_tiles - 1}], got {tile_id}")
        return t

    def get_tile_neighbours(self, tile_id: int, include_diagonals: bool = True) -> List[int]:
        """
        Calculates first-ring neighbour tile IDs for a given tile.

        Boundary Rules:
        1. Horizontal (yaw) wraps circularly: col -1 -> col 5, col 6 -> col 0.
        2. Vertical (pitch) does NOT wrap over poles: row -1 and row 4 are invalid.
        3. Excludes the tile itself.
        4. Returns sorted unique integers in [0, 23].
        """
        tid = self.validate_tile_id(tile_id)
        r = tid // self.num_cols
        c = tid % self.num_cols

        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Top, Bottom, Left, Right
        if include_diagonals:
            offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])  # 4 Diagonals

        neighbours: Set[int] = set()
        for dr, dc in offsets:
            nr = r + dr
            # Vertical clamp: poles do not wrap
            if 0 <= nr < self.num_rows:
                # Horizontal circular wrap around ERP sphere
                nc = (c + dc) % self.num_cols
                neighbour_id = nr * self.num_cols + nc
                if neighbour_id != tid:
                    neighbours.add(neighbour_id)

        return sorted(list(neighbours))

    def get_multi_ring_neighbourhood(
        self,
        active_tiles: Union[List[int], Set[int]],
        num_rings: int = 1,
        include_diagonals: bool = True
    ) -> Tuple[List[int], Dict[int, int]]:
        """
        Identifies multi-ring neighbouring tiles surrounding an active set of tiles.

        Returns:
            Tuple: (sorted list of all margin tiles, dict mapping tile_id -> ring_index).
        """
        if isinstance(num_rings, bool) or not isinstance(num_rings, int) or num_rings < 1:
            raise ValueError(f"num_rings must be an integer >= 1, got {num_rings}")

        active_set = {self.validate_tile_id(tid) for tid in active_tiles}
        visited = set(active_set)
        current_boundary = set(active_set)
        ring_assignments: Dict[int, int] = {}

        for ring_idx in range(1, num_rings + 1):
            next_boundary: Set[int] = set()
            for tid in current_boundary:
                for n_id in self.get_tile_neighbours(tid, include_diagonals=include_diagonals):
                    if n_id not in visited:
                        next_boundary.add(n_id)
                        visited.add(n_id)
                        ring_assignments[n_id] = ring_idx
            current_boundary = next_boundary
            if not current_boundary:
                break

        margin_tiles = sorted(list(ring_assignments.keys()))
        return margin_tiles, ring_assignments


class NeighbourhoodQualityAllocator:
    """
    Deterministic greedy marginal rate-utility tile quality allocator incorporating
    spatial neighbourhood guard bands surrounding the predicted viewport.
    """

    def __init__(
        self,
        config: Optional[TileQualityConfig] = None,
        neighbour_weight_factor: float = DEFAULT_NEIGHBOUR_FACTOR,
        num_rings: int = 1,
        include_diagonals: bool = True,
    ) -> None:
        """
        Parameters:
            config: Optional TileQualityConfig. If None, uses canonical 4x6 (24-tile) configuration.
            neighbour_weight_factor: Fraction of total importance allocated to the neighbourhood (0.0 <= factor < 1.0).
            num_rings: Number of concentric neighbourhood rings (default: 1).
            include_diagonals: Whether diagonal neighbours are included (default: True, 8-connectivity).
        """
        self.config = config if config is not None else TileQualityConfig()
        self.finder = TileNeighbourhoodFinder(
            num_rows=self.config.grid_rows,
            num_cols=self.config.grid_cols
        )
        self.num_rings = self._validate_num_rings(num_rings)
        self.include_diagonals = self._validate_bool(include_diagonals, "include_diagonals")
        self.neighbour_weight_factor = self._validate_factor(neighbour_weight_factor)

        # Baseline cost across all tiles at Q1
        self.baseline_bitrate_kbps: float = (
            self.config.total_tiles * self.config.get_bitrate(QualityLevel.Q1)
        )

        # Progression mappings
        self._next_tier: Dict[QualityLevel, Optional[QualityLevel]] = {
            QualityLevel.Q1: QualityLevel.Q2,
            QualityLevel.Q2: QualityLevel.Q3,
            QualityLevel.Q3: QualityLevel.Q4,
            QualityLevel.Q4: None,
        }
        self._delta_bitrate: Dict[QualityLevel, float] = {}
        self._delta_utility: Dict[QualityLevel, float] = {}
        for lvl in (QualityLevel.Q1, QualityLevel.Q2, QualityLevel.Q3):
            nxt = self._next_tier[lvl]
            assert nxt is not None
            self._delta_bitrate[lvl] = self.config.get_bitrate(nxt) - self.config.get_bitrate(lvl)
            self._delta_utility[lvl] = self.config.get_utility(nxt) - self.config.get_utility(lvl)

    @staticmethod
    def _validate_bool(val: Any, name: str) -> bool:
        if not isinstance(val, bool):
            raise TypeError(f"{name} must be a boolean, got {type(val).__name__}")
        return val

    @staticmethod
    def _validate_num_rings(num_rings: Any) -> int:
        if isinstance(num_rings, bool) or not isinstance(num_rings, int) or num_rings < 1:
            raise ValueError(f"num_rings must be an integer >= 1, got {num_rings}")
        return int(num_rings)

    @staticmethod
    def _validate_factor(factor: Any) -> float:
        if isinstance(factor, bool) or not isinstance(factor, (int, float, np.floating, np.integer)):
            raise TypeError(f"neighbour_weight_factor must be numeric, got {type(factor).__name__}")
        f = float(factor)
        if not np.isfinite(f):
            raise ValueError(f"neighbour_weight_factor must be finite, got {f}")
        if not (0.0 <= f < 1.0):
            raise ValueError(f"neighbour_weight_factor must be strictly in [0.0, 1.0), got {f}")
        return f

    def validate_tile_weights(
        self,
        tile_weights: Dict[int, float],
        tolerance: float = 1e-4
    ) -> Dict[int, float]:
        """Validates that tile_weights contains exactly 24 valid weights summing to 1.0."""
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

        missing = set(range(self.config.total_tiles)) - set(cleaned.keys())
        if missing:
            raise ValueError(f"Missing required tile IDs in tile_weights: {sorted(list(missing))}")

        sum_w = sum(cleaned.values())
        if sum_w <= 0.0 or np.isclose(sum_w, 0.0):
            raise ValueError(f"Zero-sum weights encountered (sum={sum_w}); weights must sum to 1.0.")

        if abs(sum_w - 1.0) > tolerance:
            raise ValueError(
                f"Tile weights must sum to 1.0 within numerical tolerance {tolerance}, "
                f"got sum={sum_w:.6f} (deviation={abs(sum_w - 1.0):.6f})."
            )

        return cleaned

    def compute_neighbourhood_weights(
        self,
        viewport_weights: Dict[int, float],
        neighbour_weight_factor: Optional[float] = None,
        num_rings: Optional[int] = None,
        tolerance: float = 1e-4
    ) -> Tuple[Dict[int, float], Dict[str, List[int]]]:
        """
        Calculates normalized neighbourhood-aware tile importance weights.

        Partitioning:
        - Region 1 (Viewport, V): {i | w_i^vp > 0}
        - Region 2 (Neighbourhood Margin, M): Tiles adjacent to V that are not in V.
        - Region 3 (Background, B): All remaining tiles.

        Returns:
            Tuple of:
                1. Dict[int, float]: Normalized 24-tile importance weights (sum = 1.0).
                2. Dict[str, List[int]]: Partitioned tile IDs by region ("viewport", "neighbourhood", "background").
        """
        clean_vp = self.validate_tile_weights(viewport_weights, tolerance=tolerance)
        alpha = self.neighbour_weight_factor if neighbour_weight_factor is None else self._validate_factor(neighbour_weight_factor)
        rings = self.num_rings if num_rings is None else self._validate_num_rings(num_rings)

        # Region 1: Active viewport tiles
        viewport_tiles = sorted([t_id for t_id, w in clean_vp.items() if w > 0.0])

        # If no active tiles or factor is 0.0, return original weights
        if not viewport_tiles or alpha == 0.0:
            all_tiles = set(range(self.config.total_tiles))
            background_tiles = sorted(list(all_tiles - set(viewport_tiles)))
            region_dict = {
                "viewport": viewport_tiles,
                "neighbourhood": [],
                "background": background_tiles,
            }
            return dict(clean_vp), region_dict

        # Region 2: Concentric neighbourhood rings
        margin_tiles, ring_map = self.finder.get_multi_ring_neighbourhood(
            active_tiles=viewport_tiles,
            num_rings=rings,
            include_diagonals=self.include_diagonals
        )

        # Region 3: Background tiles
        all_tiles_set = set(range(self.config.total_tiles))
        assigned_set = set(viewport_tiles).union(set(margin_tiles))
        background_tiles = sorted(list(all_tiles_set - assigned_set))

        region_dict = {
            "viewport": viewport_tiles,
            "neighbourhood": margin_tiles,
            "background": background_tiles,
        }

        # If there are no margin tiles (e.g. viewport covers entire sphere), keep viewport weights
        if not margin_tiles:
            return dict(clean_vp), region_dict

        # Compute raw overlap scores for margin tiles based on adjacency to active viewport tiles
        raw_neighbour_scores: Dict[int, float] = {}
        for m_id in margin_tiles:
            # Score is the sum of viewport weights of all adjacent active tiles, decayed by ring index
            ring_idx = ring_map[m_id]
            decay = 1.0 / float(ring_idx)
            adjacent_vp_weight = sum(
                clean_vp[v_id]
                for v_id in self.finder.get_tile_neighbours(m_id, include_diagonals=self.include_diagonals)
                if v_id in viewport_tiles
            )
            raw_neighbour_scores[m_id] = adjacent_vp_weight * decay

        sum_raw_margin = sum(raw_neighbour_scores.values())
        if sum_raw_margin <= 0.0 or np.isclose(sum_raw_margin, 0.0):
            # Fallback to uniform distribution across margin tiles if all adjacent weights are zero
            norm_margin_scores = {m_id: 1.0 / len(margin_tiles) for m_id in margin_tiles}
        else:
            norm_margin_scores = {m_id: score / sum_raw_margin for m_id, score in raw_neighbour_scores.items()}

        # Blend weights: (1 - alpha) for Region 1, alpha for Region 2, 0.0 for Region 3
        final_weights: Dict[int, float] = {}
        for t_id in range(self.config.total_tiles):
            if t_id in viewport_tiles:
                final_weights[t_id] = (1.0 - alpha) * clean_vp[t_id]
            elif t_id in margin_tiles:
                final_weights[t_id] = alpha * norm_margin_scores[t_id]
            else:
                final_weights[t_id] = 0.0

        # Renormalize to ensure exact unit sum within machine precision
        total_w = sum(final_weights.values())
        if not np.isclose(total_w, 1.0, atol=tolerance):
            raise ValueError(f"Neighbourhood weights sum diverged: {total_w:.6f}")

        # Exact normalize
        final_weights = {t_id: w / total_w for t_id, w in final_weights.items()}
        return final_weights, region_dict

    def allocate(
        self,
        viewport_weights: Dict[int, float],
        budget_kbps: float,
        neighbour_weight_factor: Optional[float] = None,
        num_rings: Optional[int] = None,
        allow_zero_weight_upgrades: bool = True,
        tolerance: float = 1e-4,
    ) -> TileQualityAllocation:
        """
        Executes the greedy marginal-utility-per-bitrate allocation using neighbourhood-aware weights.

        Parameters:
            viewport_weights: Dict of 24 original viewport weights (sum = 1.0).
            budget_kbps: Available downlink bandwidth budget in kbps.
            neighbour_weight_factor: Optional override for neighbourhood weight factor.
            num_rings: Optional override for number of rings.
            allow_zero_weight_upgrades: Whether zero-weight background tiles can use surplus budget.
            tolerance: Tolerance for unit-sum checks.

        Returns:
            TileQualityAllocation: Validated 24-tile quality allocation.

        Raises:
            InsufficientBandwidthError: If budget < 2,400 kbps (minimum Q1 baseline).
        """
        valid_budget = self.config.validate_bandwidth_budget(budget_kbps)
        if valid_budget < self.baseline_bitrate_kbps:
            raise InsufficientBandwidthError(
                f"Bandwidth budget ({valid_budget:.1f} kbps) is below minimum required "
                f"({self.baseline_bitrate_kbps:.1f} kbps) for 24 tiles at Q1."
            )

        weights, _ = self.compute_neighbourhood_weights(
            viewport_weights=viewport_weights,
            neighbour_weight_factor=neighbour_weight_factor,
            num_rings=num_rings,
            tolerance=tolerance,
        )

        # Step 1: Assign baseline Q1 to all 24 tiles
        assignments: Dict[int, QualityLevel] = {
            t_id: QualityLevel.Q1 for t_id in range(self.config.total_tiles)
        }
        current_bitrate = float(self.baseline_bitrate_kbps)

        # Step 2-7: Greedy upgrade loop
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
                    continue  # Already at Q4

                delta_r = self._delta_bitrate[curr_tier]
                if current_bitrate + delta_r > valid_budget:
                    continue  # Budget constraint violated

                w_i = weights[t_id]
                if not allow_zero_weight_upgrades and w_i <= 0.0:
                    continue

                delta_u = self._delta_utility[curr_tier]
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
                break

            assert best_next_tier is not None
            assignments[best_tile_id] = best_next_tier
            current_bitrate += best_delta_r

        return TileQualityAllocation(assignments=assignments, config=self.config)

    def allocate_with_result(
        self,
        viewport_weights: Dict[int, float],
        budget_kbps: float,
        neighbour_weight_factor: Optional[float] = None,
        num_rings: Optional[int] = None,
        allow_zero_weight_upgrades: bool = True,
        allow_under_budget_fallback: bool = False,
        tolerance: float = 1e-4,
    ) -> NeighbourhoodAllocationResult:
        """
        Executes neighbourhood-aware allocation and returns a structured audit result.
        """
        clean_vp = self.validate_tile_weights(viewport_weights, tolerance=tolerance)
        valid_budget = self.config.validate_bandwidth_budget(budget_kbps)
        alpha = self.neighbour_weight_factor if neighbour_weight_factor is None else self._validate_factor(neighbour_weight_factor)
        rings = self.num_rings if num_rings is None else self._validate_num_rings(num_rings)

        weights, region_dict = self.compute_neighbourhood_weights(
            viewport_weights=clean_vp,
            neighbour_weight_factor=alpha,
            num_rings=rings,
            tolerance=tolerance,
        )

        if valid_budget < self.baseline_bitrate_kbps:
            if allow_under_budget_fallback:
                fallback = self.config.create_uniform_allocation(QualityLevel.Q1)
                total_b = fallback.total_bitrate_kbps
                w_orig = fallback.weighted_utility(clean_vp, tolerance=tolerance)
                w_neigh = fallback.weighted_utility(weights, tolerance=tolerance)
                return NeighbourhoodAllocationResult(
                    assignments=dict(fallback.assignments),
                    total_bitrate_kbps=total_b,
                    total_bitrate_mbps=total_b / 1000.0,
                    bandwidth_budget_kbps=valid_budget,
                    remaining_budget_kbps=valid_budget - total_b,
                    original_viewport_weights=clean_vp,
                    neighbourhood_weights=weights,
                    weighted_utility_original=w_orig,
                    weighted_utility_neighbourhood=w_neigh,
                    mean_utility=fallback.mean_utility,
                    tier_distribution=fallback.tier_distribution,
                    num_upgrades_applied=0,
                    is_budget_satisfied=False,
                    is_feasible=False,
                    region_tiles=region_dict,
                    upgrades_by_region={"viewport": 0, "neighbourhood": 0, "background": 0},
                    neighbour_weight_factor=alpha,
                    num_rings=rings,
                    allocation=fallback,
                    message=f"Budget ({valid_budget:.1f} kbps) insufficient for baseline Q1 ({self.baseline_bitrate_kbps:.1f} kbps). Fallback Q1 exceeds budget.",
                )
            else:
                return NeighbourhoodAllocationResult(
                    assignments={},
                    total_bitrate_kbps=0.0,
                    total_bitrate_mbps=0.0,
                    bandwidth_budget_kbps=valid_budget,
                    remaining_budget_kbps=valid_budget,
                    original_viewport_weights=clean_vp,
                    neighbourhood_weights=weights,
                    weighted_utility_original=0.0,
                    weighted_utility_neighbourhood=0.0,
                    mean_utility=0.0,
                    tier_distribution={lvl: 0 for lvl in QualityLevel},
                    num_upgrades_applied=0,
                    is_budget_satisfied=False,
                    is_feasible=False,
                    region_tiles=region_dict,
                    upgrades_by_region={"viewport": 0, "neighbourhood": 0, "background": 0},
                    neighbour_weight_factor=alpha,
                    num_rings=rings,
                    allocation=None,
                    message=f"Budget ({valid_budget:.1f} kbps) insufficient for baseline Q1 ({self.baseline_bitrate_kbps:.1f} kbps). No allocation made.",
                )

        allocation = self.allocate(
            viewport_weights=clean_vp,
            budget_kbps=valid_budget,
            neighbour_weight_factor=alpha,
            num_rings=rings,
            allow_zero_weight_upgrades=allow_zero_weight_upgrades,
            tolerance=tolerance,
        )

        total_b = allocation.total_bitrate_kbps
        rem_b = valid_budget - total_b

        # Calculate BOTH original and neighbourhood-aware weighted utilities
        w_orig = allocation.weighted_utility(clean_vp, tolerance=tolerance)
        w_neigh = allocation.weighted_utility(weights, tolerance=tolerance)
        m_util = allocation.mean_utility

        # Count upgrades by region
        tier_indices = {QualityLevel.Q1: 0, QualityLevel.Q2: 1, QualityLevel.Q3: 2, QualityLevel.Q4: 3}
        upgrades_by_region = {"viewport": 0, "neighbourhood": 0, "background": 0}
        total_upgrades = 0

        vp_set = set(region_dict["viewport"])
        neigh_set = set(region_dict["neighbourhood"])

        for tid, q in allocation.assignments.items():
            ups = tier_indices[q]
            total_upgrades += ups
            if tid in vp_set:
                upgrades_by_region["viewport"] += ups
            elif tid in neigh_set:
                upgrades_by_region["neighbourhood"] += ups
            else:
                upgrades_by_region["background"] += ups

        return NeighbourhoodAllocationResult(
            assignments=dict(allocation.assignments),
            total_bitrate_kbps=total_b,
            total_bitrate_mbps=total_b / 1000.0,
            bandwidth_budget_kbps=valid_budget,
            remaining_budget_kbps=rem_b,
            original_viewport_weights=clean_vp,
            neighbourhood_weights=weights,
            weighted_utility_original=w_orig,
            weighted_utility_neighbourhood=w_neigh,
            mean_utility=m_util,
            tier_distribution=allocation.tier_distribution,
            num_upgrades_applied=total_upgrades,
            is_budget_satisfied=True,
            is_feasible=True,
            region_tiles=region_dict,
            upgrades_by_region=upgrades_by_region,
            neighbour_weight_factor=alpha,
            num_rings=rings,
            allocation=allocation,
            message=(
                f"Neighbourhood allocation successful (alpha={alpha:.2f}, rings={rings}): "
                f"{total_upgrades} upgrades ({upgrades_by_region['viewport']} vp, "
                f"{upgrades_by_region['neighbourhood']} margin, {upgrades_by_region['background']} bg). "
                f"Bitrate {total_b:.1f} kbps <= budget {valid_budget:.1f} kbps. "
                f"Original FoV Utility: {w_orig:.4f}, Guard-Band Utility: {w_neigh:.4f}."
            ),
        )

    def compare_all_strategies(
        self,
        viewport_weights: Dict[int, float],
        budget_kbps: float,
        neighbour_weight_factors: Optional[List[float]] = None,
        allow_zero_weight_upgrades: bool = True,
        tolerance: float = 1e-4,
    ) -> Dict[str, Any]:
        """
        Runs an end-to-end multi-strategy benchmark comparing:
        1. Uniform Allocation (Module 2)
        2. Viewport-Aware Allocation (Module 3)
        3. Neighbourhood-Aware Allocation (Module 4) across multiple neighbourhood factors.

        Evaluates ALL allocations using BOTH original viewport weights and neighbourhood-aware weights.
        """
        clean_vp = self.validate_tile_weights(viewport_weights, tolerance=tolerance)
        valid_budget = self.config.validate_bandwidth_budget(budget_kbps)
        factors = [0.0, 0.25, 0.50] if neighbour_weight_factors is None else [self._validate_factor(f) for f in neighbour_weight_factors]

        # 1. Uniform allocation
        uniform_allocator = UniformQualityAllocator(config=self.config)
        uniform_res = uniform_allocator.allocate_with_result(valid_budget)
        uniform_w_orig = 0.0
        uniform_mean = 0.0
        uniform_dist = {lvl.value: 0 for lvl in QualityLevel}
        if uniform_res.allocation is not None:
            uniform_w_orig = uniform_res.allocation.weighted_utility(clean_vp, tolerance=tolerance)
            uniform_mean = uniform_res.allocation.mean_utility
            uniform_dist = {k.value: v for k, v in uniform_res.allocation.tier_distribution.items()}

        # 2. Viewport-aware allocation
        vp_allocator = ViewportQualityAllocator(config=self.config)
        vp_res = vp_allocator.allocate_with_result(
            tile_weights=clean_vp,
            budget_kbps=valid_budget,
            allow_zero_weight_upgrades=allow_zero_weight_upgrades,
            tolerance=tolerance,
        )

        # 3. Neighbourhood-aware allocations across factors
        neigh_results: Dict[str, Any] = {}
        for alpha in factors:
            res = self.allocate_with_result(
                viewport_weights=clean_vp,
                budget_kbps=valid_budget,
                neighbour_weight_factor=alpha,
                allow_zero_weight_upgrades=allow_zero_weight_upgrades,
                tolerance=tolerance,
            )
            key = f"alpha_{alpha:.2f}"
            neigh_results[key] = {
                "factor": float(alpha),
                "total_bitrate_kbps": float(res.total_bitrate_kbps),
                "remaining_budget_kbps": float(res.remaining_budget_kbps),
                "weighted_utility_original": float(res.weighted_utility_original),
                "weighted_utility_neighbourhood": float(res.weighted_utility_neighbourhood),
                "mean_utility": float(res.mean_utility),
                "tier_distribution": {k.value: v for k, v in res.tier_distribution.items()},
                "num_upgrades_applied": int(res.num_upgrades_applied),
                "upgrades_by_region": dict(res.upgrades_by_region),
                "is_budget_satisfied": bool(res.is_budget_satisfied),
            }

        return {
            "bandwidth_budget_kbps": float(valid_budget),
            "uniform": {
                "selected_tier": uniform_res.selected_quality.value if uniform_res.selected_quality else None,
                "total_bitrate_kbps": float(uniform_res.total_bitrate_kbps),
                "remaining_budget_kbps": float(uniform_res.remaining_budget_kbps),
                "weighted_utility_original": float(uniform_w_orig),
                "mean_utility": float(uniform_mean),
                "tier_distribution": uniform_dist,
                "is_budget_satisfied": bool(uniform_res.is_budget_satisfied),
            },
            "viewport_aware": {
                "total_bitrate_kbps": float(vp_res.total_bitrate_kbps),
                "remaining_budget_kbps": float(vp_res.remaining_budget_kbps),
                "weighted_utility_original": float(vp_res.weighted_utility),
                "mean_utility": float(vp_res.mean_utility),
                "tier_distribution": {k.value: v for k, v in vp_res.tier_distribution.items()},
                "num_upgrades_applied": int(vp_res.num_upgrades_applied),
                "is_budget_satisfied": bool(vp_res.is_budget_satisfied),
            },
            "neighbourhood_aware": neigh_results,
            "scientific_disclaimer": (
                "Simulated rate-utility assumptions. Does not represent measured codec encodings, "
                "actual PSNR, SSIM, VMAF, or hardware QoE."
            ),
        }


def allocate_neighbourhood_quality(
    viewport_weights: Dict[int, float],
    budget_kbps: float,
    neighbour_weight_factor: float = DEFAULT_NEIGHBOUR_FACTOR,
    num_rings: int = 1,
    config: Optional[TileQualityConfig] = None,
    allow_zero_weight_upgrades: bool = True,
    tolerance: float = 1e-4,
) -> TileQualityAllocation:
    """
    Convenience functional interface for neighbourhood-aware quality allocation.
    """
    allocator = NeighbourhoodQualityAllocator(
        config=config,
        neighbour_weight_factor=neighbour_weight_factor,
        num_rings=num_rings,
    )
    return allocator.allocate(
        viewport_weights=viewport_weights,
        budget_kbps=budget_kbps,
        allow_zero_weight_upgrades=allow_zero_weight_upgrades,
        tolerance=tolerance,
    )

