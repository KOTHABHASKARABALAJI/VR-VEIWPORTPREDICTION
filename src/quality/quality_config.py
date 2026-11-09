"""
Module: quality_config.py
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming
Phase: Phase 5, Day 2 — Module 1: Tile Quality and Bitrate Configuration

Defines the foundational tile-quality tiers, simulated bitrate assignments,
numerical quality utility abstractions, and per-tile allocation structures
for a 24-tile (4 rows x 6 columns) Equirectangular video streaming pipeline.

Simulation Scope & Modeling Assumptions:
-----------------------------------------
1. Simulated Bitrate Tiers:
   - Q1 / Low:    100 kbps per tile
   - Q2 / Medium: 300 kbps per tile
   - Q3 / High:   600 kbps per tile
   - Q4 / Ultra: 1000 kbps per tile
   These values represent simulated rate-allocation assumptions under adaptive streaming
   and are NOT measured video encoding bitrates (no actual multi-rate transcode is created).

2. Numerical Quality Utility:
   - Default utility values: Q1 = 1.00, Q2 = 2.00, Q3 = 2.80, Q4 = 3.40.
   - Simulation Assumption: These values are illustrative, configurable simulation
     assumptions exhibiting diminishing marginal increments (+1.00, +0.80, +0.60).
   - Scientific Boundary: The exact numerical values are NOT empirically calibrated by
     the Weber-Fechner law or ITU-T Rec. P.1203. They represent a conceptual, heuristic
     model of diminishing returns.
   - Explicit Distinction: These values are unitless simulation abstractions.
     They do NOT represent empirical PSNR, SSIM, or VMAF measurements.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Set, Any, Optional, Union
import numpy as np


class QualityLevel(str, Enum):
    """
    Enumeration of the four standardized tile quality tiers.
    """
    Q1 = "Q1"  # Low quality
    Q2 = "Q2"  # Medium quality
    Q3 = "Q3"  # High quality
    Q4 = "Q4"  # Ultra quality

    @classmethod
    def from_string(cls, value: Union[str, QualityLevel]) -> QualityLevel:
        """
        Parses a string into a standardized QualityLevel enum member.
        Supports aliases such as 'low', 'q1_low', 'medium', 'high', 'ultra'.

        Raises:
            TypeError: If value is not a string or QualityLevel.
            ValueError: If string cannot be mapped to a known tier.
        """
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError(f"QualityLevel must be a string or QualityLevel enum, got {type(value).__name__}")

        normalized = value.strip().upper()
        alias_map = {
            "Q1": cls.Q1,
            "LOW": cls.Q1,
            "Q1_LOW": cls.Q1,
            "Q2": cls.Q2,
            "MEDIUM": cls.Q2,
            "MED": cls.Q2,
            "Q2_MEDIUM": cls.Q2,
            "Q3": cls.Q3,
            "HIGH": cls.Q3,
            "Q3_HIGH": cls.Q3,
            "Q4": cls.Q4,
            "ULTRA": cls.Q4,
            "Q4_ULTRA": cls.Q4,
        }

        if normalized in alias_map:
            return alias_map[normalized]

        valid_keys = sorted(list(cls.__members__.keys()))
        raise ValueError(
            f"Invalid quality level '{value}'. Valid options are: {valid_keys} or aliases (Low, Medium, High, Ultra)."
        )

    @property
    def display_name(self) -> str:
        """Human-readable display name for the tier."""
        names = {
            QualityLevel.Q1: "Q1 (Low)",
            QualityLevel.Q2: "Q2 (Medium)",
            QualityLevel.Q3: "Q3 (High)",
            QualityLevel.Q4: "Q4 (Ultra)",
        }
        return names[self]


@dataclass(frozen=True)
class QualityTierSpec:
    """
    Specification of a single simulated tile quality tier.
    """
    level: QualityLevel
    name: str
    bitrate_kbps: float
    utility: float
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.level, QualityLevel):
            raise TypeError(f"level must be a QualityLevel enum, got {type(self.level).__name__}")
        if isinstance(self.bitrate_kbps, bool) or not isinstance(self.bitrate_kbps, (int, float)):
            raise TypeError(f"bitrate_kbps must be a numeric value, got {type(self.bitrate_kbps).__name__}")
        if self.bitrate_kbps <= 0.0 or not np.isfinite(self.bitrate_kbps):
            raise ValueError(f"bitrate_kbps must be positive and finite, got {self.bitrate_kbps}")
        if isinstance(self.utility, bool) or not isinstance(self.utility, (int, float)):
            raise TypeError(f"utility must be a numeric value, got {type(self.utility).__name__}")
        if self.utility <= 0.0 or not np.isfinite(self.utility):
            raise ValueError(f"utility must be positive and finite, got {self.utility}")


class TileQualityConfig:
    """
    Configuration manager for the 4x6 (24-tile) simulated adaptive quality pipeline.
    """

    CANONICAL_GRID_ROWS: int = 4
    CANONICAL_GRID_COLS: int = 6
    CANONICAL_TOTAL_TILES: int = 24

    # Canonical simulated bitrate assumptions
    DEFAULT_BITRATES_KBPS: Dict[QualityLevel, float] = {
        QualityLevel.Q1: 100.0,
        QualityLevel.Q2: 300.0,
        QualityLevel.Q3: 600.0,
        QualityLevel.Q4: 1000.0,
    }

    # Canonical numerical quality utility assumptions (illustrative diminishing marginal returns)
    DEFAULT_UTILITIES: Dict[QualityLevel, float] = {
        QualityLevel.Q1: 1.00,
        QualityLevel.Q2: 2.00,
        QualityLevel.Q3: 2.80,
        QualityLevel.Q4: 3.40,
    }

    def __init__(
        self,
        grid_rows: int = CANONICAL_GRID_ROWS,
        grid_cols: int = CANONICAL_GRID_COLS,
        custom_bitrates_kbps: Optional[Dict[QualityLevel, float]] = None,
        custom_utilities: Optional[Dict[QualityLevel, float]] = None,
        allow_non_canonical_grid: bool = False,
    ):
        """
        Initializes and validates the tile quality configuration.

        Parameters:
            grid_rows: Number of latitude tile rows (canonical: 4).
            grid_cols: Number of longitude tile columns (canonical: 6).
            custom_bitrates_kbps: Optional dictionary overriding default tier bitrates.
            custom_utilities: Optional dictionary overriding default tier utilities.
            allow_non_canonical_grid: Set to True only for experimental exploration of non-standard
                grids. Defaults to False to strictly enforce the project's canonical 4x6 grid.
        """
        if isinstance(grid_rows, bool) or not isinstance(grid_rows, int) or grid_rows <= 0:
            raise ValueError(f"grid_rows must be a positive integer, got {grid_rows}")
        if isinstance(grid_cols, bool) or not isinstance(grid_cols, int) or grid_cols <= 0:
            raise ValueError(f"grid_cols must be a positive integer, got {grid_cols}")

        if (grid_rows != self.CANONICAL_GRID_ROWS or grid_cols != self.CANONICAL_GRID_COLS) and not allow_non_canonical_grid:
            raise ValueError(
                f"Configuration ({grid_rows} rows x {grid_cols} cols) contradicts canonical project grid "
                f"({self.CANONICAL_GRID_ROWS} rows x {self.CANONICAL_GRID_COLS} cols = {self.CANONICAL_TOTAL_TILES} tiles). "
                f"Pass allow_non_canonical_grid=True to explicitly instantiate an experimental configuration."
            )

        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.total_tiles = grid_rows * grid_cols
        self.is_canonical_grid = (grid_rows == self.CANONICAL_GRID_ROWS and grid_cols == self.CANONICAL_GRID_COLS)

        # Bitrates setup
        bitrates = dict(self.DEFAULT_BITRATES_KBPS)
        if custom_bitrates_kbps is not None:
            if not isinstance(custom_bitrates_kbps, dict):
                raise TypeError(f"custom_bitrates_kbps must be a dict, got {type(custom_bitrates_kbps).__name__}")
            for k, v in custom_bitrates_kbps.items():
                level = QualityLevel.from_string(k)
                if isinstance(v, bool) or not isinstance(v, (int, float, np.floating, np.integer)):
                    raise TypeError(f"Bitrate for {level} must be numeric, got {type(v).__name__}")
                f_val = float(v)
                if not np.isfinite(f_val):
                    raise ValueError(f"Bitrate for {level} must be finite, got {f_val}")
                if f_val <= 0.0:
                    raise ValueError(f"Bitrate for {level} must be positive, got {f_val}")
                bitrates[level] = f_val

        # Utilities setup
        utilities = dict(self.DEFAULT_UTILITIES)
        if custom_utilities is not None:
            if not isinstance(custom_utilities, dict):
                raise TypeError(f"custom_utilities must be a dict, got {type(custom_utilities).__name__}")
            for k, v in custom_utilities.items():
                level = QualityLevel.from_string(k)
                if isinstance(v, bool) or not isinstance(v, (int, float, np.floating, np.integer)):
                    raise TypeError(f"Utility for {level} must be numeric, got {type(v).__name__}")
                f_val = float(v)
                if not np.isfinite(f_val):
                    raise ValueError(f"Utility for {level} must be finite, got {f_val}")
                if f_val <= 0.0:
                    raise ValueError(f"Utility for {level} must be positive, got {f_val}")
                utilities[level] = f_val

        # Construct tier specifications
        descriptions = {
            QualityLevel.Q1: "Low quality tier (simulated baseline fallback, 100 kbps)",
            QualityLevel.Q2: "Medium quality tier (simulated standard delivery, 300 kbps)",
            QualityLevel.Q3: "High quality tier (simulated enhanced viewport, 600 kbps)",
            QualityLevel.Q4: "Ultra quality tier (simulated pristine focal center, 1000 kbps)",
        }

        names = {
            QualityLevel.Q1: "Low",
            QualityLevel.Q2: "Medium",
            QualityLevel.Q3: "High",
            QualityLevel.Q4: "Ultra",
        }

        self.tiers: Dict[QualityLevel, QualityTierSpec] = {}
        for level in QualityLevel:
            if level not in bitrates:
                raise ValueError(f"Missing bitrate configuration for level: {level}")
            if level not in utilities:
                raise ValueError(f"Missing utility configuration for level: {level}")

            self.tiers[level] = QualityTierSpec(
                level=level,
                name=names[level],
                bitrate_kbps=bitrates[level],
                utility=utilities[level],
                description=descriptions[level],
            )

        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """
        Validates internal consistency: strict monotonicity of bitrates and utilities across tiers.
        """
        ordered_levels = [QualityLevel.Q1, QualityLevel.Q2, QualityLevel.Q3, QualityLevel.Q4]
        for i in range(len(ordered_levels) - 1):
            curr_l = ordered_levels[i]
            next_l = ordered_levels[i + 1]

            curr_b = self.tiers[curr_l].bitrate_kbps
            next_b = self.tiers[next_l].bitrate_kbps
            if curr_b >= next_b:
                raise ValueError(
                    f"Bitrate must strictly increase with quality tier: {curr_l.value} ({curr_b} kbps) >= {next_l.value} ({next_b} kbps)"
                )

            curr_u = self.tiers[curr_l].utility
            next_u = self.tiers[next_l].utility
            if curr_u >= next_u:
                raise ValueError(
                    f"Utility must strictly increase with quality tier: {curr_l.value} ({curr_u}) >= {next_l.value} ({next_u})"
                )

    def get_tier(self, level: Union[QualityLevel, str]) -> QualityTierSpec:
        """Returns the tier specification for a quality level."""
        parsed = QualityLevel.from_string(level)
        return self.tiers[parsed]

    def get_bitrate(self, level: Union[QualityLevel, str]) -> float:
        """Returns the simulated bitrate in kbps for a quality level."""
        return self.get_tier(level).bitrate_kbps

    def get_utility(self, level: Union[QualityLevel, str]) -> float:
        """Returns the numerical utility for a quality level."""
        return self.get_tier(level).utility

    def validate_tile_id(self, tile_id: int) -> int:
        """
        Validates that a tile ID is a valid integer within [0, total_tiles - 1].

        Raises:
            TypeError: If tile_id is a boolean or not an int.
            ValueError: If tile_id is outside the grid bounds.
        """
        if isinstance(tile_id, bool) or not isinstance(tile_id, (int, np.integer)):
            raise TypeError(f"tile_id must be an integer, got {type(tile_id).__name__}")
        t_int = int(tile_id)
        if t_int < 0 or t_int >= self.total_tiles:
            raise ValueError(
                f"tile_id {t_int} out of valid bounds [0, {self.total_tiles - 1}] for a {self.grid_rows}x{self.grid_cols} grid."
            )
        return t_int

    def validate_bandwidth_budget(self, budget_kbps: float) -> float:
        """
        Validates that a bandwidth budget is a non-negative finite numeric value.

        Raises:
            TypeError: If budget_kbps is a boolean or non-numeric.
            ValueError: If budget_kbps is negative or non-finite.
        """
        if isinstance(budget_kbps, bool) or not isinstance(budget_kbps, (int, float, np.floating, np.integer)):
            raise TypeError(f"budget_kbps must be a numeric value, got {type(budget_kbps).__name__}")
        b_val = float(budget_kbps)
        if b_val < 0.0 or not np.isfinite(b_val):
            raise ValueError(f"budget_kbps must be non-negative and finite, got {budget_kbps}")
        return b_val

    def validate_allocation(
        self,
        allocation: Union[Dict[int, Any], TileQualityAllocation],
        allow_incomplete: bool = False
    ) -> Dict[int, QualityLevel]:
        """
        Validates a dictionary or TileQualityAllocation of per-tile assignments.

        Parameters:
            allocation: Mapping of tile_id (int) -> QualityLevel or string.
            allow_incomplete: If True, partial assignments (< total_tiles) are permitted.

        Returns:
            Dict[int, QualityLevel]: Cleaned dictionary of validated assignments.

        Raises:
            TypeError: If allocation structure is malformed.
            ValueError: If invalid tile IDs, invalid quality levels, or missing tiles (when allow_incomplete=False).
        """
        if isinstance(allocation, TileQualityAllocation):
            assign_dict = allocation.assignments
        elif isinstance(allocation, dict):
            assign_dict = allocation
        else:
            raise TypeError(f"allocation must be a dict or TileQualityAllocation, got {type(allocation).__name__}")

        if not assign_dict and not allow_incomplete:
            raise ValueError("Allocation is empty, expected full 24-tile assignment.")

        cleaned: Dict[int, QualityLevel] = {}
        for raw_id, raw_level in assign_dict.items():
            valid_id = self.validate_tile_id(raw_id)
            valid_level = QualityLevel.from_string(raw_level)
            cleaned[valid_id] = valid_level

        if not allow_incomplete:
            expected_ids = set(range(self.total_tiles))
            assigned_ids = set(cleaned.keys())
            missing = expected_ids - assigned_ids
            if missing:
                raise ValueError(
                    f"Incomplete tile allocation: missing {len(missing)} tiles: {sorted(list(missing))}"
                )

        return cleaned

    def create_uniform_allocation(self, level: Union[QualityLevel, str]) -> TileQualityAllocation:
        """
        Creates a uniform quality allocation where all 24 tiles are assigned the identical quality tier.
        """
        parsed_level = QualityLevel.from_string(level)
        assignments = {t_id: parsed_level for t_id in range(self.total_tiles)}
        return TileQualityAllocation(assignments=assignments, config=self)

    def calculate_total_bitrate(
        self,
        allocation: Union[Dict[int, Any], TileQualityAllocation]
    ) -> float:
        """Calculates total simulated bitrate in kbps across all assigned tiles."""
        cleaned = self.validate_allocation(allocation, allow_incomplete=True)
        return sum(self.get_bitrate(lvl) for lvl in cleaned.values())

    def satisfies_budget(
        self,
        allocation: Union[Dict[int, Any], TileQualityAllocation],
        budget_kbps: float
    ) -> bool:
        """Determines whether an allocation's total bitrate is within a bandwidth budget."""
        valid_budget = self.validate_bandwidth_budget(budget_kbps)
        total_bitrate = self.calculate_total_bitrate(allocation)
        return total_bitrate <= valid_budget


@dataclass
class TileQualityAllocation:
    """
    Encapsulates a per-tile quality assignment across the ERP video grid,
    providing utility, bitrate, budget satisfaction, and tier distribution methods.
    """
    assignments: Dict[int, QualityLevel]
    config: TileQualityConfig

    def __post_init__(self) -> None:
        if not isinstance(self.config, TileQualityConfig):
            raise TypeError(f"config must be an instance of TileQualityConfig, got {type(self.config).__name__}")
        # Clean and validate assignments
        self.assignments = self.config.validate_allocation(self.assignments, allow_incomplete=True)

    @property
    def total_tiles_assigned(self) -> int:
        """Number of tiles currently assigned in this allocation."""
        return len(self.assignments)

    @property
    def is_complete(self) -> bool:
        """True if all 24 tiles in the grid have been assigned a quality level."""
        return len(self.assignments) == self.config.total_tiles

    @property
    def missing_tiles(self) -> Set[int]:
        """Set of tile IDs that have not yet been assigned a quality level."""
        return set(range(self.config.total_tiles)) - set(self.assignments.keys())

    @property
    def total_bitrate_kbps(self) -> float:
        """Total simulated bitrate consumption across all assigned tiles in kbps."""
        return sum(self.config.get_bitrate(lvl) for lvl in self.assignments.values())

    @property
    def total_bitrate_mbps(self) -> float:
        """Total simulated bitrate consumption in Megabits per second (Mbps)."""
        return self.total_bitrate_kbps / 1000.0

    @property
    def mean_utility(self) -> float:
        """
        Arithmetic mean of numerical quality utility across assigned tiles.
        Returns 0.0 if no tiles are assigned.
        """
        if not self.assignments:
            return 0.0
        return float(np.mean([self.config.get_utility(lvl) for lvl in self.assignments.values()]))

    def weighted_utility(
        self,
        tile_weights: Dict[int, float],
        tolerance: float = 1e-4
    ) -> float:
        """
        Calculates spatially weighted quality utility:
            U_weighted = sum_{i} (weight_i * Utility(quality_i))

        This enables viewport-aware and saliency-aware evaluation where tiles inside
        the user's line of sight contribute proportionally to visual satisfaction.

        Parameters:
            tile_weights: Dict mapping tile_id (int) -> non-negative importance weight.
                Weights must sum to 1.0 within documented numerical tolerance.
            tolerance: Maximum allowable absolute deviation of weight sum from 1.0 (default: 1e-4).

        Raises:
            TypeError: If tile_weights is not a dictionary or contains non-numeric/boolean weights.
            ValueError: If empty dict, invalid tile IDs, negative/NaN/infinite weights,
                        zero-sum weights, weights not summing to 1.0 within tolerance,
                        or any positive weight refers to a tile missing from self.assignments.
        """
        if not isinstance(tile_weights, dict):
            raise TypeError(f"tile_weights must be a dict, got {type(tile_weights).__name__}")
        if not tile_weights:
            raise ValueError("tile_weights dictionary cannot be empty; weights must sum to 1.0.")

        cleaned_weights: Dict[int, float] = {}
        for raw_id, raw_w in tile_weights.items():
            t_id = self.config.validate_tile_id(raw_id)
            if isinstance(raw_w, bool) or not isinstance(raw_w, (int, float, np.floating, np.integer)):
                raise TypeError(f"Weight for tile {t_id} must be numeric, got {type(raw_w).__name__}")
            w = float(raw_w)
            if not np.isfinite(w):
                raise ValueError(f"Weight for tile {t_id} must be finite, got {w}")
            if w < 0.0:
                raise ValueError(f"Weight for tile {t_id} cannot be negative, got {w}")
            cleaned_weights[t_id] = w

        sum_w = sum(cleaned_weights.values())
        if sum_w <= 0.0 or np.isclose(sum_w, 0.0):
            raise ValueError(f"Zero-sum weights encountered (sum={sum_w}); weights must sum to 1.0.")

        if abs(sum_w - 1.0) > tolerance:
            raise ValueError(
                f"Tile weights must sum to 1.0 within numerical tolerance {tolerance}, "
                f"got sum={sum_w:.6f} (absolute deviation={abs(sum_w - 1.0):.6f})."
            )

        # Check for missing tiles with positive weights (do not silently ignore missing important tiles)
        for t_id, w in cleaned_weights.items():
            if w > 0.0 and t_id not in self.assignments:
                raise ValueError(
                    f"Tile {t_id} has positive viewport weight ({w}), but has not been assigned a quality level in this allocation."
                )

        # U_weighted = sum(weight_i * utility(quality_i))
        weighted_sum = sum(
            w * self.config.get_utility(self.assignments[t_id])
            for t_id, w in cleaned_weights.items()
            if w > 0.0
        )
        return float(weighted_sum)

    def satisfies_budget(self, budget_kbps: float) -> bool:
        """Returns True if the allocation total bitrate does not exceed budget_kbps."""
        b_val = self.config.validate_bandwidth_budget(budget_kbps)
        return self.total_bitrate_kbps <= b_val

    def remaining_budget(self, budget_kbps: float) -> float:
        """Returns the unused bitrate headroom (budget_kbps - total_bitrate_kbps)."""
        b_val = self.config.validate_bandwidth_budget(budget_kbps)
        return b_val - self.total_bitrate_kbps

    @property
    def tier_distribution(self) -> Dict[QualityLevel, int]:
        """Returns a breakdown of how many tiles are assigned to each quality tier."""
        counts = {level: 0 for level in QualityLevel}
        for lvl in self.assignments.values():
            counts[lvl] += 1
        return counts

    def get_tile_quality(self, tile_id: int) -> QualityLevel:
        """Returns the assigned QualityLevel for a specific tile."""
        valid_id = self.config.validate_tile_id(tile_id)
        if valid_id not in self.assignments:
            raise KeyError(f"Tile {valid_id} has not been assigned a quality level.")
        return self.assignments[valid_id]

    def set_tile_quality(self, tile_id: int, quality: Union[QualityLevel, str]) -> None:
        """Updates or sets the quality level of an individual tile."""
        valid_id = self.config.validate_tile_id(tile_id)
        valid_level = QualityLevel.from_string(quality)
        self.assignments[valid_id] = valid_level

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the allocation summary into a Python dictionary."""
        return {
            "total_tiles_assigned": self.total_tiles_assigned,
            "is_complete": self.is_complete,
            "total_bitrate_kbps": self.total_bitrate_kbps,
            "total_bitrate_mbps": self.total_bitrate_mbps,
            "mean_utility": round(self.mean_utility, 4),
            "tier_distribution": {k.value: v for k, v in self.tier_distribution.items()},
            "assignments": {k: v.value for k, v in sorted(self.assignments.items())},
        }
