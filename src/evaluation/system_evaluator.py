"""
Module: system_evaluator.py
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming
Phase: Phase 5, Day 3 — Complete System Evaluation

Implements end-to-end evaluation of the integrated system:
1. Tile prediction evaluation on the 4x6 canonical grid (Recall, Precision, FoV Coverage, Fetched Tiles).
2. Strategy comparison across bandwidth budgets (Uniform, Viewport-Aware, Neighbourhood alpha=0.25, alpha=0.50).
3. Controlled sudden head motion simulation and utility retention analysis.
4. Empirical test-trajectory evaluation (realized in-FoV utility on ground-truth viewing orientations).
5. Granular per-subject and per-video performance aggregations.
6. Robustness and sensitivity analysis across FoV sizes, ray sampling resolutions, and spherical boundary cases.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.quality.quality_config import (
    QualityLevel,
    TileQualityConfig,
    TileQualityAllocation,
)
from src.quality.uniform_allocator import (
    UniformQualityAllocator,
    UniformAllocationResult,
)
from src.quality.viewport_allocator import (
    ViewportQualityAllocator,
    ViewportAllocationResult,
)
from src.quality.neighbourhood_allocator import (
    NeighbourhoodQualityAllocator,
    NeighbourhoodAllocationResult,
    TileNeighbourhoodFinder,
)
from src.tiling.grid_partitioner import TileGridPartitioner
from src.tiling.viewport_mapper import ViewportMapper, ViewportIntersection


CANONICAL_BUDGETS_KBPS: List[float] = [2400.0, 5000.0, 7200.0, 10000.0, 14400.0, 20000.0]


@dataclass(frozen=True)
class TilePredictionBenchmarkResult:
    """
    Summary metrics for tile prediction evaluation.
    """
    strategy: str
    mean_tile_recall: float
    mean_tile_precision: float
    mean_viewport_coverage: float
    p10_viewport_coverage: float
    p90_viewport_coverage: float
    avg_tiles_fetched: float
    tile_area_ratio_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": str(self.strategy),
            "mean_tile_recall": float(self.mean_tile_recall),
            "mean_tile_precision": float(self.mean_tile_precision),
            "mean_viewport_coverage": float(self.mean_viewport_coverage),
            "p10_viewport_coverage": float(self.p10_viewport_coverage),
            "p90_viewport_coverage": float(self.p90_viewport_coverage),
            "avg_tiles_fetched": float(self.avg_tiles_fetched),
            "tile_area_ratio_pct": float(self.tile_area_ratio_pct),
        }


class SystemEvaluator:
    """
    Comprehensive system evaluator integrating Day 1 predictions/tiling with Day 2 quality allocators.
    """

    def __init__(
        self,
        config: Optional[TileQualityConfig] = None,
        grid_partitioner: Optional[TileGridPartitioner] = None,
        hfov_deg: float = 100.0,
        vfov_deg: float = 100.0,
        ray_samples_per_axis: int = 21,
    ) -> None:
        self.config = config if config is not None else TileQualityConfig()
        self.grid = grid_partitioner if grid_partitioner is not None else TileGridPartitioner()
        self.mapper = ViewportMapper(
            grid_partitioner=self.grid,
            hfov_deg=hfov_deg,
            vfov_deg=vfov_deg,
            ray_samples_per_axis=ray_samples_per_axis,
        )
        self.uniform_allocator = UniformQualityAllocator(config=self.config)
        self.viewport_allocator = ViewportQualityAllocator(config=self.config)
        self.neighbourhood_allocator_25 = NeighbourhoodQualityAllocator(
            config=self.config, neighbour_weight_factor=0.25
        )
        self.neighbourhood_allocator_50 = NeighbourhoodQualityAllocator(
            config=self.config, neighbour_weight_factor=0.50
        )
        self.finder = TileNeighbourhoodFinder(num_rows=self.config.grid_rows, num_cols=self.config.grid_cols)

    # -------------------------------------------------------------------------
    # 1. TILE PREDICTION EVALUATION
    # -------------------------------------------------------------------------
    def evaluate_tile_predictions(
        self,
        df: pd.DataFrame,
        sample_stride: int = 10
    ) -> Dict[str, TilePredictionBenchmarkResult]:
        """
        Evaluates tile prediction recall, precision, and coverage across test samples:
        - Persistence Baseline
        - Final GRU Model (No Guard Band)
        - Final GRU Model (+1 Guard Band Margin)
        """
        sub_df = df.iloc[::sample_stride].reset_index(drop=True)
        n_samples = len(sub_df)
        if n_samples == 0:
            raise ValueError("No samples available for tile evaluation with given stride.")

        # Metric accumulators
        metrics = {
            "persistence": {"recall": [], "precision": [], "coverage": [], "tiles_count": []},
            "gru_no_margin": {"recall": [], "precision": [], "coverage": [], "tiles_count": []},
            "gru_with_margin": {"recall": [], "precision": [], "coverage": [], "tiles_count": []},
        }

        for idx in range(n_samples):
            row = sub_df.iloc[idx]
            act_y = float(row["actual_yaw_deg"])
            act_p = float(row["actual_pitch_deg"])
            pred_y = float(row["predicted_yaw_deg"])
            pred_p = float(row["predicted_pitch_deg"])
            pers_y = float(row["persistence_predicted_yaw_deg"])
            pers_p = float(row["persistence_predicted_pitch_deg"])

            # 1. Ground truth intersection
            act_inter = self.mapper.get_viewport_intersection(act_y, act_p)
            act_tiles = set(act_inter.active_tiles)

            # 2. Persistence baseline
            pers_inter = self.mapper.get_viewport_intersection(pers_y, pers_p)
            pers_tiles = set(pers_inter.active_tiles)
            pers_recall = len(act_tiles.intersection(pers_tiles)) / len(act_tiles) * 100.0
            pers_prec = len(act_tiles.intersection(pers_tiles)) / len(pers_tiles) * 100.0
            pers_cov = sum(act_inter.tile_weights.get(t, 0.0) for t in pers_tiles) * 100.0
            metrics["persistence"]["recall"].append(pers_recall)
            metrics["persistence"]["precision"].append(pers_prec)
            metrics["persistence"]["coverage"].append(pers_cov)
            metrics["persistence"]["tiles_count"].append(len(pers_tiles))

            # 3. Final GRU (No Guard Band)
            gru_inter = self.mapper.get_viewport_intersection(pred_y, pred_p)
            gru_tiles = set(gru_inter.active_tiles)
            gru_recall = len(act_tiles.intersection(gru_tiles)) / len(act_tiles) * 100.0
            gru_prec = len(act_tiles.intersection(gru_tiles)) / len(gru_tiles) * 100.0
            gru_cov = sum(act_inter.tile_weights.get(t, 0.0) for t in gru_tiles) * 100.0
            metrics["gru_no_margin"]["recall"].append(gru_recall)
            metrics["gru_no_margin"]["precision"].append(gru_prec)
            metrics["gru_no_margin"]["coverage"].append(gru_cov)
            metrics["gru_no_margin"]["tiles_count"].append(len(gru_tiles))

            # 4. Final GRU (+1 Guard Band Margin)
            margin_tiles = set(gru_inter.margin_tiles)
            guard_tiles = gru_tiles.union(margin_tiles)
            guard_recall = len(act_tiles.intersection(guard_tiles)) / len(act_tiles) * 100.0
            guard_prec = len(act_tiles.intersection(guard_tiles)) / len(guard_tiles) * 100.0
            guard_cov = sum(act_inter.tile_weights.get(t, 0.0) for t in guard_tiles) * 100.0
            metrics["gru_with_margin"]["recall"].append(guard_recall)
            metrics["gru_with_margin"]["precision"].append(guard_prec)
            metrics["gru_with_margin"]["coverage"].append(guard_cov)
            metrics["gru_with_margin"]["tiles_count"].append(len(guard_tiles))

        results: Dict[str, TilePredictionBenchmarkResult] = {}
        names = {
            "persistence": "Persistence Baseline",
            "gru_no_margin": "Final GRU Model (No Guard Band)",
            "gru_with_margin": "Final GRU Model (+1 Guard Band Margin)",
        }
        for key, name in names.items():
            m = metrics[key]
            avg_tiles = float(np.mean(m["tiles_count"]))
            results[key] = TilePredictionBenchmarkResult(
                strategy=name,
                mean_tile_recall=float(np.mean(m["recall"])),
                mean_tile_precision=float(np.mean(m["precision"])),
                mean_viewport_coverage=float(np.mean(m["coverage"])),
                p10_viewport_coverage=float(np.percentile(m["coverage"], 10.0)),
                p90_viewport_coverage=float(np.percentile(m["coverage"], 90.0)),
                avg_tiles_fetched=avg_tiles,
                tile_area_ratio_pct=(avg_tiles / float(self.config.total_tiles)) * 100.0,
            )

        return results

    def evaluate_canonical_boundary_orientations(self) -> pd.DataFrame:
        """
        Tests tile prediction mapping across canonical boundary and extreme orientations:
        Equator (0,0), Zenith (0,90), Nadir (0,-90), Wraparound (179,0) vs (-179,0), Oblique (180,45).
        """
        cases = [
            ("Equator Center", 0.0, 0.0),
            ("Zenith North Pole", 0.0, 90.0),
            ("Nadir South Pole", 0.0, -90.0),
            ("Yaw Wraparound East (+179°)", 179.0, 0.0),
            ("Yaw Wraparound West (-179°)", -179.0, 0.0),
            ("Oblique Polar Boundary", 180.0, 45.0),
            ("Extreme Corner", -180.0, -45.0),
        ]
        rows = []
        for name, y, p in cases:
            inter = self.mapper.get_viewport_intersection(y, p)
            weights = ViewportQualityAllocator.expand_weights_to_grid(inter.tile_weights)
            rows.append({
                "case_name": name,
                "center_yaw_deg": y,
                "center_pitch_deg": p,
                "primary_tile_id": inter.primary_tile,
                "num_active_tiles": len(inter.active_tiles),
                "active_tile_ids": str(inter.active_tiles),
                "num_margin_tiles": len(inter.margin_tiles),
                "margin_tile_ids": str(inter.margin_tiles),
                "weights_sum": float(sum(weights.values())),
            })
        return pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # 2. STRATEGY COMPARISON ACROSS BANDWIDTH BUDGETS
    # -------------------------------------------------------------------------
    def evaluate_allocation_strategies_on_dataset(
        self,
        df: pd.DataFrame,
        budgets_kbps: Optional[List[float]] = None,
        sample_stride: int = 20,
    ) -> pd.DataFrame:
        """
        Evaluates all four quality allocation strategies on the held-out test dataset across bandwidth budgets:
        1. Uniform Allocation (Module 2)
        2. Viewport-Aware Allocation (Module 3)
        3. Neighbourhood-Aware Allocation (alpha=0.25, Module 4)
        4. Neighbourhood-Aware Allocation (alpha=0.50, Module 4)

        Measures realized in-FoV weighted utility experienced by the user on the actual ground-truth viewport.
        """
        budgets = CANONICAL_BUDGETS_KBPS if budgets_kbps is None else budgets_kbps
        sub_df = df.iloc[::sample_stride].reset_index(drop=True)
        n_samples = len(sub_df)
        if n_samples == 0:
            raise ValueError("No samples available for strategy evaluation.")

        records = []
        for b in budgets:
            # Accumulators per strategy for budget b
            stats = {
                "Uniform": {"bitrate": [], "headroom": [], "pred_util": [], "act_util": [], "mean_util": [], "q4": [], "q3": [], "q2": [], "q1": []},
                "Viewport-Aware": {"bitrate": [], "headroom": [], "pred_util": [], "act_util": [], "mean_util": [], "q4": [], "q3": [], "q2": [], "q1": []},
                "Neighbourhood (alpha=0.25)": {"bitrate": [], "headroom": [], "pred_util": [], "act_util": [], "mean_util": [], "q4": [], "q3": [], "q2": [], "q1": []},
                "Neighbourhood (alpha=0.50)": {"bitrate": [], "headroom": [], "pred_util": [], "act_util": [], "mean_util": [], "q4": [], "q3": [], "q2": [], "q1": []},
            }

            for idx in range(n_samples):
                row = sub_df.iloc[idx]
                act_y = float(row["actual_yaw_deg"])
                act_p = float(row["actual_pitch_deg"])
                pred_y = float(row["predicted_yaw_deg"])
                pred_p = float(row["predicted_pitch_deg"])

                # Ground truth actual viewport weights
                act_inter = self.mapper.get_viewport_intersection(act_y, act_p)
                w_act = ViewportQualityAllocator.expand_weights_to_grid(act_inter.tile_weights)

                # Predicted viewport weights
                pred_inter = self.mapper.get_viewport_intersection(pred_y, pred_p)
                w_pred = ViewportQualityAllocator.expand_weights_to_grid(pred_inter.tile_weights)

                # 1. Uniform
                u_res = self.uniform_allocator.allocate_with_result(b)
                if u_res.allocation is not None:
                    u_alloc = u_res.allocation
                    stats["Uniform"]["bitrate"].append(u_res.total_bitrate_kbps)
                    stats["Uniform"]["headroom"].append(u_res.remaining_budget_kbps)
                    stats["Uniform"]["pred_util"].append(u_alloc.weighted_utility(w_pred))
                    stats["Uniform"]["act_util"].append(u_alloc.weighted_utility(w_act))
                    stats["Uniform"]["mean_util"].append(u_alloc.mean_utility)
                    t_dist = u_alloc.tier_distribution
                    stats["Uniform"]["q4"].append(t_dist[QualityLevel.Q4])
                    stats["Uniform"]["q3"].append(t_dist[QualityLevel.Q3])
                    stats["Uniform"]["q2"].append(t_dist[QualityLevel.Q2])
                    stats["Uniform"]["q1"].append(t_dist[QualityLevel.Q1])

                # 2. Viewport-Aware
                vp_res = self.viewport_allocator.allocate_with_result(w_pred, b)
                if vp_res.allocation is not None:
                    vp_alloc = vp_res.allocation
                    stats["Viewport-Aware"]["bitrate"].append(vp_res.total_bitrate_kbps)
                    stats["Viewport-Aware"]["headroom"].append(vp_res.remaining_budget_kbps)
                    stats["Viewport-Aware"]["pred_util"].append(vp_res.weighted_utility)
                    stats["Viewport-Aware"]["act_util"].append(vp_alloc.weighted_utility(w_act))
                    stats["Viewport-Aware"]["mean_util"].append(vp_alloc.mean_utility)
                    t_dist = vp_alloc.tier_distribution
                    stats["Viewport-Aware"]["q4"].append(t_dist[QualityLevel.Q4])
                    stats["Viewport-Aware"]["q3"].append(t_dist[QualityLevel.Q3])
                    stats["Viewport-Aware"]["q2"].append(t_dist[QualityLevel.Q2])
                    stats["Viewport-Aware"]["q1"].append(t_dist[QualityLevel.Q1])

                # 3. Neighbourhood-Aware (alpha=0.25)
                n25_res = self.neighbourhood_allocator_25.allocate_with_result(w_pred, b)
                if n25_res.allocation is not None:
                    n25_alloc = n25_res.allocation
                    stats["Neighbourhood (alpha=0.25)"]["bitrate"].append(n25_res.total_bitrate_kbps)
                    stats["Neighbourhood (alpha=0.25)"]["headroom"].append(n25_res.remaining_budget_kbps)
                    stats["Neighbourhood (alpha=0.25)"]["pred_util"].append(n25_res.weighted_utility_original)
                    stats["Neighbourhood (alpha=0.25)"]["act_util"].append(n25_alloc.weighted_utility(w_act))
                    stats["Neighbourhood (alpha=0.25)"]["mean_util"].append(n25_alloc.mean_utility)
                    t_dist = n25_alloc.tier_distribution
                    stats["Neighbourhood (alpha=0.25)"]["q4"].append(t_dist[QualityLevel.Q4])
                    stats["Neighbourhood (alpha=0.25)"]["q3"].append(t_dist[QualityLevel.Q3])
                    stats["Neighbourhood (alpha=0.25)"]["q2"].append(t_dist[QualityLevel.Q2])
                    stats["Neighbourhood (alpha=0.25)"]["q1"].append(t_dist[QualityLevel.Q1])

                # 4. Neighbourhood-Aware (alpha=0.50)
                n50_res = self.neighbourhood_allocator_50.allocate_with_result(w_pred, b)
                if n50_res.allocation is not None:
                    n50_alloc = n50_res.allocation
                    stats["Neighbourhood (alpha=0.50)"]["bitrate"].append(n50_res.total_bitrate_kbps)
                    stats["Neighbourhood (alpha=0.50)"]["headroom"].append(n50_res.remaining_budget_kbps)
                    stats["Neighbourhood (alpha=0.50)"]["pred_util"].append(n50_res.weighted_utility_original)
                    stats["Neighbourhood (alpha=0.50)"]["act_util"].append(n50_alloc.weighted_utility(w_act))
                    stats["Neighbourhood (alpha=0.50)"]["mean_util"].append(n50_alloc.mean_utility)
                    t_dist = n50_alloc.tier_distribution
                    stats["Neighbourhood (alpha=0.50)"]["q4"].append(t_dist[QualityLevel.Q4])
                    stats["Neighbourhood (alpha=0.50)"]["q3"].append(t_dist[QualityLevel.Q3])
                    stats["Neighbourhood (alpha=0.50)"]["q2"].append(t_dist[QualityLevel.Q2])
                    stats["Neighbourhood (alpha=0.50)"]["q1"].append(t_dist[QualityLevel.Q1])

            # Compile aggregate rows for budget b
            for strat_name, s in stats.items():
                mean_bitrate = float(np.mean(s["bitrate"]))
                records.append({
                    "bandwidth_budget_kbps": float(b),
                    "strategy": strat_name,
                    "simulated_bitrate_kbps": mean_bitrate,
                    "remaining_headroom_kbps": float(np.mean(s["headroom"])),
                    "budget_utilization_pct": (mean_bitrate / float(b)) * 100.0,
                    "predicted_weighted_utility": float(np.mean(s["pred_util"])),
                    "actual_realized_utility": float(np.mean(s["act_util"])),
                    "utility_retention_pct": (float(np.mean(s["act_util"])) / float(np.mean(s["pred_util"]))) * 100.0,
                    "mean_canvas_utility": float(np.mean(s["mean_util"])),
                    "avg_q4_tiles": float(np.mean(s["q4"])),
                    "avg_q3_tiles": float(np.mean(s["q3"])),
                    "avg_q2_tiles": float(np.mean(s["q2"])),
                    "avg_q1_tiles": float(np.mean(s["q1"])),
                })

        return pd.DataFrame(records)

    # -------------------------------------------------------------------------
    # 3. CONTROLLED HEAD MOTION SIMULATION & UTILITY RETENTION
    # -------------------------------------------------------------------------
    def simulate_controlled_head_movement(
        self,
        base_yaw_deg: float = 0.0,
        base_pitch_deg: float = 0.0,
        budget_kbps: float = 7200.0,
        angle_offsets_deg: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        Simulates controlled rotational head movements away from the predicted orientation.
        Measures realized in-FoV weighted utility and retention ratio (%) for all 4 strategies:
            Retention(%) = (U_realized(delta_theta) / U_predicted(0)) * 100%

        Clearly distinguishes controlled synthetic perturbations from real viewer trajectories.
        """
        angles = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 45.0] if angle_offsets_deg is None else angle_offsets_deg

        # Base predicted intersection and weights
        pred_inter = self.mapper.get_viewport_intersection(base_yaw_deg, base_pitch_deg)
        w_pred = ViewportQualityAllocator.expand_weights_to_grid(pred_inter.tile_weights)

        # Allocate quality once based on base predicted orientation
        u_alloc = self.uniform_allocator.allocate(budget_kbps)
        vp_alloc = self.viewport_allocator.allocate(w_pred, budget_kbps)
        n25_alloc = self.neighbourhood_allocator_25.allocate(w_pred, budget_kbps)
        n50_alloc = self.neighbourhood_allocator_50.allocate(w_pred, budget_kbps)

        # Base predicted utilities at delta_theta = 0
        u_base = u_alloc.weighted_utility(w_pred)
        vp_base = vp_alloc.weighted_utility(w_pred)
        n25_base = n25_alloc.weighted_utility(w_pred)
        n50_base = n50_alloc.weighted_utility(w_pred)

        records = []
        for delta in angles:
            # Test multiple rotation directions: Yaw-only, Pitch-only, Diagonal
            directions = [
                ("Yaw Rotation", base_yaw_deg + delta, base_pitch_deg),
                ("Pitch Rotation", base_yaw_deg, np.clip(base_pitch_deg + delta, -90.0, 90.0)),
                ("Diagonal Rotation", base_yaw_deg + delta * 0.7071, np.clip(base_pitch_deg + delta * 0.7071, -90.0, 90.0)),
            ]

            for dir_name, pert_y, pert_p in directions:
                pert_inter = self.mapper.get_viewport_intersection(pert_y, pert_p)
                w_pert = ViewportQualityAllocator.expand_weights_to_grid(pert_inter.tile_weights)

                # Realized utility for each strategy under perturbed orientation
                u_act = u_alloc.weighted_utility(w_pert)
                vp_act = vp_alloc.weighted_utility(w_pert)
                n25_act = n25_alloc.weighted_utility(w_pert)
                n50_act = n50_alloc.weighted_utility(w_pert)

                records.append({
                    "delta_theta_deg": float(delta),
                    "rotation_direction": dir_name,
                    "budget_kbps": float(budget_kbps),
                    "uniform_realized_utility": float(u_act),
                    "uniform_retention_pct": (u_act / u_base) * 100.0,
                    "viewport_realized_utility": float(vp_act),
                    "viewport_retention_pct": (vp_act / vp_base) * 100.0,
                    "neighbourhood_25_realized_utility": float(n25_act),
                    "neighbourhood_25_retention_pct": (n25_act / n25_base) * 100.0,
                    "neighbourhood_50_realized_utility": float(n50_act),
                    "neighbourhood_50_retention_pct": (n50_act / n50_base) * 100.0,
                    "guard_band_gain_25_vs_vp": float(n25_act - vp_act),
                    "guard_band_gain_50_vs_vp": float(n50_act - vp_act),
                })

        return pd.DataFrame(records)

    # -------------------------------------------------------------------------
    # 4. EMPIRICAL RETENTION ON TEST TRAJECTORIES BY ERROR BIN
    # -------------------------------------------------------------------------
    def evaluate_empirical_retention_by_error_bin(
        self,
        df: pd.DataFrame,
        budget_kbps: float = 7200.0,
        sample_stride: int = 20,
    ) -> pd.DataFrame:
        """
        Bins test samples by empirical prediction error (degrees) and calculates
        actual realized utility and retention on real user trajectories.
        """
        sub_df = df.iloc[::sample_stride].reset_index(drop=True)
        bins = [(0.0, 5.0, "0° - 5° (Low Error)"),
                (5.0, 10.0, "5° - 10° (Typical Error)"),
                (10.0, 15.0, "10° - 15° (Moderate Error)"),
                (15.0, 20.0, "15° - 20° (High Error)"),
                (20.0, 180.0, "> 20° (Extreme Saccade)")]

        rows = []
        for low, high, label in bins:
            bin_mask = (sub_df["spherical_error_deg"] >= low) & (sub_df["spherical_error_deg"] < high)
            bin_samples = sub_df[bin_mask]
            if len(bin_samples) == 0:
                continue

            vp_utils, n25_utils, n50_utils, u_utils = [], [], [], []
            vp_ret, n25_ret, n50_ret = [], [], []

            for _, row in bin_samples.iterrows():
                act_y, act_p = float(row["actual_yaw_deg"]), float(row["actual_pitch_deg"])
                pred_y, pred_p = float(row["predicted_yaw_deg"]), float(row["predicted_pitch_deg"])

                w_act = ViewportQualityAllocator.expand_weights_to_grid(
                    self.mapper.get_viewport_intersection(act_y, act_p).tile_weights
                )
                w_pred = ViewportQualityAllocator.expand_weights_to_grid(
                    self.mapper.get_viewport_intersection(pred_y, pred_p).tile_weights
                )

                # Allocations
                u_alloc = self.uniform_allocator.allocate(budget_kbps)
                vp_alloc = self.viewport_allocator.allocate(w_pred, budget_kbps)
                n25_alloc = self.neighbourhood_allocator_25.allocate(w_pred, budget_kbps)
                n50_alloc = self.neighbourhood_allocator_50.allocate(w_pred, budget_kbps)

                u_act_val = u_alloc.weighted_utility(w_act)
                vp_act_val = vp_alloc.weighted_utility(w_act)
                n25_act_val = n25_alloc.weighted_utility(w_act)
                n50_act_val = n50_alloc.weighted_utility(w_act)

                vp_pred_val = vp_alloc.weighted_utility(w_pred)
                n25_pred_val = n25_alloc.weighted_utility(w_pred)
                n50_pred_val = n50_alloc.weighted_utility(w_pred)

                u_utils.append(u_act_val)
                vp_utils.append(vp_act_val)
                n25_utils.append(n25_act_val)
                n50_utils.append(n50_act_val)

                vp_ret.append((vp_act_val / vp_pred_val) * 100.0)
                n25_ret.append((n25_act_val / n25_pred_val) * 100.0)
                n50_ret.append((n50_act_val / n50_pred_val) * 100.0)

            rows.append({
                "error_bin": label,
                "sample_count": len(bin_samples),
                "mean_spherical_error_deg": float(bin_samples["spherical_error_deg"].mean()),
                "uniform_realized_utility": float(np.mean(u_utils)),
                "viewport_realized_utility": float(np.mean(vp_utils)),
                "neighbourhood_25_realized_utility": float(np.mean(n25_utils)),
                "neighbourhood_50_realized_utility": float(np.mean(n50_utils)),
                "viewport_retention_pct": float(np.mean(vp_ret)),
                "neighbourhood_25_retention_pct": float(np.mean(n25_ret)),
                "neighbourhood_50_retention_pct": float(np.mean(n50_ret)),
                "guard_band_gain_25_vs_vp": float(np.mean(n25_utils) - np.mean(vp_utils)),
            })

        return pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # 5. ROBUSTNESS & SENSITIVITY EXPERIMENTS
    # -------------------------------------------------------------------------
    def evaluate_sampling_sensitivity(
        self,
        resolutions: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Tests whether tile selection and overlap weights are sensitive to ray sampling density.
        """
        res_list = [15, 21, 31] if resolutions is None else resolutions
        test_orientations = [
            ("Center", 0.0, 0.0),
            ("Zenith", 0.0, 90.0),
            ("Yaw Wraparound", 180.0, 0.0),
        ]
        rows = []
        for res in res_list:
            m = ViewportMapper(
                grid_partitioner=self.grid,
                hfov_deg=100.0,
                vfov_deg=100.0,
                ray_samples_per_axis=res
            )
            for name, y, p in test_orientations:
                inter = m.get_viewport_intersection(y, p)
                rows.append({
                    "ray_samples_per_axis": res,
                    "total_rays": m.num_rays,
                    "orientation": name,
                    "num_active_tiles": len(inter.active_tiles),
                    "active_tile_ids": str(inter.active_tiles),
                    "primary_tile": inter.primary_tile,
                })
        return pd.DataFrame(rows)

    def evaluate_fov_sensitivity(
        self,
        fov_degrees: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Tests system behavior across different HMD Field of View assumptions (e.g. 90°, 100°, 110°).
        """
        fovs = [90.0, 100.0, 110.0] if fov_degrees is None else fov_degrees
        rows = []
        for fov in fovs:
            m = ViewportMapper(
                grid_partitioner=self.grid,
                hfov_deg=fov,
                vfov_deg=fov,
                ray_samples_per_axis=21
            )
            inter = m.get_viewport_intersection(0.0, 0.0)
            w = ViewportQualityAllocator.expand_weights_to_grid(inter.tile_weights)
            vp_alloc = self.viewport_allocator.allocate_with_result(w, budget_kbps=7200.0)
            n25_alloc = self.neighbourhood_allocator_25.allocate_with_result(w, budget_kbps=7200.0)

            rows.append({
                "fov_deg": fov,
                "active_tiles_count": len(inter.active_tiles),
                "margin_tiles_count": len(inter.margin_tiles),
                "viewport_weighted_utility": vp_alloc.weighted_utility,
                "neighbourhood_weighted_utility": n25_alloc.weighted_utility_original,
                "viewport_q4_tiles": vp_alloc.tier_distribution[QualityLevel.Q4],
                "neighbourhood_margin_upgrades": n25_alloc.upgrades_by_region["neighbourhood"],
            })
        return pd.DataFrame(rows)

