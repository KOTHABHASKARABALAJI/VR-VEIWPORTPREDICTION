"""
Module: viewport_mapper.py
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming

Computes the 3D spherical projection of the user's Head-Mounted Display (HMD) Viewport
(typically 100° x 100° Field of View) onto the 24 spatial tiles of the ERP canvas.

Features:
- Rigorous 3D viewing frustum rotation and equirectangular ray casting.
- Sampled ray-based viewport tile weights: computes the fraction of sampled frustum rays
  intersecting each spatial tile (discrete ray approximation, not continuous analytic geometry).
- Guard band / margin tile calculation for network jitter and saccade motion buffer.
- Tile Hit Ratio (Recall), Tile Precision, and Sampled Viewport Coverage metrics for prediction evaluation.
- Sampling sensitivity analysis across varying ray grid resolutions (21x21, 41x41, 81x81).
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import pandas as pd

from src.tiling.grid_partitioner import TileGridPartitioner, TileSpec


@dataclass
class ViewportIntersection:
    """
    Results of intersecting a user viewport (orientation + FoV) with the 24-tile grid.
    """
    center_yaw_deg: float
    center_pitch_deg: float
    hfov_deg: float
    vfov_deg: float
    active_tiles: List[int]                  # Sorted list of tile IDs intersecting the viewport
    tile_weights: Dict[int, float]           # Fraction of sampled viewport rays in each tile (sum = 1.0)
    primary_tile: int                        # Unique tile containing the line of sight (center)
    margin_tiles: List[int]                  # 1-ring adjacent guard-band tiles surrounding active tiles


class ViewportMapper:
    """
    Maps 3D viewport Field-of-View (FoV) frustums to 24-tile Equirectangular projections
    using sampled ray casting.
    """

    def __init__(
        self,
        grid_partitioner: Optional[TileGridPartitioner] = None,
        hfov_deg: float = 100.0,
        vfov_deg: float = 100.0,
        ray_samples_per_axis: int = 21
    ):
        """
        Parameters:
            grid_partitioner: Instance of TileGridPartitioner (defaults to 4x6 on 3840x1920).
            hfov_deg: Horizontal Field of View in degrees (must be strictly in (0, 180)).
            vfov_deg: Vertical Field of View in degrees (must be strictly in (0, 180)).
            ray_samples_per_axis: Sampling grid resolution per axis (integer >= 2, e.g. 21 -> 441 rays).
        """
        if isinstance(hfov_deg, bool) or not np.isfinite(float(hfov_deg)) or not (0.0 < float(hfov_deg) < 180.0):
            raise ValueError(f"hfov_deg must be strictly between 0° and 180° exclusive, got: {hfov_deg}")
        if isinstance(vfov_deg, bool) or not np.isfinite(float(vfov_deg)) or not (0.0 < float(vfov_deg) < 180.0):
            raise ValueError(f"vfov_deg must be strictly between 0° and 180° exclusive, got: {vfov_deg}")
        if isinstance(ray_samples_per_axis, bool) or not isinstance(ray_samples_per_axis, int) or ray_samples_per_axis < 2:
            raise ValueError(f"ray_samples_per_axis must be an integer >= 2, got: {ray_samples_per_axis}")

        self.grid = grid_partitioner if grid_partitioner is not None else TileGridPartitioner()
        self.hfov_deg = float(hfov_deg)
        self.vfov_deg = float(vfov_deg)
        self.ray_samples = int(ray_samples_per_axis)

        # Precompute canonical local viewing rays in camera coordinate frame (facing +Z)
        half_h_rad = np.radians(self.hfov_deg / 2.0)
        half_v_rad = np.radians(self.vfov_deg / 2.0)
        u_vals = np.linspace(-np.tan(half_h_rad), np.tan(half_h_rad), self.ray_samples)
        v_vals = np.linspace(-np.tan(half_v_rad), np.tan(half_v_rad), self.ray_samples)

        uu, vv = np.meshgrid(u_vals, v_vals)
        uu = uu.flatten()
        vv = vv.flatten()
        ww = np.ones_like(uu)

        # Normalize to unit 3D direction vectors in camera frame: (X_cam=u, Y_cam=v, Z_cam=1)
        norms = np.sqrt(uu**2 + vv**2 + ww**2)
        self._local_rays = np.stack([uu / norms, vv / norms, ww / norms], axis=1)  # (N_rays, 3)
        self.num_rays = len(self._local_rays)

    def _get_rotation_matrix(self, yaw_deg: float, pitch_deg: float) -> np.ndarray:
        """
        Computes 3D rotation matrix transforming camera-local rays to world coordinates:
        R = R_yaw(yaw) @ R_pitch(pitch)
        """
        yaw_rad = np.radians(float(yaw_deg))
        pitch_rad = np.radians(float(pitch_deg))

        # R_pitch (rotation around X axis: tilt up/down)
        cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
        R_pitch = np.array([
            [1.0,    0.0,    0.0],
            [0.0,  cos_p,  sin_p],
            [0.0, -sin_p,  cos_p]
        ], dtype=np.float64)

        # R_yaw (rotation around Y axis: pan left/right)
        cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
        R_yaw = np.array([
            [ cos_y, 0.0, sin_y],
            [   0.0, 1.0,   0.0],
            [-sin_y, 0.0, cos_y]
        ], dtype=np.float64)

        return R_yaw @ R_pitch

    def get_viewport_intersection(
        self,
        yaw_deg: float,
        pitch_deg: float
    ) -> ViewportIntersection:
        """
        Computes active tiles and sampled ray-based overlap weights for given viewport orientation.

        Parameters:
            yaw_deg: Center yaw in degrees [-180, +180].
            pitch_deg: Center pitch in degrees [-90, +90].

        Returns:
            ViewportIntersection dataclass.
        """
        if isinstance(yaw_deg, bool) or not np.isfinite(float(yaw_deg)):
            raise ValueError(f"Invalid yaw_deg: {yaw_deg}")
        if isinstance(pitch_deg, bool) or not np.isfinite(float(pitch_deg)):
            raise ValueError(f"Invalid pitch_deg: {pitch_deg}")
        if float(pitch_deg) < -90.001 or float(pitch_deg) > 90.001:
            raise ValueError(f"pitch_deg must be within [-90°, 90°], got {pitch_deg}")

        yaw_val = float(yaw_deg)
        pitch_val = float(pitch_deg)

        # Primary tile containing the center line-of-sight
        primary_tile = self.grid.point_to_tile(yaw_val, pitch_val)

        # Rotate local camera rays into world spherical coordinate frame
        R = self._get_rotation_matrix(yaw_val, pitch_val)
        world_rays = self._local_rays @ R.T  # (N_rays, 3)

        wx = world_rays[:, 0]
        wy = world_rays[:, 1]
        wz = world_rays[:, 2]

        # Convert 3D world unit vectors to spherical coordinates:
        # pitch = arcsin(wy), yaw = arctan2(wx, wz)
        ray_pitch_deg = np.degrees(np.arcsin(np.clip(wy, -1.0, 1.0)))
        ray_yaw_deg = np.degrees(np.arctan2(wx, wz))

        # Vectorized mapping to ERP pixels
        norm_yaw = ((ray_yaw_deg + 180.0) % 360.0) - 180.0
        pitch_clamped = np.clip(ray_pitch_deg, -90.0, 90.0)

        px = (norm_yaw + 180.0) / 360.0 * self.grid.frame_width
        py = (90.0 - pitch_clamped) / 180.0 * self.grid.frame_height

        cols = np.clip((px // self.grid.tile_width).astype(int), 0, self.grid.num_cols - 1)
        rows = np.clip((py // self.grid.tile_height).astype(int), 0, self.grid.num_rows - 1)
        tile_ids = rows * self.grid.num_cols + cols

        # Count occurrences and compute sampled ray-based overlap weights
        unique_tiles, counts = np.unique(tile_ids, return_counts=True)
        tile_weights = {int(tid): float(cnt / self.num_rays) for tid, cnt in zip(unique_tiles, counts)}
        active_tiles = sorted(list(tile_weights.keys()))

        # Compute 1-ring margin tiles surrounding all active tiles
        margin_set: Set[int] = set()
        for tid in active_tiles:
            for neighbor in self.grid.get_adjacent_tiles(tid, include_diagonals=True):
                if neighbor not in tile_weights:
                    margin_set.add(neighbor)
        margin_tiles = sorted(list(margin_set))

        return ViewportIntersection(
            center_yaw_deg=yaw_val,
            center_pitch_deg=pitch_val,
            hfov_deg=self.hfov_deg,
            vfov_deg=self.vfov_deg,
            active_tiles=active_tiles,
            tile_weights=tile_weights,
            primary_tile=primary_tile,
            margin_tiles=margin_tiles
        )

    def evaluate_tile_prediction(
        self,
        actual_yaw_deg: float,
        actual_pitch_deg: float,
        pred_yaw_deg: float,
        pred_pitch_deg: float,
        include_margin: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluates tile prediction accuracy between ground-truth and predicted viewports.

        Parameters:
            actual_yaw_deg, actual_pitch_deg: True future head orientation.
            pred_yaw_deg, pred_pitch_deg: Model predicted head orientation.
            include_margin: Whether predicted set includes 1-ring guard band margin tiles.

        Returns:
            Dictionary with Hit Ratio (Recall), Precision, Sampled Viewport Coverage, and Tile lists.
        """
        gt_inter = self.get_viewport_intersection(actual_yaw_deg, actual_pitch_deg)
        pred_inter = self.get_viewport_intersection(pred_yaw_deg, pred_pitch_deg)

        actual_set = set(gt_inter.active_tiles)
        pred_set = set(pred_inter.active_tiles)
        if include_margin:
            pred_set.update(pred_inter.margin_tiles)

        intersection = actual_set.intersection(pred_set)

        # Tile Hit Ratio (Recall): Fraction of required actual tiles successfully predicted/fetched
        recall = len(intersection) / len(actual_set) if actual_set else 1.0

        # Tile Precision: Fraction of predicted/fetched tiles that were actually visible
        precision = len(intersection) / len(pred_set) if pred_set else 1.0

        # Sampled Viewport Coverage: Percentage of user's field-of-view covered by fetched tiles
        # (weighted by ray fraction so central pixels matter proportionally)
        coverage = sum(gt_inter.tile_weights.get(tid, 0.0) for tid in pred_set)

        return {
            "tile_hit_ratio_recall": float(recall),
            "tile_precision": float(precision),
            "viewport_coverage": float(coverage),
            "num_actual_tiles": len(actual_set),
            "num_predicted_tiles": len(pred_set),
            "actual_tiles": sorted(list(actual_set)),
            "predicted_tiles": sorted(list(pred_set)),
            "hit_tiles": sorted(list(intersection)),
            "missed_tiles": sorted(list(actual_set - pred_set)),
            "wasted_tiles": sorted(list(pred_set - actual_set)),
        }

    @staticmethod
    def run_sampling_sensitivity_experiment(
        test_orientations: Optional[List[Tuple[float, float]]] = None,
        resolutions: List[int] = [21, 41, 81]
    ) -> pd.DataFrame:
        """
        Runs ray sampling resolution sensitivity benchmark across resolutions (e.g. 21x21, 41x41, 81x81).
        Compares active tile count, max weight difference, and coverage stability.
        """
        if test_orientations is None:
            test_orientations = [
                (0.0, 0.0),       # Equator center
                (30.0, 22.5),     # Tile boundary vicinity
                (-120.0, -45.0),  # Nadir edge
                (179.0, 10.0),    # Wrap-around seam
                (45.0, 70.0),     # Near-polar latitude
            ]

        results = []
        mappers = {r: ViewportMapper(ray_samples_per_axis=r) for r in resolutions}
        baseline_res = resolutions[0]  # e.g. 21x21

        for yaw, pitch in test_orientations:
            base_inter = mappers[baseline_res].get_viewport_intersection(yaw, pitch)
            for r in resolutions:
                mapper = mappers[r]
                inter = mapper.get_viewport_intersection(yaw, pitch)

                # Weight difference against baseline on common tiles
                all_tiles = set(base_inter.tile_weights.keys()) | set(inter.tile_weights.keys())
                max_w_diff = max(
                    abs(base_inter.tile_weights.get(t, 0.0) - inter.tile_weights.get(t, 0.0))
                    for t in all_tiles
                )

                results.append({
                    "yaw_deg": yaw,
                    "pitch_deg": pitch,
                    "resolution": f"{r}x{r} ({r*r} rays)",
                    "num_rays": r * r,
                    "active_tile_count": len(inter.active_tiles),
                    "primary_tile": inter.primary_tile,
                    "max_weight_diff_vs_21x21": float(max_w_diff),
                    "active_tiles_match_baseline": (set(inter.active_tiles) == set(base_inter.active_tiles))
                })

        return pd.DataFrame(results)


if __name__ == "__main__":
    mapper = ViewportMapper()
    center_inter = mapper.get_viewport_intersection(0.0, 0.0)
    print("=== Viewport Intersection at (yaw=0°, pitch=0°) ===")
    print(f"Primary Tile: {center_inter.primary_tile}")
    print(f"Active Tiles ({len(center_inter.active_tiles)}): {center_inter.active_tiles}")
    print("Weights:", center_inter.tile_weights)
    print("Weight Sum:", sum(center_inter.tile_weights.values()))

    print("\n=== Running Ray Sampling Sensitivity Experiment ===")
    df_sens = ViewportMapper.run_sampling_sensitivity_experiment()
    print(df_sens.to_string(index=False))
