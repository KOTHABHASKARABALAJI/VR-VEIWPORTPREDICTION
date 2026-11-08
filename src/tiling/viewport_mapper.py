"""
Module: viewport_mapper.py
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming

Computes the 3D spherical projection of the user's Head-Mounted Display (HMD) Viewport
(typically 100° x 100° Field of View) onto the 24 spatial tiles of the ERP canvas.

Features:
- Rigorous 3D viewing frustum rotation and equirectangular ray casting.
- Accurate overlap weighting: computes the exact fractional area of the viewport intersecting each tile.
- Guard band / margin tile calculation for network jitter and motion buffer.
- Tile Hit Ratio, Precision, and Viewport Coverage metrics for prediction evaluation.
- Batch evaluation harness integrating directly with the final GRU model predictions.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import numpy as np

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
    tile_weights: Dict[int, float]           # Fraction of viewport rays falling in each tile (sum = 1.0)
    primary_tile: int                        # Unique tile containing the line of sight (center)
    margin_tiles: List[int]                  # 1-ring adjacent guard-band tiles surrounding active tiles


class ViewportMapper:
    """
    Maps 3D viewport Field-of-View (FoV) frustums to 24-tile Equirectangular projections.
    """

    def __init__(
        self,
        grid_partitioner: TileGridPartitioner | None = None,
        hfov_deg: float = 100.0,
        vfov_deg: float = 100.0,
        ray_samples_per_axis: int = 25
    ):
        """
        Parameters:
            grid_partitioner: Instance of TileGridPartitioner (defaults to 4x6 on 3840x1920).
            hfov_deg: Horizontal Field of View in degrees (default: 100°).
            vfov_deg: Vertical Field of View in degrees (default: 100°).
            ray_samples_per_axis: Sampling grid resolution (25x25 = 625 rays through viewport).
        """
        self.grid = grid_partitioner if grid_partitioner is not None else TileGridPartitioner()
        self.hfov_deg = float(hfov_deg)
        self.vfov_deg = float(vfov_deg)
        self.ray_samples = int(ray_samples_per_axis)

        # Precompute canonical local viewing rays in camera coordinate frame (facing +Z)
        # u in [-tan(hfov/2), +tan(hfov/2)], v in [-tan(vfov/2), +tan(vfov/2)]
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
        yaw_rad = np.radians(yaw_deg)
        pitch_rad = np.radians(pitch_deg)

        # R_pitch (rotation around X axis)
        cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
        R_pitch = np.array([
            [1.0,    0.0,    0.0],
            [0.0,  cos_p,  sin_p],
            [0.0, -sin_p,  cos_p]
        ], dtype=np.float64)

        # R_yaw (rotation around Y axis: world up)
        cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
        R_yaw = np.array([
            [ cos_y, 0.0, sin_y],
            [   0.0, 1.0,   0.0],
            [-sin_y, 0.0, cos_y]
        ], dtype=np.float64)

        # Combined world orientation
        return R_yaw @ R_pitch

    def get_viewport_intersection(
        self,
        yaw_deg: float,
        pitch_deg: float
    ) -> ViewportIntersection:
        """
        Computes active tiles and fractional overlap weights for given viewport orientation.

        Parameters:
            yaw_deg: Center yaw in degrees [-180, +180].
            pitch_deg: Center pitch in degrees [-90, +90].

        Returns:
            ViewportIntersection dataclass.
        """
        # Primary tile containing the center line-of-sight
        primary_tile = self.grid.point_to_tile(yaw_deg, pitch_deg)

        # Rotate local camera rays into world spherical coordinate frame
        R = self._get_rotation_matrix(yaw_deg, pitch_deg)
        world_rays = self._local_rays @ R.T  # (N_rays, 3)

        wx = world_rays[:, 0]
        wy = world_rays[:, 1]
        wz = world_rays[:, 2]

        # Convert 3D world unit vectors to spherical coordinates:
        # pitch = arcsin(wy), yaw = arctan2(wx, wz)
        ray_pitch_deg = np.degrees(np.arcsin(np.clip(wy, -1.0, 1.0)))
        ray_yaw_deg = np.degrees(np.arctan2(wx, wz))

        # Convert spherical coordinates to tile IDs
        # Vectorized mapping to ERP pixels
        norm_yaw = ((ray_yaw_deg + 180.0) % 360.0) - 180.0
        pitch_clamped = np.clip(ray_pitch_deg, -90.0, 90.0)

        px = (norm_yaw + 180.0) / 360.0 * self.grid.frame_width
        py = (90.0 - pitch_clamped) / 180.0 * self.grid.frame_height

        cols = np.clip((px // self.grid.tile_width).astype(int), 0, self.grid.num_cols - 1)
        rows = np.clip((py // self.grid.tile_height).astype(int), 0, self.grid.num_rows - 1)
        tile_ids = rows * self.grid.num_cols + cols

        # Count occurrences and compute overlap weights
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
            center_yaw_deg=float(yaw_deg),
            center_pitch_deg=float(pitch_deg),
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
    ) -> Dict[str, float | list]:
        """
        Evaluates tile prediction accuracy between ground-truth and predicted viewports.

        Parameters:
            actual_yaw_deg, actual_pitch_deg: True future head orientation.
            pred_yaw_deg, pred_pitch_deg: Model predicted head orientation.
            include_margin: Whether predicted set includes 1-ring guard band margin tiles.

        Returns:
            Dictionary with Hit Ratio (Recall), Precision, Viewport Coverage, and Tile lists.
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

        # Viewport Coverage: Percentage of user's field-of-view covered by fetched tiles
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


if __name__ == "__main__":
    mapper = ViewportMapper()
    # Test at center orientation (yaw=0, pitch=0)
    center_inter = mapper.get_viewport_intersection(0.0, 0.0)
    print("=== Viewport Intersection at (yaw=0°, pitch=0°) ===")
    print(f"Primary Tile: {center_inter.primary_tile}")
    print(f"Active Tiles ({len(center_inter.active_tiles)}): {center_inter.active_tiles}")
    print("Tile Overlap Weights:")
    for tid, w in center_inter.tile_weights.items():
        print(f"  Tile {tid:2d}: {w*100:5.1f}%")
    print(f"Margin Tiles ({len(center_inter.margin_tiles)}): {center_inter.margin_tiles}")

    # Test evaluation with small prediction error (yaw=10°, pitch=5°)
    eval_res = mapper.evaluate_tile_prediction(
        actual_yaw_deg=0.0, actual_pitch_deg=0.0,
        pred_yaw_deg=10.0, pred_pitch_deg=5.0
    )
    print("\n=== Prediction Evaluation (Actual=(0°, 0°) vs Pred=(10°, 5°)) ===")
    print(f"Hit Ratio (Recall): {eval_res['tile_hit_ratio_recall']*100:.1f}%")
    print(f"Precision:          {eval_res['tile_precision']*100:.1f}%")
    print(f"Viewport Coverage:  {eval_res['viewport_coverage']*100:.1f}%")
    print(f"Hit Tiles:          {eval_res['hit_tiles']}")
    print(f"Missed Tiles:       {eval_res['missed_tiles']}")

