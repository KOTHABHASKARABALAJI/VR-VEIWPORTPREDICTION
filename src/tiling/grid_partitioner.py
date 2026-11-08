"""
Module: grid_partitioner.py
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming

Defines the mathematical 24-tile spatial partitioning grid (4 rows x 6 columns) on
Equirectangular Projection (ERP) 360° video canvases (canonical 3840 x 1920 resolution).

Provides:
- Exact pixel coordinate bounding boxes for each tile (640 x 480 pixels).
- Spherical coordinate angular boundaries (yaw in [-180, +180], pitch in [-90, +90]).
- Solid angle (steradian) computation accounting for equirectangular latitude compression.
- Point-to-tile and pixel-to-spherical bidirectional projection mappings.
- Circular boundary wrap-around topological neighbor lookup.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass(frozen=True)
class TileSpec:
    """
    Specification and geometric boundaries of an individual spatial tile on the ERP canvas.
    """
    tile_id: int              # Global linear tile index [0, 23]
    row: int                  # Latitude row index [0, 3] (0 = zenith / top, 3 = nadir / bottom)
    col: int                  # Longitude column index [0, 5] (0 = -180° to -120°)
    x_min: int                # Left pixel boundary [0, 3840)
    x_max: int                # Right pixel boundary (exclusive) [0, 3840]
    y_min: int                # Top pixel boundary [0, 1920)
    y_max: int                # Bottom pixel boundary (exclusive) [0, 1920]
    yaw_min_deg: float        # Minimum yaw angle in degrees [-180.0, 180.0)
    yaw_max_deg: float        # Maximum yaw angle in degrees (-180.0, 180.0]
    yaw_center_deg: float     # Geometric center yaw in degrees
    pitch_min_deg: float      # Minimum pitch angle in degrees [-90.0, 90.0)
    pitch_max_deg: float      # Maximum pitch angle in degrees (-90.0, 90.0]
    pitch_center_deg: float   # Geometric center pitch in degrees
    solid_angle_sr: float     # True spherical area in steradians (integral over sphere)
    area_fraction: float      # Fraction of total 4π sphere covered by tile


class TileGridPartitioner:
    """
    Partitions a 360° Equirectangular video into a standard 4x6 spatial grid (24 tiles).
    """

    def __init__(
        self,
        frame_width: int = 3840,
        frame_height: int = 1920,
        num_rows: int = 4,
        num_cols: int = 6
    ):
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.num_rows = int(num_rows)
        self.num_cols = int(num_cols)
        self.total_tiles = self.num_rows * self.num_cols

        if self.frame_width % self.num_cols != 0:
            raise ValueError(f"Width {self.frame_width} must be divisible by {self.num_cols} cols.")
        if self.frame_height % self.num_rows != 0:
            raise ValueError(f"Height {self.frame_height} must be divisible by {self.num_rows} rows.")

        self.tile_width = self.frame_width // self.num_cols    # 640 px
        self.tile_height = self.frame_height // self.num_rows  # 480 px

        self.yaw_span_deg = 360.0 / self.num_cols              # 60.0°
        self.pitch_span_deg = 180.0 / self.num_rows            # 45.0°

        self.tiles: Dict[int, TileSpec] = {}
        self._build_grid()

    def _build_grid(self) -> None:
        """Constructs and validates all 24 TileSpec definitions."""
        total_sphere_sr = 4.0 * np.pi

        for r in range(self.num_rows):
            # Row 0 is top (zenith, pitch = +90° down to +45°)
            # Row 3 is bottom (nadir, pitch = -45° down to -90°)
            pitch_max = 90.0 - r * self.pitch_span_deg
            pitch_min = 90.0 - (r + 1) * self.pitch_span_deg
            pitch_center = (pitch_max + pitch_min) / 2.0

            # Pixel rows: top = 0, bottom = height
            y_min = r * self.tile_height
            y_max = (r + 1) * self.tile_height

            # Spherical solid angle integral:
            # Omega = (yaw_max - yaw_min)_rad * (sin(pitch_max_rad) - sin(pitch_min_rad))
            delta_lambda_rad = np.radians(self.yaw_span_deg)
            sin_p_max = np.sin(np.radians(pitch_max))
            sin_p_min = np.sin(np.radians(pitch_min))
            solid_angle = float(delta_lambda_rad * (sin_p_max - sin_p_min))
            area_frac = solid_angle / total_sphere_sr

            for c in range(self.num_cols):
                tile_id = r * self.num_cols + c
                yaw_min = -180.0 + c * self.yaw_span_deg
                yaw_max = -180.0 + (c + 1) * self.yaw_span_deg
                yaw_center = -180.0 + (c + 0.5) * self.yaw_span_deg

                x_min = c * self.tile_width
                x_max = (c + 1) * self.tile_width

                self.tiles[tile_id] = TileSpec(
                    tile_id=tile_id,
                    row=r,
                    col=c,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    yaw_min_deg=round(yaw_min, 4),
                    yaw_max_deg=round(yaw_max, 4),
                    yaw_center_deg=round(yaw_center, 4),
                    pitch_min_deg=round(pitch_min, 4),
                    pitch_max_deg=round(pitch_max, 4),
                    pitch_center_deg=round(pitch_center, 4),
                    solid_angle_sr=round(solid_angle, 6),
                    area_fraction=round(area_frac, 6)
                )

    def get_tile(self, tile_id: int) -> TileSpec:
        """Retrieves metadata and bounding boxes for a given tile ID."""
        if tile_id not in self.tiles:
            raise KeyError(f"Invalid tile_id {tile_id}. Must be in [0, {self.total_tiles - 1}].")
        return self.tiles[tile_id]

    def spherical_to_pixel(self, yaw_deg: float, pitch_deg: float) -> Tuple[float, float]:
        """
        Maps continuous spherical angles (yaw in [-180, 180], pitch in [-90, 90])
        to continuous ERP canvas pixel coordinates (x in [0, W], y in [0, H]).
        """
        # Wrap yaw into [-180, +180)
        norm_yaw = ((yaw_deg + 180.0) % 360.0) - 180.0
        pitch_clamped = np.clip(pitch_deg, -90.0, 90.0)

        # X: longitude -180 -> 0, +180 -> frame_width
        x = (norm_yaw + 180.0) / 360.0 * self.frame_width
        # Y: latitude +90 (zenith) -> 0, -90 (nadir) -> frame_height
        y = (90.0 - pitch_clamped) / 180.0 * self.frame_height

        return float(x), float(y)

    def pixel_to_spherical(self, x: float, y: float) -> Tuple[float, float]:
        """
        Maps continuous ERP pixel coordinates to spherical angles (yaw, pitch).
        """
        x_clamped = np.clip(x, 0.0, self.frame_width)
        y_clamped = np.clip(y, 0.0, self.frame_height)

        yaw = (x_clamped / self.frame_width) * 360.0 - 180.0
        pitch = 90.0 - (y_clamped / self.frame_height) * 180.0

        return float(yaw), float(pitch)

    def point_to_tile(self, yaw_deg: float, pitch_deg: float) -> int:
        """
        Finds the unique tile containing the given spherical direction vector.
        """
        x, y = self.spherical_to_pixel(yaw_deg, pitch_deg)
        return self.pixel_to_tile(x, y)

    def pixel_to_tile(self, x: float, y: float) -> int:
        """
        Finds the unique tile containing the given pixel (x, y).
        """
        col = int(x // self.tile_width)
        row = int(y // self.tile_height)

        # Handle boundary edges
        col = min(max(col, 0), self.num_cols - 1)
        row = min(max(row, 0), self.num_rows - 1)

        return row * self.num_cols + col

    def get_adjacent_tiles(self, tile_id: int, include_diagonals: bool = True) -> List[int]:
        """
        Returns adjacent neighbor tile IDs, correctly handling circular 360° longitude wrapping.
        Top and bottom rows do not wrap vertically (poles terminate at Zenith and Nadir).
        """
        spec = self.get_tile(tile_id)
        r, c = spec.row, spec.col
        neighbors = []

        # Row offsets (-1, 0, +1), Col offsets (-1, 0, +1)
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if include_diagonals:
            offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

        for dr, dc in offsets:
            nr = r + dr
            # Vertical clamp: cannot exceed Zenith (row 0) or Nadir (row num_rows-1)
            if 0 <= nr < self.num_rows:
                # Horizontal circular wrap: -1 wraps to num_cols-1, num_cols wraps to 0
                nc = (c + dc) % self.num_cols
                neighbor_id = nr * self.num_cols + nc
                neighbors.append(neighbor_id)

        return sorted(list(set(neighbors)))

    def summary(self) -> str:
        """Returns a formatted tabular summary of the 24 tile grid geometry."""
        lines = [
            "=" * 90,
            f" 24-TILE SPATIAL PARTITIONING GRID ({self.num_rows}x{self.num_cols} on {self.frame_width}x{self.frame_height} ERP Canvas)",
            "=" * 90,
            f"{'ID':>3} | {'Row':>3} {'Col':>3} | {'Pixel Box (Xmin,Ymin -> Xmax,Ymax)':^34} | {'Yaw Range':^15} | {'Pitch Range':^13} | {'Solid Angle':^10}",
            "-" * 90
        ]
        for tid in range(self.total_tiles):
            t = self.tiles[tid]
            pbox = f"({t.x_min:4d},{t.y_min:4d}) -> ({t.x_max:4d},{t.y_max:4d})"
            yrange = f"[{t.yaw_min_deg:+6.1f}°, {t.yaw_max_deg:+6.1f}°]"
            prange = f"[{t.pitch_min_deg:+5.1f}°, {t.pitch_max_deg:+5.1f}°]"
            lines.append(
                f"{t.tile_id:3d} | {t.row:3d} {t.col:3d} | {pbox:^34} | {yrange:^15} | {prange:^13} | {t.solid_angle_sr:8.4f} sr"
            )
        lines.append("=" * 90)
        return "\n".join(lines)


if __name__ == "__main__":
    grid = TileGridPartitioner()
    print(grid.summary())

