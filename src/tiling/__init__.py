"""
Package: src.tiling
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming

Purpose:
    Provides modules for spatial grid partitioning of Equirectangular Projection (ERP)
    360° video canvases, spherical-to-planar tile coordinate mapping, predicted viewport
    intersection calculations, and adaptive bitrate allocation optimization under network
    bandwidth constraints.

Modules (Phase 5):
    - grid_partitioner: Splits ERP canvases into M x N rectangular tiles (e.g., 4 x 6 = 24 tiles).
    - viewport_mapper: Calculates spherical FoV (Field of View) overlap with spatial tiles.
    - adaptive_allocator: Solves the bitrate utility maximization problem for tile streams.
"""

from .grid_partitioner import TileGridPartitioner, TileSpec
from .viewport_mapper import ViewportMapper, ViewportIntersection

__all__ = [
    "TileGridPartitioner",
    "TileSpec",
    "ViewportMapper",
    "ViewportIntersection",
]


