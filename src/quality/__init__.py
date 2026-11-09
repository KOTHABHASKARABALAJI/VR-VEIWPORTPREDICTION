"""
Package: src.quality
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming
Phase: Phase 5, Day 2 — Adaptive Tile Quality Allocation System

Exposes quality level definitions, simulated bitrate specifications, numerical quality utility
abstractions, and tile quality allocation managers.
"""

from .quality_config import (
    QualityLevel,
    QualityTierSpec,
    TileQualityConfig,
    TileQualityAllocation,
)
from .uniform_allocator import (
    UniformQualityAllocator,
    UniformAllocationResult,
    InsufficientBandwidthError,
    allocate_uniform_quality,
)
from .viewport_allocator import (
    ViewportQualityAllocator,
    ViewportAllocationResult,
    allocate_viewport_quality,
)
from .neighbourhood_allocator import (
    TileNeighbourhoodFinder,
    NeighbourhoodQualityAllocator,
    NeighbourhoodAllocationResult,
    DEFAULT_NEIGHBOUR_FACTOR,
    allocate_neighbourhood_quality,
)

__all__ = [
    "QualityLevel",
    "QualityTierSpec",
    "TileQualityConfig",
    "TileQualityAllocation",
    "UniformQualityAllocator",
    "UniformAllocationResult",
    "InsufficientBandwidthError",
    "allocate_uniform_quality",
    "ViewportQualityAllocator",
    "ViewportAllocationResult",
    "allocate_viewport_quality",
    "TileNeighbourhoodFinder",
    "NeighbourhoodQualityAllocator",
    "NeighbourhoodAllocationResult",
    "DEFAULT_NEIGHBOUR_FACTOR",
    "allocate_neighbourhood_quality",
]
