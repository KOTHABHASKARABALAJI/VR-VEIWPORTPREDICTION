"""
Feature engineering module for 360° viewport trajectory prediction:
- unwrap_yaw_degrees: Unwraps yaw trajectory to eliminate circular jumps.
- compute_sin_cos_yaw: Trigonometric sine and cosine continuous encoding.
- compute_derivatives: Angular velocity and acceleration computation.
- extract_7_features: Constructs the standard 7-feature kinematic matrix.
- standardize_features: Z-score normalization using training split statistics.
"""

from .features import (
    unwrap_yaw_degrees,
    compute_sin_cos_yaw,
    compute_derivatives,
    extract_7_features,
    standardize_features,
)

__all__ = [
    "unwrap_yaw_degrees",
    "compute_sin_cos_yaw",
    "compute_derivatives",
    "extract_7_features",
    "standardize_features",
]
