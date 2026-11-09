"""
Module: features.py
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming

Purpose:
    Implements strictly causal 7-feature kinematic extraction and trigonometric
    continuous encoding for 360° video viewport trajectories:
    - Feature 0: sin_yaw = sin(radians(yaw))
    - Feature 1: cos_yaw = cos(radians(yaw))
    - Feature 2: pitch (degrees)
    - Feature 3: yaw_vel = causal backward difference d(yaw)/dt (deg/s)
    - Feature 4: pitch_vel = causal backward difference d(pitch)/dt (deg/s)
    - Feature 5: yaw_acc = causal backward difference d(yaw_vel)/dt (deg/s²)
    - Feature 6: pitch_acc = causal backward difference d(pitch_vel)/dt (deg/s²)

Mathematical & Causal Guarantees:
    - Strictly causal: all features at timestep t depend strictly on observations at timesteps <= t.
    - Zero future-information leakage: future frames never influence past or present features.
    - Seamless circular wrap-around: shortest-arc difference across the +/-180° yaw boundary.
    - Scientifically grounded boundary policy: acceleration at t=0 and t=1 is set to 0.0,
      avoiding artificial acceleration spikes from assuming unobserved prior velocity is zero.
    - Strict validation: rejects non-finite values, invalid pitch ranges, non-monotonic timestamps,
      and dimension mismatches.
"""

from typing import Tuple, Union
import numpy as np


def unwrap_yaw_degrees(yaw_degrees: np.ndarray) -> np.ndarray:
    """
    Unwraps circular yaw angle trajectory across the +/-180 degree boundary.

    Parameters:
        yaw_degrees (np.ndarray): 1D array of yaw angles in degrees [-180, +180].

    Returns:
        np.ndarray: Continuous unwrapped yaw trajectory in degrees.
    """
    yaw_rad = np.radians(yaw_degrees)
    unwrapped_rad = np.unwrap(yaw_rad)
    return np.degrees(unwrapped_rad)


def compute_sin_cos_yaw(yaw_degrees: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes continuous trigonometric 2D Cartesian coordinates [sin(yaw), cos(yaw)].

    Eliminates angular boundary discontinuity at +/-180° for neural network inputs.

    Parameters:
        yaw_degrees (np.ndarray): 1D array of yaw angles in degrees.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (sin_yaw, cos_yaw) arrays.
    """
    yaw_rad = np.radians(yaw_degrees)
    return np.sin(yaw_rad), np.cos(yaw_rad)


def compute_circular_difference(angles_degrees: np.ndarray) -> np.ndarray:
    """
    Computes the shortest signed angular difference between consecutive angles on the circle.

    Formula:
        diff = ((angles[t] - angles[t-1]) + 180.0) % 360.0 - 180.0

    Tie-Breaking Policy:
        For an exact 180.0° difference (opposite poles of the circle), the formula
        evaluates to -180.0° (deterministic clockwise tie-breaking).

    Parameters:
        angles_degrees (np.ndarray): 1D array of angles in degrees.

    Returns:
        np.ndarray: Shortest signed angular step array of length (len(angles_degrees) - 1).
    """
    diff = angles_degrees[1:] - angles_degrees[:-1]
    return (diff + 180.0) % 360.0 - 180.0


def compute_derivatives(
    values: np.ndarray,
    dt: float | np.ndarray = 0.10091,
    is_angular: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates first-order velocity and second-order acceleration using
    STRICTLY CAUSAL backward differences.

    Mathematical Formulation:
        Velocity:
            vel[0] = 0.0          (no prior observation at t < 0)
            vel[t] = (values[t] - values[t-1]) / dt[t-1], for t >= 1

        Acceleration:
            acc[0] = 0.0          (no prior velocity at t < 0)
            acc[1] = 0.0          (no prior velocity observation before t=0; avoids artificial
                                   spike resulting from assuming unobserved vel[-1] = 0)
            acc[t] = (vel[t] - vel[t-1]) / dt[t-1], for t >= 2

    Parameters:
        values (np.ndarray): 1D array of kinematic observations (length N).
        dt (float | np.ndarray): Positive sampling interval (scalar or length N - 1 array).
        is_angular (bool): If True, applies circular shortest-distance difference for yaw wrap-around.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (velocity, acceleration) arrays of length N.
    """
    vals = np.asarray(values, dtype=np.float64)
    n = len(vals)
    vel = np.zeros(n, dtype=np.float64)
    acc = np.zeros(n, dtype=np.float64)

    if n <= 1:
        return vel, acc

    if isinstance(dt, bool):
        raise TypeError("dt cannot be a boolean value.")
    if np.isscalar(dt):
        dt_val = float(dt)
        if not np.isfinite(dt_val) or dt_val <= 0.0:
            raise ValueError(f"dt must be positive and finite, got {dt}")
        dt_arr = np.full(n - 1, dt_val, dtype=np.float64)
    else:
        dt_raw = np.asarray(dt, dtype=np.float64)
        if dt_raw.ndim != 1 or len(dt_raw) != n - 1:
            raise ValueError(f"Array dt must be 1D with length {n - 1}, got shape {dt_raw.shape}")
        if not np.all(np.isfinite(dt_raw)) or np.any(dt_raw <= 0.0):
            raise ValueError("Array dt must contain strictly positive and finite values.")
        dt_arr = dt_raw

    # 1. First derivative: Backward difference velocity
    if is_angular:
        delta_v = compute_circular_difference(vals)
    else:
        delta_v = vals[1:] - vals[:-1]

    vel[1:] = delta_v / dt_arr

    # 2. Second derivative: Backward difference acceleration
    # Boundary: at t=1, velocity vel[0]=0 is an initialization placeholder, not a measured
    # physical velocity. Calculating vel[1] - vel[0] would produce an artificial acceleration spike.
    # Therefore, acc[0] = 0.0 and acc[1] = 0.0.
    if n >= 3:
        delta_a = vel[2:] - vel[1:-1]
        acc[2:] = delta_a / dt_arr[1:]

    return vel, acc


def extract_7_features(
    yaw_degrees: np.ndarray,
    pitch_degrees: np.ndarray,
    dt: float | np.ndarray | None = None,
    timestamps: np.ndarray | None = None
) -> np.ndarray:
    """
    Extracts the standard 7-feature matrix for a viewing trajectory using strictly causal differences.

    Features:
        Index 0: sin_yaw = sin(radians(yaw))
        Index 1: cos_yaw = cos(radians(yaw))
        Index 2: pitch (degrees)
        Index 3: yaw_vel = causal backward difference d(yaw)/dt (deg/s)
        Index 4: pitch_vel = causal backward difference d(pitch)/dt (deg/s)
        Index 5: yaw_acc = causal backward difference d(yaw_vel)/dt (deg/s²)
        Index 6: pitch_acc = causal backward difference d(pitch_vel)/dt (deg/s²)

    Parameters:
        yaw_degrees (np.ndarray): Raw yaw angles in degrees [-180, +180].
        pitch_degrees (np.ndarray): Raw pitch angles in degrees [-90, +90].
        dt (float | np.ndarray | None): Fixed sampling interval (scalar or length N-1 array).
            Mutually exclusive with timestamps.
        timestamps (np.ndarray | None): Trajectory timestamps in seconds (length N array).
            Must be strictly monotonically increasing. Mutually exclusive with dt.

    Returns:
        np.ndarray: Feature matrix of shape (T, 7) and dtype float32.

    Raises:
        TypeError: If input types are invalid.
        ValueError: If input dimensions, pitch boundaries, or timestamp properties are violated.
    """
    yaw_arr = np.asarray(yaw_degrees, dtype=np.float64)
    pitch_arr = np.asarray(pitch_degrees, dtype=np.float64)

    # Dimensionality validation
    if yaw_arr.ndim != 1 or pitch_arr.ndim != 1:
        raise ValueError(
            f"yaw_degrees and pitch_degrees must be 1-dimensional, got yaw.ndim={yaw_arr.ndim}, pitch.ndim={pitch_arr.ndim}"
        )

    if len(yaw_arr) != len(pitch_arr):
        raise ValueError(
            f"yaw_degrees and pitch_degrees must have equal length, got yaw={len(yaw_arr)}, pitch={len(pitch_arr)}"
        )

    n = len(yaw_arr)
    if n == 0:
        return np.empty((0, 7), dtype=np.float32)

    # Finite values validation
    if not np.all(np.isfinite(yaw_arr)):
        raise ValueError("yaw_degrees contains non-finite values (NaN or Inf).")
    if not np.all(np.isfinite(pitch_arr)):
        raise ValueError("pitch_degrees contains non-finite values (NaN or Inf).")

    # Domain validation for pitch [-90.0, +90.0]
    if np.any(pitch_arr < -90.001) or np.any(pitch_arr > 90.001):
        raise ValueError(
            f"pitch_degrees must be within [-90.0°, 90.0°], got min={pitch_arr.min():.4f}, max={pitch_arr.max():.4f}"
        )

    # Resolve timestamps vs dt parameters
    if timestamps is not None and dt is not None:
        raise ValueError("Provide either timestamps or dt, not both.")

    if timestamps is not None:
        ts_arr = np.asarray(timestamps, dtype=np.float64)
        if ts_arr.ndim != 1:
            raise ValueError(f"timestamps must be 1-dimensional, got ndim={ts_arr.ndim}")
        if len(ts_arr) != n:
            raise ValueError(f"timestamps length ({len(ts_arr)}) must match yaw/pitch length ({n})")
        if not np.all(np.isfinite(ts_arr)):
            raise ValueError("timestamps contains non-finite values (NaN or Inf).")

        if n > 1:
            dts = ts_arr[1:] - ts_arr[:-1]
            if np.any(dts <= 0.0):
                bad_idx = np.where(dts <= 0.0)[0][0]
                raise ValueError(
                    f"timestamps must be strictly monotonically increasing (dt > 0). "
                    f"Violation at index {bad_idx}: t[{bad_idx}]={ts_arr[bad_idx]}, t[{bad_idx+1}]={ts_arr[bad_idx+1]}, dt={dts[bad_idx]}"
                )
            dt_arr = dts
        else:
            dt_arr = np.empty(0, dtype=np.float64)
    elif dt is not None:
        if isinstance(dt, bool):
            raise TypeError("dt cannot be a boolean value.")
        if np.isscalar(dt):
            dt_val = float(dt)
            if not np.isfinite(dt_val) or dt_val <= 0.0:
                raise ValueError(f"Scalar dt must be positive and finite, got {dt}")
            dt_arr = np.full(n - 1, dt_val, dtype=np.float64)
        else:
            dt_raw = np.asarray(dt, dtype=np.float64)
            if dt_raw.ndim != 1 or len(dt_raw) != n - 1:
                raise ValueError(
                    f"Array dt must be 1-dimensional with length {n - 1}, got shape {dt_raw.shape}"
                )
            if not np.all(np.isfinite(dt_raw)) or np.any(dt_raw <= 0.0):
                raise ValueError("Array dt must contain strictly positive and finite values.")
            dt_arr = dt_raw
    else:
        # Default nominal sampling interval (~10 Hz AVTrack360)
        dt_arr = np.full(n - 1, 0.10091, dtype=np.float64)

    sin_yaw, cos_yaw = compute_sin_cos_yaw(yaw_arr)
    yaw_vel, yaw_acc = compute_derivatives(yaw_arr, dt_arr, is_angular=True)
    pitch_vel, pitch_acc = compute_derivatives(pitch_arr, dt_arr, is_angular=False)

    features = np.column_stack([
        sin_yaw,
        cos_yaw,
        pitch_arr,
        yaw_vel,
        pitch_vel,
        yaw_acc,
        pitch_acc
    ])

    return features.astype(np.float32)


def standardize_features(
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray
) -> np.ndarray:
    """
    Applies standard Z-score normalization: X_norm = (X - mean) / std.

    Parameters:
        features (np.ndarray): Input feature array of shape (..., 7).
        mean (np.ndarray): Feature-wise mean vector of shape (7,).
        std (np.ndarray): Feature-wise standard deviation vector of shape (7,).

    Returns:
        np.ndarray: Standardized feature array of identical shape.
    """
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)

    if mean_arr.shape != (7,) or std_arr.shape != (7,):
        raise ValueError(f"Normalization vectors must have shape (7,), got mean={mean_arr.shape}, std={std_arr.shape}")
    if np.any(std_arr <= 0.0) or not np.all(np.isfinite(std_arr)):
        raise ValueError("Normalization standard deviations must be strictly positive and finite.")

    return (features - mean_arr) / std_arr


def normalize_pitch(
    pitch_degrees: np.ndarray,
    pitch_mean: float,
    pitch_std: float
) -> np.ndarray:
    """
    Normalizes pitch angles using Z-score standardization: (pitch - mean) / std.
    """
    if pitch_std <= 0.0 or not np.isfinite(pitch_std) or not np.isfinite(pitch_mean):
        raise ValueError(f"pitch_std must be positive and finite, got std={pitch_std}, mean={pitch_mean}")
    return (pitch_degrees - pitch_mean) / pitch_std


def denormalize_pitch(
    norm_pitch: np.ndarray,
    pitch_mean: float,
    pitch_std: float
) -> np.ndarray:
    """
    Denormalizes normalized pitch angles back to physical degrees: norm_pitch * std + mean.
    """
    if pitch_std <= 0.0 or not np.isfinite(pitch_std) or not np.isfinite(pitch_mean):
        raise ValueError(f"pitch_std must be positive and finite, got std={pitch_std}, mean={pitch_mean}")
    return norm_pitch * pitch_std + pitch_mean
