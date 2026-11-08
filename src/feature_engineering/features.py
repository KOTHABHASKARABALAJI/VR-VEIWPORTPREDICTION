"""
Module: features.py
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming

Purpose:
    Implements the 7-feature kinematic extraction and trigonometric continuous encoding
    for 360° video viewport trajectory analysis:
    - Feature 1: sin_yaw = sin(radians(yaw))
    - Feature 2: cos_yaw = cos(radians(yaw))
    - Feature 3: pitch (degrees)
    - Feature 4: yaw_vel = d(unwrapped_yaw) / dt (deg/s)
    - Feature 5: pitch_vel = d(pitch) / dt (deg/s)
    - Feature 6: yaw_acc = d(yaw_vel) / dt (deg/s²)
    - Feature 7: pitch_acc = d(pitch_vel) / dt (deg/s²)
"""

import numpy as np


def unwrap_yaw_degrees(yaw_degrees: np.ndarray) -> np.ndarray:
    """
    Unwraps circular yaw angle trajectory across the +/-180 degree boundary.

    Eliminates artificial angular jumps when a user crosses the rear boundary
    (+179° to -179°), preventing derivative spikes during velocity calculation.

    Parameters:
        yaw_degrees (np.ndarray): 1D array of yaw angles in degrees [-180, +180].

    Returns:
        np.ndarray: Continuous unwrapped yaw trajectory in degrees.
    """
    yaw_rad = np.radians(yaw_degrees)
    unwrapped_rad = np.unwrap(yaw_rad)
    return np.degrees(unwrapped_rad)


def compute_sin_cos_yaw(yaw_degrees: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes continuous trigonometric 2D Cartesian coordinates [sin(yaw), cos(yaw)].

    Eliminates angular boundary discontinuity in recurrent neural network inputs.

    Parameters:
        yaw_degrees (np.ndarray): 1D array of yaw angles in degrees.

    Returns:
        tuple[np.ndarray, np.ndarray]: (sin_yaw, cos_yaw) arrays.
    """
    yaw_rad = np.radians(yaw_degrees)
    return np.sin(yaw_rad), np.cos(yaw_rad)


def compute_circular_difference(angles_degrees: np.ndarray) -> np.ndarray:
    """
    Computes shortest signed angular difference between consecutive angles on the circle [-180, +180].

    Correctly handles circular wrap-around across the +/-180° boundary:
    e.g., from +179° to -179° yields +2° (clockwise turn), avoiding artificial -358° spikes.

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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates first-order angular velocity and second-order angular acceleration
    using STRICTLY CAUSAL backward differences:
        vel[t] = (values[t] - values[t-1]) / dt[t-1]
        acc[t] = (vel[t] - vel[t-1]) / dt[t-1]
    
    Guarantees zero future-information leakage: the derivative at timestep t
    depends exclusively on observations at timesteps <= t.
    At boundary t=0, velocity and acceleration are initialized to 0.0.

    Parameters:
        values (np.ndarray): 1D array of kinematic observations (yaw or pitch).
        dt (float | np.ndarray): Sampling interval in seconds (scalar or 1D array of intervals/timestamps).
        is_angular (bool): If True, applies circular shortest-distance difference for yaw wrap-around.

    Returns:
        tuple[np.ndarray, np.ndarray]: (velocity in deg/s, acceleration in deg/s²).
    """
    vals = np.asarray(values, dtype=np.float64)
    n = len(vals)
    vel = np.zeros(n, dtype=np.float64)
    acc = np.zeros(n, dtype=np.float64)

    if n <= 1:
        return vel, acc

    # Parse sampling intervals dt safely
    if np.isscalar(dt):
        dt_val = 0.10091 if (dt is None or dt <= 0.0 or np.isnan(dt)) else float(dt)
        dt_arr = np.full(n - 1, dt_val, dtype=np.float64)
    else:
        dt_raw = np.asarray(dt, dtype=np.float64)
        if len(dt_raw) == n:
            # Provided as timestamps -> compute successive differences
            dt_arr = dt_raw[1:] - dt_raw[:-1]
        elif len(dt_raw) == n - 1:
            dt_arr = dt_raw.copy()
        else:
            dt_arr = np.full(n - 1, 0.10091, dtype=np.float64)
        # Protect against non-positive, zero, NaN or Inf intervals
        dt_arr = np.where((dt_arr <= 0.0) | np.isnan(dt_arr) | np.isinf(dt_arr), 0.10091, dt_arr)

    # 1. First derivative: Backward difference velocity
    if is_angular:
        delta_v = compute_circular_difference(vals)
    else:
        delta_v = vals[1:] - vals[:-1]

    vel[1:] = delta_v / dt_arr

    # 2. Second derivative: Backward difference acceleration (linear deg/s)
    delta_a = vel[1:] - vel[:-1]
    acc[1:] = delta_a / dt_arr

    return vel, acc


def extract_7_features(
    yaw_degrees: np.ndarray,
    pitch_degrees: np.ndarray,
    dt: float | np.ndarray = 0.10091,
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

    Guarantees:
        - Strictly causal: feature vector at time t depends only on frames <= t.
        - Circular wrap-around handled seamlessly across +/-180° boundary.
        - Robust against irregular timestamps and zero-interval duplicates.
        - Preserves identical feature ordering and output dimensions (T, 7).

    Parameters:
        yaw_degrees (np.ndarray): Raw yaw angles in degrees [-180, +180].
        pitch_degrees (np.ndarray): Raw pitch angles in degrees [-90, +90].
        dt (float | np.ndarray): Mean sampling interval (default: 0.10091s).
        timestamps (np.ndarray | None): Optional trajectory timestamps for exact dt calculation.

    Returns:
        np.ndarray: Feature matrix of shape (T, 7) and dtype float32.
    """
    yaw_arr = np.asarray(yaw_degrees, dtype=np.float64)
    pitch_arr = np.asarray(pitch_degrees, dtype=np.float64)
    n = len(yaw_arr)

    if n == 0:
        return np.empty((0, 7), dtype=np.float32)

    # Determine time interval specification
    time_spec = timestamps if timestamps is not None else dt

    sin_yaw, cos_yaw = compute_sin_cos_yaw(yaw_arr)
    yaw_vel, yaw_acc = compute_derivatives(yaw_arr, dt=time_spec, is_angular=True)
    pitch_vel, pitch_acc = compute_derivatives(pitch_arr, dt=time_spec, is_angular=False)

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
    return (features - mean) / std


def normalize_pitch(
    pitch_degrees: np.ndarray,
    pitch_mean: float = -1.259844,
    pitch_std: float = 12.062758
) -> np.ndarray:
    """
    Normalizes pitch angles using Z-score standardization: (pitch - mean) / std.

    Parameters:
        pitch_degrees (np.ndarray): Pitch angle(s) in degrees.
        pitch_mean (float): Mean pitch from training split (default: -1.259844).
        pitch_std (float): Standard deviation of pitch from training split (default: 12.062758).

    Returns:
        np.ndarray: Normalized pitch.
    """
    return (pitch_degrees - pitch_mean) / pitch_std


def denormalize_pitch(
    norm_pitch: np.ndarray,
    pitch_mean: float = -1.259844,
    pitch_std: float = 12.062758
) -> np.ndarray:
    """
    Denormalizes normalized pitch angles back to physical degrees: norm_pitch * std + mean.

    Parameters:
        norm_pitch (np.ndarray): Normalized pitch value(s).
        pitch_mean (float): Mean pitch from training split (default: -1.259844).
        pitch_std (float): Standard deviation of pitch from training split (default: 12.062758).

    Returns:
        np.ndarray: Physical pitch in degrees [-90, +90].
    """
    return norm_pitch * pitch_std + pitch_mean


