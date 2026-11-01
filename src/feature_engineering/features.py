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


def compute_derivatives(values: np.ndarray, dt: float = 0.10091) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates first-order angular velocity and second-order angular acceleration.

    Parameters:
        values (np.ndarray): 1D array of angular values (e.g., unwrapped yaw or pitch).
        dt (float): Sampling time step in seconds (default: 0.10091s, ~10 Hz).

    Returns:
        tuple[np.ndarray, np.ndarray]: (velocity in deg/s, acceleration in deg/s²).
    """
    vel = np.gradient(values, dt)
    acc = np.gradient(vel, dt)
    return vel, acc


def extract_7_features(
    yaw_degrees: np.ndarray,
    pitch_degrees: np.ndarray,
    dt: float = 0.10091
) -> np.ndarray:
    """
    Extracts the full 7-feature matrix for a single continuous viewing trajectory.

    Features:
        [sin_yaw, cos_yaw, pitch, yaw_vel, pitch_vel, yaw_acc, pitch_acc]

    Parameters:
        yaw_degrees (np.ndarray): Raw yaw angles in degrees.
        pitch_degrees (np.ndarray): Raw pitch angles in degrees.
        dt (float): Mean sampling interval (default: 0.10091s).

    Returns:
        np.ndarray: Feature matrix of shape (T, 7).
    """
    sin_yaw, cos_yaw = compute_sin_cos_yaw(yaw_degrees)
    unwrapped_yaw = unwrap_yaw_degrees(yaw_degrees)
    yaw_vel, yaw_acc = compute_derivatives(unwrapped_yaw, dt)
    pitch_vel, pitch_acc = compute_derivatives(pitch_degrees, dt)

    return np.column_stack([
        sin_yaw,
        cos_yaw,
        pitch_degrees,
        yaw_vel,
        pitch_vel,
        yaw_acc,
        pitch_acc
    ])


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

