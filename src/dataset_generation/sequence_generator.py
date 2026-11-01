"""
Module: sequence_generator.py
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming

Purpose:
    Implements sliding-window supervised sequence generation with historical
    sequence length L=10 (1.0s at 10 Hz) and prediction horizon k=5 (0.5s into future):
    - Generates input sequence tensors X of shape (N, 10, 7)
    - Generates target vectors y of shape (N, 3) -> [sin(future_yaw), cos(future_yaw), future_pitch]
    - Generates raw evaluation targets y_deg of shape (N, 2) -> [future_yaw_deg, future_pitch_deg]
    - Enforces strict trajectory boundary protection (discarding incomplete boundary windows)
    - Manages subject-independent disjoint dataset partitioning (33 train, 7 val, 8 test)
"""

import numpy as np


def create_sliding_windows(
    features: np.ndarray,
    yaw_deg: np.ndarray,
    pitch_deg: np.ndarray,
    seq_len: int = 10,
    horizon: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs sliding-window input sequences and multi-step future prediction targets
    for a single continuous viewing trajectory.

    For a trajectory of length T:
    - Input sequence i: frames i to i + seq_len - 1 (10 historical steps)
    - Future target index: i + seq_len + horizon - 1 (5 steps ahead of last observed frame)
    - Number of valid sequences: T - (seq_len + horizon - 1) = T - 14

    Boundary Protection:
        The final 14 frames cannot form a complete input + horizon window and are discarded.

    Parameters:
        features (np.ndarray): 7-feature matrix of shape (T, 7).
        yaw_deg (np.ndarray): Yaw angle degrees of shape (T,).
        pitch_deg (np.ndarray): Pitch angle degrees of shape (T,).
        seq_len (int): Historical window length (default: 10 frames = 1.0s at 10 Hz).
        horizon (int): Prediction lead time (default: 5 frames = 0.5s ahead).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - X: Input sequence tensor of shape (M, 10, 7).
            - y: Continuous target array of shape (M, 3) -> [sin, cos, pitch_deg].
            - y_deg: Raw degree target array of shape (M, 2) -> [yaw_deg, pitch_deg].
    """
    total_frames = len(features)
    boundary_offset = seq_len + horizon - 1  # 10 + 5 - 1 = 14 frames
    num_samples = total_frames - boundary_offset

    if num_samples <= 0:
        return (
            np.empty((0, seq_len, features.shape[1]), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32)
        )

    X_list = []
    y_list = []
    y_deg_list = []

    for i in range(num_samples):
        # Input history: frames i through i + 9
        x_seq = features[i : i + seq_len]
        # Target frame: frame i + 14 (0.5s into the future)
        target_idx = i + boundary_offset

        fut_yaw = yaw_deg[target_idx]
        fut_pitch = pitch_deg[target_idx]

        fut_yaw_rad = np.radians(fut_yaw)
        target_cont = [np.sin(fut_yaw_rad), np.cos(fut_yaw_rad), fut_pitch]
        target_deg = [fut_yaw, fut_pitch]

        X_list.append(x_seq)
        y_list.append(target_cont)
        y_deg_list.append(target_deg)

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        np.array(y_deg_list, dtype=np.float32)
    )


def get_subject_split_mapping() -> dict[str, list[int]]:
    """
    Returns the official subject partition mapping used in the AVTrack360 experiments:
    - 33 Training subjects (~70%)
    - 7 Validation subjects (~15%)
    - 8 Testing subjects (~15%)
    Strictly disjoint to guarantee zero data leakage.

    Returns:
        dict[str, list[int]]: Mapping of split name to list of subject IDs.
    """
    return {
        "train": [2, 3, 5, 6, 7, 8, 10, 11, 13, 14, 15, 17, 18, 19, 21, 23, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 39, 41, 42, 45, 46, 47, 48],
        "validation": [4, 12, 24, 25, 37, 38, 43],
        "test": [9, 16, 20, 22, 30, 40, 44, 49]
    }

