"""
Module: sequence_generator.py
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming

Purpose:
    Implements sliding-window supervised sequence generation with historical
    sequence length L=10 (1.0s at 10 Hz) and prediction horizon k=5 (0.5s into future):
    - Input sequence tensors X of shape (N, 10, 7)
    - Target vectors y of shape (N, 3) -> [sin(future_yaw), cos(future_yaw), normalized_future_pitch]
    - Target degrees y_deg of shape (N, 2) -> [future_yaw_deg, future_pitch_deg]
    - Complete sample metadata stored directly in NPZ files:
      subject_id, video, trajectory_id, window_start_index, target_idx,
      target_timestamp, last_input_timestamp, target_yaw, target_pitch
    - Strict trajectory boundary protection (discarding incomplete boundary windows)
    - Subject-independent disjoint dataset partitioning (33 train, 7 val, 8 test)
    - Normalization statistics fitted strictly on training data only
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering.features import extract_7_features, standardize_features

REQUIRED_CSV_COLUMNS = ["subject_id", "video", "timestamp", "pitch", "yaw", "roll"]

OFFICIAL_SPLIT = {
    "train": [2, 3, 5, 6, 7, 8, 10, 11, 13, 14, 15, 17, 18, 19, 21, 23, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 39, 41, 42, 45, 46, 47, 48],
    "validation": [4, 12, 24, 25, 37, 38, 43],
    "test": [9, 16, 20, 22, 30, 40, 44, 49]
}


def validate_raw_trajectories_df(df: pd.DataFrame) -> None:
    """
    Validates structural integrity, types, nulls, and expected subject count of raw trajectory DataFrame.

    Raises:
        ValueError or TypeError if validation fails.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")

    if df.empty:
        raise ValueError("Trajectories DataFrame is empty.")

    missing_cols = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in trajectories DataFrame: {missing_cols}")

    # Check nulls
    null_counts = df[REQUIRED_CSV_COLUMNS].isna().sum()
    if null_counts.any():
        raise ValueError(f"DataFrame contains null values: {dict(null_counts[null_counts > 0])}")

    # Check numeric types
    for col in ["subject_id", "timestamp", "pitch", "yaw", "roll"]:
        if not np.issubdtype(df[col].dtype, np.number):
            raise TypeError(f"Column '{col}' must be numeric, got dtype {df[col].dtype}")

    # Check finite values
    for col in ["timestamp", "pitch", "yaw"]:
        vals = df[col].to_numpy()
        if not np.all(np.isfinite(vals)):
            raise ValueError(f"Column '{col}' contains non-finite values (NaN or Inf).")

    # Check unique subject count (48 subjects expected in AVTrack360)
    unique_subs = set(df["subject_id"].unique())
    if len(unique_subs) != 48:
        raise ValueError(f"Expected 48 unique subjects, found {len(unique_subs)}: {sorted(list(unique_subs))}")


def get_subject_split_mapping(split_csv_path: Path | str | None = None) -> Dict[str, List[int]]:
    """
    Returns the subject partition mapping (33 train, 7 val, 8 test).
    Loads from CSV if provided and exists; otherwise uses and saves canonical mapping.
    """
    if split_csv_path is not None:
        p = Path(split_csv_path)
        if p.exists():
            split_df = pd.read_csv(p)
            mapping: Dict[str, List[int]] = {"train": [], "validation": [], "test": []}
            for _, r in split_df.iterrows():
                mapping[r["split"]].append(int(r["subject_id"]))
            return mapping

    return {
        "train": sorted(list(OFFICIAL_SPLIT["train"])),
        "validation": sorted(list(OFFICIAL_SPLIT["validation"])),
        "test": sorted(list(OFFICIAL_SPLIT["test"]))
    }


def validate_subject_splits(splits: Dict[str, List[int]]) -> None:
    """
    Validates that:
    - Exactly 48 subjects are present across all splits.
    - 33 train, 7 val, 8 test subjects.
    - All splits are strictly disjoint (zero subject leakage).
    """
    train_set = set(splits.get("train", []))
    val_set = set(splits.get("validation", []))
    test_set = set(splits.get("test", []))

    if len(train_set) != 33:
        raise ValueError(f"Expected 33 training subjects, got {len(train_set)}")
    if len(val_set) != 7:
        raise ValueError(f"Expected 7 validation subjects, got {len(val_set)}")
    if len(test_set) != 8:
        raise ValueError(f"Expected 8 testing subjects, got {len(test_set)}")

    # Check disjointness
    tv_overlap = train_set.intersection(val_set)
    tt_overlap = train_set.intersection(test_set)
    vt_overlap = val_set.intersection(test_set)

    if tv_overlap:
        raise ValueError(f"Data leakage detected: subjects in both train and validation: {tv_overlap}")
    if tt_overlap:
        raise ValueError(f"Data leakage detected: subjects in both train and test: {tt_overlap}")
    if vt_overlap:
        raise ValueError(f"Data leakage detected: subjects in both validation and test: {vt_overlap}")

    all_subs = train_set | val_set | test_set
    if len(all_subs) != 48:
        raise ValueError(f"Total unique subjects across splits must be 48, got {len(all_subs)}")


def preprocess_and_validate_trajectories(
    df: pd.DataFrame,
    handle_initial_duplicates: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Inspects and sanitizes trajectory timestamps.

    Policy:
        - Rejects any non-monotonic timestamps (dt < 0).
        - If duplicate timestamps (dt == 0) are encountered:
            * In AVTrack360, duplicate timestamps occur exclusively at t=0.0 during headset calibration.
            * If handle_initial_duplicates=True: records every duplicate instance in audit metadata
              and deduplicates by retaining the clean consecutive sample, ensuring strictly increasing dt > 0.
            * If handle_initial_duplicates=False: raises ValueError.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned DataFrame and audit report.
    """
    validate_raw_trajectories_df(df)

    audit_records = []
    clean_dfs = []
    total_raw_rows = len(df)
    total_zero_intervals = 0

    for (sid, vid), group in df.groupby(["subject_id", "video"], sort=False):
        ts = group["timestamp"].to_numpy(dtype=np.float64)
        dts = ts[1:] - ts[:-1]

        # Check negative dt
        if np.any(dts < 0.0):
            neg_idx = np.where(dts < 0.0)[0][0]
            raise ValueError(
                f"Non-monotonic timestamps detected in trajectory sub_{sid}_vid_{vid} at index {neg_idx}: "
                f"t[{neg_idx}]={ts[neg_idx]}, t[{neg_idx+1}]={ts[neg_idx+1]}"
            )

        # Check zero dt
        zero_mask = (dts == 0.0)
        num_zero = int(np.sum(zero_mask))

        if num_zero > 0:
            total_zero_intervals += num_zero
            if not handle_initial_duplicates:
                raise ValueError(
                    f"Duplicate timestamps (dt=0) detected in trajectory sub_{sid}_vid_{vid} "
                    f"({num_zero} occurrences). Duplicate handling disabled."
                )

            # Record each duplicate in audit
            zero_indices = np.where(zero_mask)[0]
            for z in zero_indices:
                audit_records.append({
                    "subject_id": int(sid),
                    "video": str(vid),
                    "trajectory_id": f"sub_{sid}_vid_{str(vid).replace('.mp4', '')}",
                    "index": int(z),
                    "timestamp": float(ts[z]),
                    "yaw_before": float(group["yaw"].iloc[z]),
                    "yaw_after": float(group["yaw"].iloc[z + 1]),
                    "pitch_before": float(group["pitch"].iloc[z]),
                    "pitch_after": float(group["pitch"].iloc[z + 1]),
                    "correction": "Deduplicated identical/initial calibration timestamp at t=0"
                })

            # Drop duplicate consecutive timestamps (keep the last observation at the duplicated timestamp)
            group_clean = group.drop_duplicates(subset=["timestamp"], keep="last").copy()
            clean_dfs.append(group_clean)
        else:
            clean_dfs.append(group.copy())

    clean_df = pd.concat(clean_dfs, ignore_index=True)
    post_dts = []
    for (sid, vid), g in clean_df.groupby(["subject_id", "video"], sort=False):
        ts = g["timestamp"].to_numpy(dtype=np.float64)
        post_dts.extend(ts[1:] - ts[:-1])

    post_dts_arr = np.array(post_dts)
    audit_summary = {
        "total_raw_rows": total_raw_rows,
        "total_cleaned_rows": len(clean_df),
        "duplicate_timestamps_corrected": len(audit_records),
        "trajectories_count": len(clean_dfs),
        "dt_min_s": float(np.min(post_dts_arr)),
        "dt_max_s": float(np.max(post_dts_arr)),
        "dt_mean_s": float(np.mean(post_dts_arr)),
        "dt_median_s": float(np.median(post_dts_arr)),
        "dt_std_s": float(np.std(post_dts_arr)),
        "nominal_frequency_hz": float(1.0 / np.mean(post_dts_arr)),
        "correction_details": audit_records
    }

    return clean_df, audit_summary


def create_sliding_windows_with_metadata(
    features: np.ndarray,
    yaw_deg: np.ndarray,
    pitch_deg: np.ndarray,
    timestamps: np.ndarray,
    subject_id: int,
    video_name: str,
    trajectory_id: str,
    seq_len: int = 10,
    horizon: int = 5
) -> Dict[str, np.ndarray]:
    """
    Constructs sliding-window input sequences, multi-step future targets,
    and associated per-sample metadata.

    Mathematical Verification:
        - Input window i: frames [i, i + seq_len - 1] (length seq_len = 10)
        - Last observed input frame: index (i + seq_len - 1) = i + 9
        - Target frame: index (i + seq_len + horizon - 1) = i + 14
        - Prediction lead time: timestamps[i + 14] - timestamps[i + 9]

    Parameters:
        features: Shape (T, 7)
        yaw_deg, pitch_deg, timestamps: Shape (T,)
        subject_id, video_name, trajectory_id: Identifiers
        seq_len, horizon: Integers (10, 5)

    Returns:
        Dict with arrays: X, y_cont, y_deg, subject_ids, video_names, trajectory_ids,
        window_start_indices, target_indices, target_timestamps, last_input_timestamps,
        target_yaws, target_pitches.
    """
    total_frames = len(features)
    boundary_offset = seq_len + horizon - 1  # 10 + 5 - 1 = 14 frames
    num_samples = total_frames - boundary_offset

    if num_samples <= 0:
        return {
            "X": np.empty((0, seq_len, 7), dtype=np.float32),
            "y": np.empty((0, 3), dtype=np.float32),
            "y_deg": np.empty((0, 2), dtype=np.float32),
            "subject_ids": np.empty(0, dtype=np.int64),
            "video_names": np.empty(0, dtype=str),
            "trajectory_ids": np.empty(0, dtype=str),
            "window_start_indices": np.empty(0, dtype=np.int64),
            "target_indices": np.empty(0, dtype=np.int64),
            "target_timestamps": np.empty(0, dtype=np.float64),
            "last_input_timestamps": np.empty(0, dtype=np.float64),
            "target_yaws": np.empty(0, dtype=np.float32),
            "target_pitches": np.empty(0, dtype=np.float32),
        }

    X_list = []
    y_list = []
    y_deg_list = []
    target_ts_list = []
    last_input_ts_list = []
    target_yaw_list = []
    target_pitch_list = []
    win_start_list = []
    target_idx_list = []

    for i in range(num_samples):
        # Input history: frames i through i + 9
        x_seq = features[i : i + seq_len]
        # Target frame: frame i + 14 (5 frames ahead of frame i + 9)
        t_idx = i + boundary_offset
        last_in_idx = i + seq_len - 1

        fut_yaw = float(yaw_deg[t_idx])
        fut_pitch = float(pitch_deg[t_idx])
        fut_yaw_rad = np.radians(fut_yaw)

        X_list.append(x_seq)
        y_list.append([np.sin(fut_yaw_rad), np.cos(fut_yaw_rad), fut_pitch])
        y_deg_list.append([fut_yaw, fut_pitch])
        target_ts_list.append(timestamps[t_idx])
        last_input_ts_list.append(timestamps[last_in_idx])
        target_yaw_list.append(fut_yaw)
        target_pitch_list.append(fut_pitch)
        win_start_list.append(i)
        target_idx_list.append(t_idx)

    sub_arr = np.full(num_samples, subject_id, dtype=np.int64)
    vid_arr = np.full(num_samples, str(video_name), dtype='<U32')
    traj_arr = np.full(num_samples, str(trajectory_id), dtype='<U64')

    return {
        "X": np.array(X_list, dtype=np.float32),
        "y": np.array(y_list, dtype=np.float32),
        "y_deg": np.array(y_deg_list, dtype=np.float32),
        "subject_ids": sub_arr,
        "video_names": vid_arr,
        "trajectory_ids": traj_arr,
        "window_start_indices": np.array(win_start_list, dtype=np.int64),
        "target_indices": np.array(target_idx_list, dtype=np.int64),
        "target_timestamps": np.array(target_ts_list, dtype=np.float64),
        "last_input_timestamps": np.array(last_input_ts_list, dtype=np.float64),
        "target_yaws": np.array(target_yaw_list, dtype=np.float32),
        "target_pitches": np.array(target_pitch_list, dtype=np.float32),
    }


def create_sliding_windows(
    features: np.ndarray,
    yaw_deg: np.ndarray,
    pitch_deg: np.ndarray,
    seq_len: int = 10,
    horizon: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Backward-compatible sliding window helper returning (X, y, y_deg).
    For metadata-rich window generation, use create_sliding_windows_with_metadata.
    """
    T = len(features)
    dummy_ts = np.arange(T, dtype=np.float64) * 0.1
    res = create_sliding_windows_with_metadata(
        features=features,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        timestamps=dummy_ts,
        subject_id=0,
        video_name="unknown",
        trajectory_id="unknown",
        seq_len=seq_len,
        horizon=horizon
    )
    return res["X"], res["y"], res["y_deg"]


def generate_supervised_datasets(
    trajectories_csv_path: Path | str,
    output_dir: Path | str,
    seq_len: int = 10,
    horizon: int = 5
) -> Dict[str, Any]:
    """
    End-to-end dataset generation pipeline:
    1. Validates and sanitizes raw trajectories (documenting all corrections).
    2. Enforces strictly disjoint 33/7/8 subject split.
    3. Computes strictly causal 7-features using actual timestamps.
    4. Constructs sliding windows and attaches complete per-sample metadata.
    5. Fits Z-score normalization strictly on training split.
    6. Normalizes features across train, validation, and test splits.
    7. Saves compressed NPZ files, subject split CSV, and detailed summary reports.
    """
    csv_path = Path(trajectories_csv_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw trajectory dataset from: {csv_path}")
    df_raw = pd.read_csv(csv_path)

    # 1. Validate and clean timestamps
    clean_df, audit_summary = preprocess_and_validate_trajectories(df_raw, handle_initial_duplicates=True)
    print(
        f"Validation complete: {audit_summary['total_cleaned_rows']:,} rows across {audit_summary['trajectories_count']} trajectories. "
        f"Corrected duplicates: {audit_summary['duplicate_timestamps_corrected']}"
    )

    # 2. Validate subject split
    split_csv_path = out_dir / "lstm_subject_split.csv"
    splits = get_subject_split_mapping(split_csv_path if split_csv_path.exists() else None)
    validate_subject_splits(splits)

    # Save subject split CSV
    split_rows = []
    for split_name, sub_list in splits.items():
        for s in sub_list:
            split_rows.append({"subject_id": s, "split": split_name})
    split_df = pd.DataFrame(split_rows).sort_values("subject_id").reset_index(drop=True)
    split_df.to_csv(split_csv_path, index=False)
    print(f"Saved verified subject split mapping to: {split_csv_path}")

    train_subs = set(splits["train"])
    val_subs = set(splits["validation"])
    test_subs = set(splits["test"])

    split_data: Dict[str, Dict[str, List[np.ndarray]]] = {
        "train": {k: [] for k in ["X", "y", "y_deg", "subject_ids", "video_names", "trajectory_ids",
                                  "window_start_indices", "target_indices", "target_timestamps",
                                  "last_input_timestamps", "target_yaws", "target_pitches"]},
        "validation": {k: [] for k in ["X", "y", "y_deg", "subject_ids", "video_names", "trajectory_ids",
                                       "window_start_indices", "target_indices", "target_timestamps",
                                       "last_input_timestamps", "target_yaws", "target_pitches"]},
        "test": {k: [] for k in ["X", "y", "y_deg", "subject_ids", "video_names", "trajectory_ids",
                                 "window_start_indices", "target_indices", "target_timestamps",
                                 "last_input_timestamps", "target_yaws", "target_pitches"]},
    }

    # 3. Process every trajectory
    grouped = clean_df.groupby(["subject_id", "video"], sort=False)
    for (sid, vid), grp in grouped:
        yaw = grp["yaw"].to_numpy(dtype=np.float64)
        pitch = grp["pitch"].to_numpy(dtype=np.float64)
        ts = grp["timestamp"].to_numpy(dtype=np.float64)

        # Causal 7-feature extraction using exact timestamps
        features = extract_7_features(yaw, pitch, timestamps=ts)

        traj_id = f"sub_{sid}_vid_{str(vid).replace('.mp4', '')}"
        win_dict = create_sliding_windows_with_metadata(
            features=features,
            yaw_deg=yaw,
            pitch_deg=pitch,
            timestamps=ts,
            subject_id=int(sid),
            video_name=str(vid),
            trajectory_id=traj_id,
            seq_len=seq_len,
            horizon=horizon
        )

        if len(win_dict["X"]) == 0:
            continue

        target_split = "train" if sid in train_subs else ("validation" if sid in val_subs else "test")
        for k in split_data[target_split]:
            split_data[target_split][k].append(win_dict[k])

    # 4. Concatenate per split
    final_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    for s_name in ["train", "validation", "test"]:
        final_arrays[s_name] = {}
        for k, arr_list in split_data[s_name].items():
            if k in ["X", "y", "y_deg"]:
                final_arrays[s_name][k] = np.vstack(arr_list)
            else:
                final_arrays[s_name][k] = np.concatenate(arr_list)

    # 5. Fit Z-score normalization strictly on training split
    X_train_raw = final_arrays["train"]["X"]
    train_flat = X_train_raw.reshape(-1, 7)
    norm_mean = np.mean(train_flat, axis=0).astype(np.float32)
    norm_std = np.std(train_flat, axis=0).astype(np.float32)
    norm_std[norm_std == 0.0] = 1.0  # Safe zero-variance guard

    # Pitch normalization stats from training split targets
    pitch_mean = float(np.mean(final_arrays["train"]["y_deg"][:, 1]))
    pitch_std = float(np.std(final_arrays["train"]["y_deg"][:, 1]))
    if pitch_std == 0.0:
        pitch_std = 1.0

    # 6. Apply normalization
    feature_names = np.array([
        "sin_yaw", "cos_yaw", "pitch", "yaw_vel", "pitch_vel", "yaw_acc", "pitch_acc"
    ])

    for s_name in ["train", "validation", "test"]:
        X_raw = final_arrays[s_name]["X"]
        final_arrays[s_name]["X_norm"] = (X_raw - norm_mean) / norm_std

        # Normalize pitch in target representation: [sin, cos, norm_pitch]
        y_raw = final_arrays[s_name]["y"]
        y_norm_pitch = (y_raw[:, 2] - pitch_mean) / pitch_std
        final_arrays[s_name]["y_norm"] = np.column_stack([y_raw[:, 0], y_raw[:, 1], y_norm_pitch]).astype(np.float32)

    # 7. Save compressed NPZ files
    for s_name, filename in [("train", "lstm_train.npz"), ("validation", "lstm_validation.npz"), ("test", "lstm_test.npz")]:
        npz_path = out_dir / filename
        data_to_save = {
            "X": final_arrays[s_name]["X_norm"],
            "y": final_arrays[s_name]["y_norm"],
            "y_deg": final_arrays[s_name]["y_deg"],
            "norm_mean": norm_mean,
            "norm_std": norm_std,
            "pitch_mean": np.float32(pitch_mean),
            "pitch_std": np.float32(pitch_std),
            "feature_names": feature_names,
            "subject_ids": final_arrays[s_name]["subject_ids"],
            "video_names": final_arrays[s_name]["video_names"],
            "trajectory_ids": final_arrays[s_name]["trajectory_ids"],
            "window_start_indices": final_arrays[s_name]["window_start_indices"],
            "target_indices": final_arrays[s_name]["target_indices"],
            "target_timestamps": final_arrays[s_name]["target_timestamps"],
            "last_input_timestamps": final_arrays[s_name]["last_input_timestamps"],
            "target_yaws": final_arrays[s_name]["target_yaws"],
            "target_pitches": final_arrays[s_name]["target_pitches"],
        }
        np.savez_compressed(npz_path, **data_to_save)
        print(f"Saved {s_name} split to: {npz_path} ({len(final_arrays[s_name]['X_norm']):,} samples)")

    # 8. Save audit metadata JSON
    audit_file = out_dir / "dataset_audit_metadata.json"
    audit_payload = {
        "dataset_name": "AVTrack360 Preprocessed Causal Sequences",
        "sequence_parameters": {
            "history_length_L": seq_len,
            "prediction_horizon_k": horizon,
            "input_dimension": 7,
            "target_dimension": 3
        },
        "sample_counts": {
            "train": int(len(final_arrays["train"]["X_norm"])),
            "validation": int(len(final_arrays["validation"]["X_norm"])),
            "test": int(len(final_arrays["test"]["X_norm"])),
            "total": int(len(final_arrays["train"]["X_norm"]) + len(final_arrays["validation"]["X_norm"]) + len(final_arrays["test"]["X_norm"]))
        },
        "subject_split": {
            "train_subjects": splits["train"],
            "validation_subjects": splits["validation"],
            "test_subjects": splits["test"]
        },
        "normalization": {
            "mean": norm_mean.tolist(),
            "std": norm_std.tolist(),
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "features": feature_names.tolist()
        },
        "sampling_interval_audit": audit_summary
    }
    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump(audit_payload, f, indent=2)
    print(f"Saved dataset audit metadata to: {audit_file}")

    return audit_payload


if __name__ == "__main__":
    traj_path = PROJECT_ROOT / "data" / "processed" / "viewport_trajectories.csv"
    out_path = PROJECT_ROOT / "data" / "processed"
    generate_supervised_datasets(traj_path, out_path, seq_len=10, horizon=5)
