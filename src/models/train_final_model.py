"""
Script: train_final_model.py
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming

Phase 4: Retrains and evaluates the verified, optimal final viewport prediction model:
- Architecture: ConfigurableGRU (input_dim=7, hidden_dim=64, proj_dim=32, output_dim=3, dropout=0.0)
- Sequence Length: L = 10 frames (~1.0s historical context at ~10 Hz)
- Prediction Horizon: k = 5 frames (~0.5s lead time)
- Feature Representation: Strictly causal 7-feature kinematic representation
- Optimization: Adam (lr=0.0005, batch_size=256, MSE loss on [sin, cos, norm_pitch])
- Early Stopping: Monitored strictly on Validation MSE Loss (patience=5 epochs)
- Subject Partitioning: Disjoint subject split loaded dynamically from dataset metadata (zero leakage)
- Metadata Alignment: Direct loading of target timestamps and trajectory IDs from NPZ datasets

Outputs:
- Checkpoint: saved_models/final_viewport_model.pth
- Configuration JSON: saved_models/final_model_config.json
- Test Predictions: results/predictions/final_model_test_predictions.csv
- Metrics CSV: results/metrics/final_model_evaluation_metrics.csv
- Comparison CSV: results/metrics/final_model_comparison.csv
- Diagnostic Plots: results/plots/final_model_loss_curve.png, results/plots/final_model_error_cdf.png, results/plots/final_model_vs_baselines_mae.png
"""

import sys
import os
import json
import time
import random
import platform
from pathlib import Path
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.architectures import ConfigurableGRU, ModelA_LSTM64, ModelC_GRU64

DATA_DIR = PROJECT_ROOT / "data" / "processed"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
RESULTS_METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
RESULTS_PREDS_DIR = PROJECT_ROOT / "results" / "predictions"
RESULTS_PLOTS_DIR = PROJECT_ROOT / "results" / "plots"

SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PREDS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


def set_seed(seed: int = 42) -> torch.Generator:
    """Sets seeds across Python random, NumPy, and PyTorch for deterministic execution."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def compute_spherical_error_arrays(act_yaw, act_pitch, pred_yaw, pred_pitch):
    """Computes pointwise angular differences and orthodromic great-circle distance in degrees."""
    yaw_diff = np.abs((pred_yaw - act_yaw + 180.0) % 360.0 - 180.0)
    pitch_diff = np.abs(pred_pitch - act_pitch)

    p1 = np.radians(act_pitch.astype(np.float64))
    y1 = np.radians(act_yaw.astype(np.float64))
    p2 = np.radians(pred_pitch.astype(np.float64))
    y2 = np.radians(pred_yaw.astype(np.float64))

    dot = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(y1 - y2)
    dot = np.clip(dot, -1.0, 1.0)
    sph_err = np.degrees(np.arccos(dot))

    return {
        'yaw_diff': yaw_diff,
        'pitch_diff': pitch_diff,
        'sph_err': sph_err,
        'sph_mae': float(np.mean(sph_err)),
        'sph_median': float(np.median(sph_err)),
        'sph_p90': float(np.percentile(sph_err, 90)),
        'sph_rmse': float(np.sqrt(np.mean(sph_err ** 2))),
        'yaw_mae': float(np.mean(yaw_diff)),
        'yaw_rmse': float(np.sqrt(np.mean(yaw_diff ** 2))),
        'pitch_mae': float(np.mean(pitch_diff)),
        'pitch_rmse': float(np.sqrt(np.mean(pitch_diff ** 2))),
    }


def validate_npz_datasets(tr_npz, va_npz, te_npz):
    """Validates data shapes, finite numbers, metadata lengths, and zero subject overlap."""
    for name, npz in [("train", tr_npz), ("validation", va_npz), ("test", te_npz)]:
        X = npz["X"]
        y = npz["y"]
        y_deg = npz["y_deg"]
        n_samples = len(X)

        if X.ndim != 3 or X.shape[1] != 10 or X.shape[2] != 7:
            raise ValueError(f"Invalid X shape in {name} split: {X.shape}, expected (N, 10, 7)")
        if y.ndim != 2 or y.shape[1] != 3:
            raise ValueError(f"Invalid y shape in {name} split: {y.shape}, expected (N, 3)")
        if y_deg.ndim != 2 or y_deg.shape[1] != 2:
            raise ValueError(f"Invalid y_deg shape in {name} split: {y_deg.shape}, expected (N, 2)")

        if not np.all(np.isfinite(X)):
            raise ValueError(f"Non-finite values detected in {name} X")
        if not np.all(np.isfinite(y)):
            raise ValueError(f"Non-finite values detected in {name} y")
        if not np.all(np.isfinite(y_deg)):
            raise ValueError(f"Non-finite values detected in {name} y_deg")

        # Validate metadata fields
        for meta_key in ["subject_ids", "video_names", "trajectory_ids", "target_timestamps"]:
            if meta_key not in npz:
                raise KeyError(f"Missing required metadata key '{meta_key}' in {name} NPZ")
            if len(npz[meta_key]) != n_samples:
                raise ValueError(
                    f"Metadata '{meta_key}' length ({len(npz[meta_key])}) does not match sample count ({n_samples})"
                )

    # Disjoint subject check
    train_subs = set(tr_npz["subject_ids"])
    val_subs = set(va_npz["subject_ids"])
    test_subs = set(te_npz["subject_ids"])

    tv_overlap = train_subs & val_subs
    tt_overlap = train_subs & test_subs
    vt_overlap = val_subs & test_subs

    if tv_overlap or tt_overlap or vt_overlap:
        raise ValueError(
            f"Subject overlap detected! Train-Val: {tv_overlap}, Train-Test: {tt_overlap}, Val-Test: {vt_overlap}"
        )


def benchmark_latency(model_or_fn, sample_input, is_pytorch=True, repetitions=2000):
    """Benchmarks inference latency using identical warm-up and timing policy."""
    # Warmup
    for _ in range(100):
        if is_pytorch:
            with torch.inference_mode():
                _ = model_or_fn(sample_input)
        else:
            _ = model_or_fn(sample_input)

    times = []
    for _ in range(repetitions):
        t0 = time.perf_counter()
        if is_pytorch:
            with torch.inference_mode():
                _ = model_or_fn(sample_input)
        else:
            _ = model_or_fn(sample_input)
        times.append((time.perf_counter() - t0) * 1e6)  # microseconds

    times_arr = np.array(times)
    return float(np.mean(times_arr)), float(np.median(times_arr))


def main():
    print("================================================================================")
    print(" PHASE 4: FINAL MODEL RETRAINING & COMPREHENSIVE BENCHMARK")
    print("================================================================================")

    # 0. Set Seeds and Determinism
    g_loader = set_seed(SEED)

    # 1. Load Preprocessed Datasets
    print("\n--- 1. Loading Causal Preprocessed Datasets ---")
    tr_path = DATA_DIR / 'lstm_train.npz'
    va_path = DATA_DIR / 'lstm_validation.npz'
    te_path = DATA_DIR / 'lstm_test.npz'

    for p in [tr_path, va_path, te_path]:
        if not p.exists():
            raise FileNotFoundError(f"Dataset file missing: {p}")

    tr_npz = np.load(tr_path)
    va_npz = np.load(va_path)
    te_npz = np.load(te_path)

    # Validate datasets
    validate_npz_datasets(tr_npz, va_npz, te_npz)

    X_tr = tr_npz['X']
    y_tr = tr_npz['y']
    X_va = va_npz['X']
    y_va = va_npz['y']
    yd_va = va_npz['y_deg']
    X_te = te_npz['X']
    y_te = te_npz['y']
    yd_te = te_npz['y_deg']

    norm_mean = tr_npz['norm_mean']
    norm_std = tr_npz['norm_std']
    pitch_mean = float(tr_npz['pitch_mean'])
    pitch_std = float(tr_npz['pitch_std'])
    feature_names = [str(x) for x in tr_npz['feature_names']]

    train_subs = sorted(list(set(tr_npz['subject_ids'])))
    val_subs = sorted(list(set(va_npz['subject_ids'])))
    test_subs = sorted(list(set(te_npz['subject_ids'])))

    test_sub_ids = te_npz['subject_ids']
    test_vids = te_npz['video_names']
    test_traj_ids = te_npz['trajectory_ids']
    test_target_ts = te_npz['target_timestamps']
    test_last_input_ts = te_npz['last_input_timestamps']

    print(f"Train split:      {X_tr.shape[0]:,} samples ({len(train_subs)} subjects: {train_subs})")
    print(f"Validation split: {X_va.shape[0]:,} samples ({len(val_subs)} subjects: {val_subs})")
    print(f"Test split:       {X_te.shape[0]:,} samples ({len(test_subs)} subjects: {test_subs})")
    print(f"Feature names:    {feature_names}")
    print(f"Norm Mean:        {np.round(norm_mean, 4)}")
    print(f"Norm Std:         {np.round(norm_std, 4)}")

    # 2. Dataloaders
    batch_size = 256
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True, generator=g_loader
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_va, dtype=torch.float32), torch.tensor(y_va, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_te, dtype=torch.float32), torch.tensor(y_te, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False
    )

    # 3. Model Architecture Instantiation
    print("\n--- 2. Instantiating Optimal Model Architecture ---")
    model = ConfigurableGRU(input_dim=7, hidden_dim=64, dropout=0.0)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Architecture: ConfigurableGRU (input_dim=7, hidden_dim=64, proj_dim=32, output_dim=3, dropout=0.0)")
    print(f"Trainable Parameters: {param_count:,}")

    # 4. Training Loop
    lr = 0.0005
    max_epochs = 20
    patience = 5
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n--- 3. Retraining Final Model (lr={lr}, batch_size={batch_size}, patience={patience}) ---")
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_weights = None
    history = {'train_loss': [], 'val_loss': []}

    start_train_time = time.time()

    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= len(X_tr)

        # Validate
        model.eval()
        val_loss = 0.0
        val_preds = []
        with torch.no_grad():
            for bx, by in val_loader:
                out = model(bx)
                loss = criterion(out, by)
                val_loss += loss.item() * len(bx)
                val_preds.append(out.numpy())
        val_loss /= len(X_va)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Compute validation angular metrics
        val_preds = np.vstack(val_preds)
        val_sin, val_cos = val_preds[:, 0], val_preds[:, 1]
        val_pred_yaw = np.degrees(np.arctan2(val_sin, val_cos))
        val_pred_pitch = val_preds[:, 2] * pitch_std + pitch_mean
        v_metrics = compute_spherical_error_arrays(yd_va[:, 0], yd_va[:, 1], val_pred_yaw, val_pred_pitch)

        print(f"Epoch {epoch:2d}/{max_epochs:2d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Sph MAE: {v_metrics['sph_mae']:.4f}°")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Restoring best weights from epoch {best_epoch}.")
                break

    train_duration = time.time() - start_train_time
    print(f"\nTraining completed in {train_duration:.2f} seconds.")
    print(f"Best Validation Loss: {best_val_loss:.6f} at Epoch {best_epoch}")

    # Load best weights
    model.load_state_dict(best_weights)

    # 5. Measure Inference Latency
    model.eval()
    sample_input = torch.tensor(X_te[:1], dtype=torch.float32)
    lat_mean_us, lat_med_us = benchmark_latency(model, sample_input, is_pytorch=True, repetitions=2000)
    print(f"Inference Latency: Mean = {lat_mean_us:.2f} µs, Median = {lat_med_us:.2f} µs per sample")

    # 6. Save Checkpoint to saved_models/final_viewport_model.pth
    ckpt_path = SAVED_MODELS_DIR / "final_viewport_model.pth"
    torch.save(best_weights, ckpt_path)
    model_size_kb = ckpt_path.stat().st_size / 1024
    print(f"Saved final model checkpoint to: {ckpt_path} ({model_size_kb:.1f} KB)")

    # 7. Final Evaluation on Untouched Test Set
    print("\n--- 4. Evaluating Final Model on Untouched Test Set ---")
    test_preds = []
    with torch.no_grad():
        for bx, _ in test_loader:
            test_preds.append(model(bx).numpy())
    test_preds = np.vstack(test_preds)

    t_sin, t_cos = test_preds[:, 0], test_preds[:, 1]
    final_pred_yaw = np.degrees(np.arctan2(t_sin, t_cos))
    final_pred_pitch = test_preds[:, 2] * pitch_std + pitch_mean

    final_metrics = compute_spherical_error_arrays(
        yd_te[:, 0], yd_te[:, 1], final_pred_yaw, final_pred_pitch
    )

    print(f"Final Model Test Performance:")
    print(f"  Spherical MAE:    {final_metrics['sph_mae']:.4f}°")
    print(f"  Spherical Median: {final_metrics['sph_median']:.4f}°")
    print(f"  Spherical P90:    {final_metrics['sph_p90']:.4f}°")
    print(f"  Spherical RMSE:   {final_metrics['sph_rmse']:.4f}°")
    print(f"  Yaw MAE:          {final_metrics['yaw_mae']:.4f}°")
    print(f"  Pitch MAE:        {final_metrics['pitch_mae']:.4f}°")

    # 8. Compare Against Baselines
    print("\n--- 5. Comparative Evaluation Against Baselines ---")
    # A. Persistence Baseline
    sin_last = X_te[:, -1, 0] * norm_std[0] + norm_mean[0]
    cos_last = X_te[:, -1, 1] * norm_std[1] + norm_mean[1]
    pers_yaw = np.degrees(np.arctan2(sin_last, cos_last))
    pers_pitch = X_te[:, -1, 2] * norm_std[2] + norm_mean[2]
    pers_metrics = compute_spherical_error_arrays(yd_te[:, 0], yd_te[:, 1], pers_yaw, pers_pitch)

    # Benchmark Persistence Latency
    def run_persistence(sample):
        sl = sample[0, -1, 0] * norm_std[0] + norm_mean[0]
        cl = sample[0, -1, 1] * norm_std[1] + norm_mean[1]
        py = np.degrees(np.arctan2(sl, cl))
        pp = sample[0, -1, 2] * norm_std[2] + norm_mean[2]
        return py, pp

    pers_lat_mean, pers_lat_med = benchmark_latency(run_persistence, X_te[:1], is_pytorch=False, repetitions=2000)

    # B. Constant Velocity Baseline (using actual sample-specific lead times)
    yaw_vel_test = X_te[:, -1, 3] * norm_std[3] + norm_mean[3]
    pitch_vel_test = X_te[:, -1, 4] * norm_std[4] + norm_mean[4]
    lead_time_s = test_target_ts - test_last_input_ts
    cvel_yaw = (pers_yaw + yaw_vel_test * lead_time_s + 180.0) % 360.0 - 180.0
    cvel_pitch = np.clip(pers_pitch + pitch_vel_test * lead_time_s, -90.0, 90.0)
    cvel_metrics = compute_spherical_error_arrays(yd_te[:, 0], yd_te[:, 1], cvel_yaw, cvel_pitch)

    def run_cvel(sample):
        py, pp = run_persistence(sample)
        yv = sample[0, -1, 3] * norm_std[3] + norm_mean[3]
        pv = sample[0, -1, 4] * norm_std[4] + norm_mean[4]
        cy = (py + yv * 0.50455 + 180.0) % 360.0 - 180.0
        cp = np.clip(pp + pv * 0.50455, -90.0, 90.0)
        return cy, cp

    cvel_lat_mean, cvel_lat_med = benchmark_latency(run_cvel, X_te[:1], is_pytorch=False, repetitions=2000)

    # C. Existing Model A (LSTM-64 Baseline) compatibility verification
    lstm_ckpt_path = SAVED_MODELS_DIR / 'viewport_lstm_model.pth'
    has_lstm = False
    if lstm_ckpt_path.exists():
        try:
            m_lstm = ModelA_LSTM64()
            m_lstm.load_state_dict(torch.load(lstm_ckpt_path, map_location='cpu', weights_only=True))
            m_lstm.eval()
            with torch.inference_mode():
                lstm_preds = m_lstm(torch.tensor(X_te, dtype=torch.float32)).numpy()
            l_yaw = np.degrees(np.arctan2(lstm_preds[:, 0], lstm_preds[:, 1]))
            l_pitch = lstm_preds[:, 2] * pitch_std + pitch_mean
            lstm_metrics = compute_spherical_error_arrays(yd_te[:, 0], yd_te[:, 1], l_yaw, l_pitch)
            lstm_lat_mean, lstm_lat_med = benchmark_latency(m_lstm, sample_input, is_pytorch=True, repetitions=2000)
            has_lstm = True
        except Exception as e:
            print(f"Model A (LSTM-64) checkpoint incompatible: {e}. Excluded from benchmark.")

    # D. Existing Model C (GRU-64 Baseline) compatibility verification
    gru_ckpt_path = SAVED_MODELS_DIR / 'best_viewport_model.pth'
    has_gru_base = False
    if gru_ckpt_path.exists():
        try:
            m_gru = ModelC_GRU64()
            m_gru.load_state_dict(torch.load(gru_ckpt_path, map_location='cpu', weights_only=True))
            m_gru.eval()
            with torch.inference_mode():
                gru_preds = m_gru(torch.tensor(X_te, dtype=torch.float32)).numpy()
            g_yaw = np.degrees(np.arctan2(gru_preds[:, 0], gru_preds[:, 1]))
            g_pitch = gru_preds[:, 2] * pitch_std + pitch_mean
            gru_metrics = compute_spherical_error_arrays(yd_te[:, 0], yd_te[:, 1], g_yaw, g_pitch)
            gru_lat_mean, gru_lat_med = benchmark_latency(m_gru, sample_input, is_pytorch=True, repetitions=2000)
            has_gru_base = True
        except Exception as e:
            print(f"Model C (GRU-64) checkpoint incompatible: {e}. Excluded from benchmark.")

    imp_pers = (pers_metrics['sph_mae'] - final_metrics['sph_mae']) / pers_metrics['sph_mae'] * 100
    imp_cvel = (cvel_metrics['sph_mae'] - final_metrics['sph_mae']) / cvel_metrics['sph_mae'] * 100

    print(f"Improvement over Persistence Baseline:           {imp_pers:+.2f}%")
    print(f"Improvement over Constant Velocity Baseline:     {imp_cvel:+.2f}%")

    # 9. Save Test Predictions CSV
    df_preds = pd.DataFrame({
        'sample_index': np.arange(len(yd_te)),
        'subject_id': test_sub_ids,
        'video': test_vids,
        'trajectory_id': test_traj_ids,
        'target_timestamp': test_target_ts,
        'actual_yaw_deg': yd_te[:, 0],
        'actual_pitch_deg': yd_te[:, 1],
        'predicted_yaw_deg': final_pred_yaw,
        'predicted_pitch_deg': final_pred_pitch,
        'spherical_error_deg': final_metrics['sph_err'],
        'persistence_predicted_yaw_deg': pers_yaw,
        'persistence_predicted_pitch_deg': pers_pitch,
        'persistence_sph_err_deg': pers_metrics['sph_err'],
        'const_vel_sph_err_deg': cvel_metrics['sph_err']
    })
    preds_csv_path = RESULTS_PREDS_DIR / "final_model_test_predictions.csv"
    df_preds.to_csv(preds_csv_path, index=False)
    print(f"Saved test predictions to: {preds_csv_path}")

    # 10. Save Detailed Final Comparison Table CSV
    comparison_table = [
        {
            'Model / Method': 'Persistence Baseline (Zero-Order Hold)',
            'Method Type': 'Heuristic Baseline',
            'Parameters': 0,
            'Inference Latency Mean (us)': pers_lat_mean,
            'Inference Latency Median (us)': pers_lat_med,
            'Spherical MAE (deg)': pers_metrics['sph_mae'],
            'Spherical Median (deg)': pers_metrics['sph_median'],
            'Spherical P90 (deg)': pers_metrics['sph_p90'],
            'Spherical RMSE (deg)': pers_metrics['sph_rmse'],
            'Yaw MAE (deg)': pers_metrics['yaw_mae'],
            'Pitch MAE (deg)': pers_metrics['pitch_mae'],
            'Error Reduction vs Persistence (%)': 0.0
        },
        {
            'Model / Method': 'Constant Angular Velocity Baseline',
            'Method Type': 'Kinematic Extrapolation',
            'Parameters': 0,
            'Inference Latency Mean (us)': cvel_lat_mean,
            'Inference Latency Median (us)': cvel_lat_med,
            'Spherical MAE (deg)': cvel_metrics['sph_mae'],
            'Spherical Median (deg)': cvel_metrics['sph_median'],
            'Spherical P90 (deg)': cvel_metrics['sph_p90'],
            'Spherical RMSE (deg)': cvel_metrics['sph_rmse'],
            'Yaw MAE (deg)': cvel_metrics['yaw_mae'],
            'Pitch MAE (deg)': cvel_metrics['pitch_mae'],
            'Error Reduction vs Persistence (%)': (pers_metrics['sph_mae'] - cvel_metrics['sph_mae']) / pers_metrics['sph_mae'] * 100
        }
    ]

    if has_lstm:
        comparison_table.append({
            'Model / Method': 'Model A — LSTM-64 Baseline',
            'Method Type': 'Deep Recurrent NN',
            'Parameters': 20835,
            'Inference Latency Mean (us)': lstm_lat_mean,
            'Inference Latency Median (us)': lstm_lat_med,
            'Spherical MAE (deg)': lstm_metrics['sph_mae'],
            'Spherical Median (deg)': lstm_metrics['sph_median'],
            'Spherical P90 (deg)': lstm_metrics['sph_p90'],
            'Spherical RMSE (deg)': lstm_metrics['sph_rmse'],
            'Yaw MAE (deg)': lstm_metrics['yaw_mae'],
            'Pitch MAE (deg)': lstm_metrics['pitch_mae'],
            'Error Reduction vs Persistence (%)': (pers_metrics['sph_mae'] - lstm_metrics['sph_mae']) / pers_metrics['sph_mae'] * 100
        })

    if has_gru_base:
        comparison_table.append({
            'Model / Method': 'Model C — GRU-64 Baseline',
            'Method Type': 'Deep Recurrent NN',
            'Parameters': 16195,
            'Inference Latency Mean (us)': gru_lat_mean,
            'Inference Latency Median (us)': gru_lat_med,
            'Spherical MAE (deg)': gru_metrics['sph_mae'],
            'Spherical Median (deg)': gru_metrics['sph_median'],
            'Spherical P90 (deg)': gru_metrics['sph_p90'],
            'Spherical RMSE (deg)': gru_metrics['sph_rmse'],
            'Yaw MAE (deg)': gru_metrics['yaw_mae'],
            'Pitch MAE (deg)': gru_metrics['pitch_mae'],
            'Error Reduction vs Persistence (%)': (pers_metrics['sph_mae'] - gru_metrics['sph_mae']) / pers_metrics['sph_mae'] * 100
        })

    comparison_table.append({
        'Model / Method': 'FINAL OPTIMAL MODEL — GRU-64 (Causal Parity)',
        'Method Type': 'Deep Recurrent NN (Optimal)',
        'Parameters': param_count,
        'Inference Latency Mean (us)': lat_mean_us,
        'Inference Latency Median (us)': lat_med_us,
        'Spherical MAE (deg)': final_metrics['sph_mae'],
        'Spherical Median (deg)': final_metrics['sph_median'],
        'Spherical P90 (deg)': final_metrics['sph_p90'],
        'Spherical RMSE (deg)': final_metrics['sph_rmse'],
        'Yaw MAE (deg)': final_metrics['yaw_mae'],
        'Pitch MAE (deg)': final_metrics['pitch_mae'],
        'Error Reduction vs Persistence (%)': imp_pers
    })

    df_comp = pd.DataFrame(comparison_table)
    comp_csv_path = RESULTS_METRICS_DIR / "final_model_comparison.csv"
    df_comp.to_csv(comp_csv_path, index=False)
    print(f"Saved final comparison table to: {comp_csv_path}")

    # Save metrics CSV
    metrics_summary = [
        {"Metric": "Spherical MAE", "Value_deg": final_metrics['sph_mae']},
        {"Metric": "Spherical Median", "Value_deg": final_metrics['sph_median']},
        {"Metric": "Spherical P90", "Value_deg": final_metrics['sph_p90']},
        {"Metric": "Spherical RMSE", "Value_deg": final_metrics['sph_rmse']},
        {"Metric": "Yaw MAE", "Value_deg": final_metrics['yaw_mae']},
        {"Metric": "Pitch MAE", "Value_deg": final_metrics['pitch_mae']},
    ]
    pd.DataFrame(metrics_summary).to_csv(RESULTS_METRICS_DIR / "final_model_evaluation_metrics.csv", index=False)

    # 11. Save Configuration JSON
    final_config = {
        "model_name": "final_viewport_model",
        "project_title": "Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360 Video Streaming",
        "reproducibility": {
            "random_seed": SEED,
            "deterministic_algorithms_enabled": True,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "os_platform": platform.platform(),
            "cpu_architecture": platform.processor(),
        },
        "architecture": {
            "type": "ConfigurableGRU",
            "input_dim": 7,
            "hidden_dim": 64,
            "proj_dim": 32,
            "output_dim": 3,
            "dropout": 0.0,
            "param_count": param_count,
            "model_size_kb": model_size_kb
        },
        "sequence_parameters": {
            "history_frames_L": 10,
            "history_duration_s": 1.0091,
            "prediction_horizon_k": 5,
            "prediction_lead_time_s": 0.50455,
            "sampling_frequency_hz": 9.91
        },
        "features": {
            "feature_count": 7,
            "feature_names": [str(x) for x in feature_names],
            "feature_descriptions": [
                "sin(radians(yaw))",
                "cos(radians(yaw))",
                "pitch (degrees)",
                "causal backward yaw velocity (deg/s)",
                "causal backward pitch velocity (deg/s)",
                "causal backward yaw acceleration (deg/s^2)",
                "causal backward pitch acceleration (deg/s^2)"
            ],
            "causal_difference_type": "strictly causal backward difference",
            "acceleration_boundary_policy": "acc[0]=0.0, acc[1]=0.0 (no unobserved velocity prior to t=0)",
            "angular_wrap_handling": "shortest circular difference on [-180, 180]"
        },
        "normalization": {
            "mean": [float(x) for x in norm_mean],
            "std": [float(x) for x in norm_std],
            "pitch_mean": float(pitch_mean),
            "pitch_std": float(pitch_std),
            "fitted_on": f"training_set_only ({len(train_subs)} subjects)"
        },
        "training_hyperparameters": {
            "optimizer": "Adam",
            "learning_rate": float(lr),
            "batch_size": int(batch_size),
            "loss_function": "MSELoss",
            "max_epochs": int(max_epochs),
            "patience": int(patience),
            "best_epoch": int(best_epoch),
            "training_duration_s": round(float(train_duration), 2),
            "early_stopping_metric": "validation_loss_mse"
        },
        "subject_split": {
            "train_subjects": [int(x) for x in train_subs],
            "validation_subjects": [int(x) for x in val_subs],
            "test_subjects": [int(x) for x in test_subs]
        },
        "evaluation_metrics": {
            "test_spherical_mae_deg": float(final_metrics['sph_mae']),
            "test_spherical_median_deg": float(final_metrics['sph_median']),
            "test_spherical_p90_deg": float(final_metrics['sph_p90']),
            "test_spherical_rmse_deg": float(final_metrics['sph_rmse']),
            "test_yaw_mae_deg": float(final_metrics['yaw_mae']),
            "test_pitch_mae_deg": float(final_metrics['pitch_mae']),
            "improvement_over_persistence_pct": float(imp_pers),
            "improvement_over_const_vel_pct": float(imp_cvel),
            "inference_latency_mean_us": float(lat_mean_us),
            "inference_latency_median_us": float(lat_med_us)
        }
    }

    config_path = SAVED_MODELS_DIR / "final_model_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(final_config, f, indent=4)
    print(f"Saved final model config JSON to: {config_path}")

    # 12. Diagnostic Plots
    print("\n--- 6. Generating Publication-Quality Diagnostic Plots ---")
    # Plot 1: Loss curves
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(range(1, len(history['train_loss']) + 1), history['train_loss'], 'b-o', label='Train Loss (MSE)', linewidth=1.8)
    ax1.plot(range(1, len(history['val_loss']) + 1), history['val_loss'], 'r-s', label='Validation Loss (MSE)', linewidth=1.8)
    ax1.axvline(best_epoch, color='green', linestyle='--', label=f'Best Epoch ({best_epoch})')
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Mean Squared Error Loss', fontsize=11, fontweight='bold')
    ax1.set_title('Final Model Training & Validation Loss Progression\n(GRU-64, lr=0.0005, Causal Parity)', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(RESULTS_PLOTS_DIR / 'final_model_loss_curve.png', dpi=300)
    plt.close()

    # Plot 2: Error CDF Comparison
    fig, ax2 = plt.subplots(figsize=(9, 6))
    max_err = 60.0

    cdf_items = [
        ('Persistence Baseline', pers_metrics['sph_err'], '#7f8c8d', '--'),
        ('Constant Angular Velocity', cvel_metrics['sph_err'], '#e67e22', '-.'),
    ]
    if has_lstm:
        cdf_items.append(('Model A (LSTM-64 Baseline)', lstm_metrics['sph_err'], '#3498db', ':'))
    if has_gru_base:
        cdf_items.append(('Model C (GRU-64 Baseline)', gru_metrics['sph_err'], '#9b59b6', '-.'))
    cdf_items.append(('FINAL OPTIMAL MODEL (GRU-64)', final_metrics['sph_err'], '#27ae60', '-'))

    for name, err_arr, col, ls in cdf_items:
        sorted_err = np.sort(err_arr)
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        ax2.plot(sorted_err, cdf, label=name, color=col, linestyle=ls, linewidth=2.0)

    ax2.set_xlim(0, max_err)
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel('Great-Circle Spherical Angular Error (degrees)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cumulative Distribution Function (CDF)', fontsize=11, fontweight='bold')
    ax2.set_title(f'Cumulative Error Distribution: Final Model vs. Baselines (Test Set, N={len(yd_te):,})', fontsize=12, fontweight='bold')
    ax2.axvline(final_metrics['sph_median'], color='#27ae60', linestyle=':', label=f'Final Median: {final_metrics["sph_median"]:.2f}°')
    ax2.axvline(final_metrics['sph_p90'], color='#27ae60', linestyle='--', label=f'Final P90: {final_metrics["sph_p90"]:.2f}°')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=9, loc='lower right')
    plt.tight_layout()
    plt.savefig(RESULTS_PLOTS_DIR / 'final_model_error_cdf.png', dpi=300)
    plt.close()

    # Plot 3: Bar Comparison of MAE, Median, P90
    fig, ax3 = plt.subplots(figsize=(10, 6))
    methods = [d['Model / Method'].replace(' — ', '\n').split('(')[0].strip() for d in comparison_table]
    maes = [d['Spherical MAE (deg)'] for d in comparison_table]
    meds = [d['Spherical Median (deg)'] for d in comparison_table]
    p90s = [d['Spherical P90 (deg)'] for d in comparison_table]

    x_idx = np.arange(len(methods))
    width = 0.25

    ax3.bar(x_idx - width, maes, width, label='Spherical MAE', color='#3498db', edgecolor='black', alpha=0.85)
    ax3.bar(x_idx, meds, width, label='Spherical Median', color='#2ecc71', edgecolor='black', alpha=0.85)
    ax3.bar(x_idx + width, p90s, width, label='Spherical P90', color='#e74c3c', edgecolor='black', alpha=0.85)

    ax3.set_ylabel('Error (degrees)', fontsize=11, fontweight='bold')
    ax3.set_title('Test Set Error Benchmark: Final Model vs. Baselines', fontsize=13, fontweight='bold')
    ax3.set_xticks(x_idx)
    ax3.set_xticklabels(methods, fontsize=9)
    ax3.grid(axis='y', linestyle=':', alpha=0.6)
    ax3.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(RESULTS_PLOTS_DIR / 'final_model_vs_baselines_mae.png', dpi=300)
    plt.close()

    print("All diagnostic plots generated successfully.")
    print("================================================================================")
    print(" PHASE 4 RETRAINING & EVALUATION COMPLETE!")
    print("================================================================================")


if __name__ == "__main__":
    main()
