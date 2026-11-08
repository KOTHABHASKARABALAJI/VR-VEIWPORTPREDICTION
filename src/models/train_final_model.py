"""
Script: train_final_model.py
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming

Phase 4: Retrains and evaluates the verified, optimal final viewport prediction model:
- Architecture: ConfigurableGRU (input_dim=7, hidden_dim=64, proj_dim=32, output_dim=3, dropout=0.0)
- Sequence Length: L = 10 frames (1.0s historical context at ~10 Hz)
- Prediction Horizon: k = 5 frames (~0.5s lead time)
- Feature Representation: 7 Kinematic Features (sin_yaw, cos_yaw, pitch, yaw_vel, pitch_vel, yaw_acc, pitch_acc)
- Optimization: Adam (lr=0.0005, batch_size=256, MSE loss on [sin, cos, norm_pitch])
- Early Stopping: Monitored strictly on Validation MSE Loss (patience=5 epochs)
- Subject Partitioning: 33 Train, 7 Validation, 8 Test disjoint subjects (Zero leakage)

Outputs:
- Checkpoint: saved_models/final_viewport_model.pth
- Configuration JSON: saved_models/final_model_config.json
- Test Predictions: results/predictions/final_model_test_predictions.csv
- Metrics CSV: results/metrics/final_model_evaluation_metrics.csv
- Comparison CSV: results/metrics/final_model_comparison.csv
- Diagnostic Plots: results/plots/final_model_loss_curve.png, results/plots/final_model_error_cdf.png, results/plots/final_model_vs_baselines_mae.png
"""

import sys
import json
import time
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
from src.feature_engineering.features import extract_7_features

DATA_DIR = PROJECT_ROOT / "data" / "processed"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
RESULTS_METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
RESULTS_PREDS_DIR = PROJECT_ROOT / "results" / "predictions"
RESULTS_PLOTS_DIR = PROJECT_ROOT / "results" / "plots"

SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PREDS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_spherical_error_arrays(act_yaw, act_pitch, pred_yaw, pred_pitch):
    """Computes pointwise angular differences and orthodromic great-circle distance in degrees."""
    yaw_diff = np.abs((pred_yaw - act_yaw + 180.0) % 360.0 - 180.0)
    pitch_diff = np.abs(pred_pitch - act_pitch)

    p1 = np.radians(act_pitch)
    y1 = np.radians(act_yaw)
    p2 = np.radians(pred_pitch)
    y2 = np.radians(pred_yaw)

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


def main():
    print("================================================================================")
    print(" PHASE 4: FINAL MODEL RETRAINING & COMPREHENSIVE BENCHMARK")
    print("================================================================================")

    # 1. Load Causal Datasets
    print("\n--- 1. Loading Causal Preprocessed Datasets ---")
    tr_npz = np.load(DATA_DIR / 'lstm_train.npz')
    va_npz = np.load(DATA_DIR / 'lstm_validation.npz')
    te_npz = np.load(DATA_DIR / 'lstm_test.npz')

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
    feature_names = list(tr_npz['feature_names'])

    test_subs = te_npz['subject_ids']
    test_vids = te_npz['video_names']

    print(f"Train split:      {X_tr.shape[0]:,} samples (33 subjects)")
    print(f"Validation split: {X_va.shape[0]:,} samples (7 subjects)")
    print(f"Test split:       {X_te.shape[0]:,} samples (8 subjects)")
    print(f"Feature names:    {feature_names}")
    print(f"Norm Mean:        {np.round(norm_mean, 4)}")
    print(f"Norm Std:         {np.round(norm_std, 4)}")

    # Prepare normalized targets: [sin_yaw, cos_yaw, norm_pitch]
    pitch_mean = float(norm_mean[2])
    pitch_std = float(norm_std[2])

    y_tr_norm = np.column_stack([y_tr[:, 0], y_tr[:, 1], (y_tr[:, 2] - pitch_mean) / pitch_std])
    y_va_norm = np.column_stack([y_va[:, 0], y_va[:, 1], (y_va[:, 2] - pitch_mean) / pitch_std])
    y_te_norm = np.column_stack([y_te[:, 0], y_te[:, 1], (y_te[:, 2] - pitch_mean) / pitch_std])

    # 2. Dataloaders
    batch_size = 256
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr_norm, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_va, dtype=torch.float32), torch.tensor(y_va_norm, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_te, dtype=torch.float32), torch.tensor(y_te_norm, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False
    )

    # 3. Model Architecture Instantiation
    print("\n--- 2. Instantiating Optimal Model Architecture ---")
    model = ConfigurableGRU(input_dim=7, hidden_dim=64, dropout=0.0)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Architecture: ConfigurableGRU (input_dim=7, hidden_dim=64, dropout=0.0)")
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
    best_weights = None
    epochs_no_improve = 0

    history = {'train_loss': [], 'val_loss': [], 'val_sph_mae': []}
    start_train_time = time.time()

    for epoch in range(1, max_epochs + 1):
        # Training Phase
        model.train()
        running_tr_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            running_tr_loss += loss.item() * len(bx)
        tr_loss = running_tr_loss / len(X_tr)

        # Validation Phase
        model.eval()
        running_va_loss = 0.0
        val_preds = []
        with torch.no_grad():
            for bx, by in val_loader:
                p = model(bx)
                loss = criterion(p, by)
                running_va_loss += loss.item() * len(bx)
                val_preds.append(p.numpy())
        va_loss = running_va_loss / len(X_va)
        val_preds = np.vstack(val_preds)

        # Decode validation spherical MAE
        v_sin, v_cos = val_preds[:, 0], val_preds[:, 1]
        v_yaw = np.degrees(np.arctan2(v_sin, v_cos))
        v_pitch = val_preds[:, 2] * pitch_std + pitch_mean
        va_metrics = compute_spherical_error_arrays(yd_va[:, 0], yd_va[:, 1], v_yaw, v_pitch)
        va_mae = va_metrics['sph_mae']

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['val_sph_mae'].append(va_mae)

        print(f"Epoch {epoch:02d}/{max_epochs:02d} | Train Loss: {tr_loss:.6f} | Val Loss: {va_loss:.6f} | Val Sph MAE: {va_mae:.4f}°", end="")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print("  [Best Model Checkpointed]")
        else:
            epochs_no_improve += 1
            print(f"  [Patience: {epochs_no_improve}/{patience}]")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}. Restoring weights from epoch {best_epoch}.")
            break

    train_duration = time.time() - start_train_time
    print(f"\nTraining completed in {train_duration:.2f} seconds.")
    print(f"Best Validation Loss: {best_val_loss:.6f} at Epoch {best_epoch}")

    # Load best weights
    model.load_state_dict(best_weights)

    # 5. Measure Inference Latency
    model.eval()
    sample_input = torch.tensor(X_te[:1], dtype=torch.float32)
    # Warmup
    for _ in range(100):
        _ = model(sample_input)
    # Benchmark 2000 runs
    lat_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(2000):
            _ = model(sample_input)
    lat_end = time.perf_counter()
    latency_us_per_sample = ((lat_end - lat_start) / 2000) * 1e6
    print(f"Inference Latency: {latency_us_per_sample:.2f} µs per sample")

    # 6. Save Checkpoint to saved_models/final_viewport_model.pth
    ckpt_path = SAVED_MODELS_DIR / "final_viewport_model.pth"
    torch.save(best_weights, ckpt_path)
    model_size_kb = ckpt_path.stat().st_size / 1024
    print(f"Saved final model checkpoint to: {ckpt_path} ({model_size_kb:.1f} KB)")

    # 7. Final Evaluation on Untouched Test Set
    print("\n--- 4. Evaluating Final Model on Untouched Test Set (45,125 Samples) ---")
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

    # B. Constant Velocity Baseline
    yaw_vel_test = X_te[:, -1, 3] * norm_std[3] + norm_mean[3]
    pitch_vel_test = X_te[:, -1, 4] * norm_std[4] + norm_mean[4]
    lead_time_s = 5 * 0.10091 # ~0.50455s
    cvel_yaw = (pers_yaw + yaw_vel_test * lead_time_s + 180.0) % 360.0 - 180.0
    cvel_pitch = np.clip(pers_pitch + pitch_vel_test * lead_time_s, -90.0, 90.0)
    cvel_metrics = compute_spherical_error_arrays(yd_te[:, 0], yd_te[:, 1], cvel_yaw, cvel_pitch)

    # C. Existing Model A (LSTM-64 Baseline)
    m_lstm = ModelA_LSTM64()
    m_lstm.load_state_dict(torch.load(SAVED_MODELS_DIR / 'viewport_lstm_model.pth', weights_only=True))
    m_lstm.eval()
    with torch.no_grad():
        lstm_preds = m_lstm(torch.tensor(X_te, dtype=torch.float32)).numpy()
    l_yaw = np.degrees(np.arctan2(lstm_preds[:, 0], lstm_preds[:, 1]))
    l_pitch = lstm_preds[:, 2] * pitch_std + pitch_mean
    lstm_metrics = compute_spherical_error_arrays(yd_te[:, 0], yd_te[:, 1], l_yaw, l_pitch)

    # D. Existing Model C (GRU-64 Baseline)
    m_gru = ModelC_GRU64()
    m_gru.load_state_dict(torch.load(SAVED_MODELS_DIR / 'best_viewport_model.pth', weights_only=True))
    m_gru.eval()
    with torch.no_grad():
        gru_preds = m_gru(torch.tensor(X_te, dtype=torch.float32)).numpy()
    g_yaw = np.degrees(np.arctan2(gru_preds[:, 0], gru_preds[:, 1]))
    g_pitch = gru_preds[:, 2] * pitch_std + pitch_mean
    gru_metrics = compute_spherical_error_arrays(yd_te[:, 0], yd_te[:, 1], g_yaw, g_pitch)

    imp_pers = (pers_metrics['sph_mae'] - final_metrics['sph_mae']) / pers_metrics['sph_mae'] * 100
    imp_cvel = (cvel_metrics['sph_mae'] - final_metrics['sph_mae']) / cvel_metrics['sph_mae'] * 100
    imp_lstm = (lstm_metrics['sph_mae'] - final_metrics['sph_mae']) / lstm_metrics['sph_mae'] * 100
    imp_gru = (gru_metrics['sph_mae'] - final_metrics['sph_mae']) / gru_metrics['sph_mae'] * 100

    print(f"Improvement over Persistence Baseline:           {imp_pers:+.2f}%")
    print(f"Improvement over Constant Velocity Baseline:     {imp_cvel:+.2f}%")
    print(f"Improvement over Existing LSTM-64 Checkpoint:    {imp_lstm:+.2f}%")
    print(f"Improvement over Existing GRU-64 Checkpoint:     {imp_gru:+.2f}%")

    # 9. Save Test Predictions CSV
    df_preds = pd.DataFrame({
        'sample_index': np.arange(len(yd_te)),
        'subject_id': test_subs,
        'video': test_vids,
        'actual_yaw_deg': yd_te[:, 0],
        'actual_pitch_deg': yd_te[:, 1],
        'predicted_yaw_deg': final_pred_yaw,
        'predicted_pitch_deg': final_pred_pitch,
        'spherical_error_deg': final_metrics['sph_err'],
        'persistence_sph_err_deg': pers_metrics['sph_err'],
        'const_vel_sph_err_deg': cvel_metrics['sph_err']
    })
    preds_csv_path = RESULTS_PREDS_DIR / "final_model_test_predictions.csv"
    df_preds.to_csv(preds_csv_path, index=False)
    print(f"Saved test predictions to: {preds_csv_path}")

    # 10. Save Detailed Final Comparison Table CSV
    comparison_table = [
        {
            'Model / Method': 'Persistence Baseline (Last-Known)',
            'Method Type': 'Heuristic Baseline',
            'Parameters': 0,
            'Inference Latency (us)': 0.1,
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
            'Inference Latency (us)': 0.5,
            'Spherical MAE (deg)': cvel_metrics['sph_mae'],
            'Spherical Median (deg)': cvel_metrics['sph_median'],
            'Spherical P90 (deg)': cvel_metrics['sph_p90'],
            'Spherical RMSE (deg)': cvel_metrics['sph_rmse'],
            'Yaw MAE (deg)': cvel_metrics['yaw_mae'],
            'Pitch MAE (deg)': cvel_metrics['pitch_mae'],
            'Error Reduction vs Persistence (%)': (pers_metrics['sph_mae'] - cvel_metrics['sph_mae']) / pers_metrics['sph_mae'] * 100
        },
        {
            'Model / Method': 'Model A — LSTM-64 Baseline',
            'Method Type': 'Deep Recurrent NN',
            'Parameters': 20835,
            'Inference Latency (us)': 12.8,
            'Spherical MAE (deg)': lstm_metrics['sph_mae'],
            'Spherical Median (deg)': lstm_metrics['sph_median'],
            'Spherical P90 (deg)': lstm_metrics['sph_p90'],
            'Spherical RMSE (deg)': lstm_metrics['sph_rmse'],
            'Yaw MAE (deg)': lstm_metrics['yaw_mae'],
            'Pitch MAE (deg)': lstm_metrics['pitch_mae'],
            'Error Reduction vs Persistence (%)': (pers_metrics['sph_mae'] - lstm_metrics['sph_mae']) / pers_metrics['sph_mae'] * 100
        },
        {
            'Model / Method': 'Model C — GRU-64 Baseline',
            'Method Type': 'Deep Recurrent NN',
            'Parameters': 16195,
            'Inference Latency (us)': 11.4,
            'Spherical MAE (deg)': gru_metrics['sph_mae'],
            'Spherical Median (deg)': gru_metrics['sph_median'],
            'Spherical P90 (deg)': gru_metrics['sph_p90'],
            'Spherical RMSE (deg)': gru_metrics['sph_rmse'],
            'Yaw MAE (deg)': gru_metrics['yaw_mae'],
            'Pitch MAE (deg)': gru_metrics['pitch_mae'],
            'Error Reduction vs Persistence (%)': (pers_metrics['sph_mae'] - gru_metrics['sph_mae']) / pers_metrics['sph_mae'] * 100
        },
        {
            'Model / Method': 'FINAL OPTIMAL MODEL — GRU-64 (Tuned lr=0.0005, Causal)',
            'Method Type': 'Deep Recurrent NN (Optimal)',
            'Parameters': param_count,
            'Inference Latency (us)': latency_us_per_sample,
            'Spherical MAE (deg)': final_metrics['sph_mae'],
            'Spherical Median (deg)': final_metrics['sph_median'],
            'Spherical P90 (deg)': final_metrics['sph_p90'],
            'Spherical RMSE (deg)': final_metrics['sph_rmse'],
            'Yaw MAE (deg)': final_metrics['yaw_mae'],
            'Pitch MAE (deg)': final_metrics['pitch_mae'],
            'Error Reduction vs Persistence (%)': imp_pers
        }
    ]

    df_comp = pd.DataFrame(comparison_table)
    comp_csv_path = RESULTS_METRICS_DIR / "final_model_comparison.csv"
    df_comp.to_csv(comp_csv_path, index=False)
    print(f"Saved final comparison table to: {comp_csv_path}")

    # 11. Save Configuration JSON
    final_config = {
        "model_name": "final_viewport_model",
        "project_title": "Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360 Video Streaming",
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
            "causal_difference_type": "backward difference",
            "angular_wrap_handling": "shortest circular difference on [-180, 180]"
        },
        "normalization": {
            "mean": [float(x) for x in norm_mean],
            "std": [float(x) for x in norm_std],
            "pitch_mean": float(pitch_mean),
            "pitch_std": float(pitch_std),
            "fitted_on": "training_set_only (33 subjects)"
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
            "train_subjects": [int(x) for x in [2, 3, 5, 6, 7, 8, 10, 11, 13, 14, 15, 17, 18, 19, 21, 23, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 39, 41, 42, 45, 46, 47, 48]],
            "validation_subjects": [int(x) for x in [4, 12, 24, 25, 37, 38, 43]],
            "test_subjects": [int(x) for x in [9, 16, 20, 22, 30, 40, 44, 49]]
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
            "inference_latency_us": float(latency_us_per_sample)
        }
    }

    def default_json(o):
        if isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.integer, np.int32, np.int64)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    config_path = SAVED_MODELS_DIR / "final_model_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(final_config, f, indent=4, default=default_json)
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
    ax1.set_title('Final Model Training & Validation Loss Progression\n(GRU-64, lr=0.0005, Causal Pipeline)', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(RESULTS_PLOTS_DIR / 'final_model_loss_curve.png', dpi=300)
    plt.close()

    # Plot 2: Error CDF Comparison
    fig, ax2 = plt.subplots(figsize=(9, 6))
    max_err = 60.0
    bins = np.linspace(0, max_err, 500)

    for name, err_arr, col, ls in [
        ('Persistence Baseline', pers_metrics['sph_err'], '#7f8c8d', '--'),
        ('Constant Angular Velocity', cvel_metrics['sph_err'], '#e67e22', '-.'),
        ('Model A (LSTM-64 Baseline)', lstm_metrics['sph_err'], '#3498db', ':'),
        ('FINAL OPTIMAL MODEL (GRU-64)', final_metrics['sph_err'], '#27ae60', '-')
    ]:
        sorted_err = np.sort(err_arr)
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        ax2.plot(sorted_err, cdf, label=name, color=col, linestyle=ls, linewidth=2.0)

    ax2.set_xlim(0, max_err)
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel('Great-Circle Spherical Angular Error (degrees)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cumulative Distribution Function (CDF)', fontsize=11, fontweight='bold')
    ax2.set_title('Cumulative Error Distribution: Final Model vs. Baselines (Test Set, N=45,125)', fontsize=12, fontweight='bold')
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
