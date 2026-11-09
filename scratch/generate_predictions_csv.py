"""
Script: scratch/generate_predictions_csv.py
Regenerates results/predictions/final_model_test_predictions.csv using the exact
saved checkpoint saved_models/final_viewport_model.pth without retraining.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.architectures import ConfigurableGRU

def compute_spherical_error_arrays(act_yaw, act_pitch, pred_yaw, pred_pitch):
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
        'pitch_mae': float(np.mean(pitch_diff)),
    }

def main():
    data_path = PROJECT_ROOT / "data" / "processed" / "lstm_test.npz"
    ckpt_path = PROJECT_ROOT / "saved_models" / "final_viewport_model.pth"
    traj_path = PROJECT_ROOT / "data" / "processed" / "viewport_trajectories.csv"
    out_path = PROJECT_ROOT / "results" / "predictions" / "final_model_test_predictions.csv"

    print(f"Loading test dataset from: {data_path}")
    test_data = np.load(data_path)
    X_te = test_data['X']
    yd_te = test_data['y_deg']
    norm_mean = test_data['norm_mean']
    norm_std = test_data['norm_std']
    test_subs = test_data['subject_ids']
    test_vids = test_data['video_names']

    pitch_mean = norm_mean[2]
    pitch_std = norm_std[2]

    print(f"Loading final model checkpoint from: {ckpt_path}")
    model = ConfigurableGRU(input_dim=7, hidden_dim=64, dropout=0.0)
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    print("Running inference on test set (45,125 samples)...")
    with torch.no_grad():
        preds = model(torch.tensor(X_te, dtype=torch.float32)).numpy()

    t_sin, t_cos = preds[:, 0], preds[:, 1]
    final_pred_yaw = np.degrees(np.arctan2(t_sin, t_cos))
    final_pred_pitch = preds[:, 2] * pitch_std + pitch_mean

    final_metrics = compute_spherical_error_arrays(
        yd_te[:, 0], yd_te[:, 1], final_pred_yaw, final_pred_pitch
    )
    print(f"Final Model Test Performance:")
    print(f"  Spherical MAE:    {final_metrics['sph_mae']:.4f}°")
    print(f"  Spherical Median: {final_metrics['sph_median']:.4f}°")
    print(f"  Spherical P90:    {final_metrics['sph_p90']:.4f}°")
    print(f"  Spherical RMSE:   {final_metrics['sph_rmse']:.4f}°")

    # Persistence Baseline
    sin_last = X_te[:, -1, 0] * norm_std[0] + norm_mean[0]
    cos_last = X_te[:, -1, 1] * norm_std[1] + norm_mean[1]
    pers_yaw = np.degrees(np.arctan2(sin_last, cos_last))
    pers_pitch = X_te[:, -1, 2] * norm_std[2] + norm_mean[2]
    pers_metrics = compute_spherical_error_arrays(yd_te[:, 0], yd_te[:, 1], pers_yaw, pers_pitch)
    print(f"Persistence Baseline Test Spherical MAE: {pers_metrics['sph_mae']:.4f}°")

    # Constant Velocity Baseline
    yaw_vel_test = X_te[:, -1, 3] * norm_std[3] + norm_mean[3]
    pitch_vel_test = X_te[:, -1, 4] * norm_std[4] + norm_mean[4]
    lead_time_s = 5 * 0.10091 # ~0.50455s
    cvel_yaw = (pers_yaw + yaw_vel_test * lead_time_s + 180.0) % 360.0 - 180.0
    cvel_pitch = np.clip(pers_pitch + pitch_vel_test * lead_time_s, -90.0, 90.0)
    cvel_metrics = compute_spherical_error_arrays(yd_te[:, 0], yd_te[:, 1], cvel_yaw, cvel_pitch)
    print(f"Constant Velocity Baseline Test Spherical MAE: {cvel_metrics['sph_mae']:.4f}°")

    # Temporal & Trajectory Mapping
    if traj_path.exists():
        v_df = pd.read_csv(traj_path)
        traj_dict = {}
        for (sub, vid), group in v_df.groupby(["subject_id", "video"]):
            traj_dict[(sub, vid)] = group["timestamp"].values

        target_timestamps = np.empty(len(test_subs), dtype=np.float64)
        trajectory_ids = []
        start = 0
        curr = (test_subs[0], test_vids[0])
        for i in range(1, len(test_subs)):
            pair = (test_subs[i], test_vids[i])
            if pair != curr:
                count = i - start
                ts_arr = traj_dict[curr]
                target_timestamps[start:i] = ts_arr[14 : 14 + count]
                vid_clean = str(curr[1]).replace(".mp4", "")
                trajectory_ids.extend([f"sub_{curr[0]}_vid_{vid_clean}"] * count)
                curr = pair
                start = i
        count = len(test_subs) - start
        ts_arr = traj_dict[curr]
        target_timestamps[start:] = ts_arr[14 : 14 + count]
        vid_clean = str(curr[1]).replace(".mp4", "")
        trajectory_ids.extend([f"sub_{curr[0]}_vid_{vid_clean}"] * count)
    else:
        target_timestamps = np.full(len(test_subs), np.nan)
        trajectory_ids = [f"sub_{s}_vid_{str(v).replace('.mp4', '')}" for s, v in zip(test_subs, test_vids)]

    df_preds = pd.DataFrame({
        'sample_index': np.arange(len(yd_te)),
        'subject_id': test_subs,
        'video': test_vids,
        'trajectory_id': trajectory_ids,
        'target_timestamp': target_timestamps,
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_preds.to_csv(out_path, index=False)
    print(f"Successfully saved {len(df_preds):,} predictions to: {out_path}")
    print(f"Columns: {list(df_preds.columns)}")

if __name__ == "__main__":
    main()
