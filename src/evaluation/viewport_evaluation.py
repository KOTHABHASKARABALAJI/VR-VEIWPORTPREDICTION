"""
Module: viewport_evaluation.py
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming
Phase: Phase 5, Day 3 — Complete System Evaluation

Implements viewport orientation prediction accuracy evaluation:
- Great-circle orthodromic spherical angular error.
- Circular yaw error (accounting for [-180, +180] wrapping) and pitch error.
- Comprehensive distributional metrics: MAE, RMSE, Median (P50), P90, P95, P99.
- Benchmarking: Final GRU Model vs. Persistence Baseline vs. Constant Velocity Baseline.
- Granular aggregation by unseen test subject and video sequence.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ViewportAccuracyMetrics:
    """
    Immutable container for spherical and coordinate-wise error metrics.
    """
    count: int
    spherical_mae_deg: float
    spherical_median_deg: float
    spherical_p90_deg: float
    spherical_p95_deg: float
    spherical_p99_deg: float
    spherical_rmse_deg: float
    yaw_mae_deg: float
    yaw_rmse_deg: float
    pitch_mae_deg: float
    pitch_rmse_deg: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": int(self.count),
            "spherical_mae_deg": float(self.spherical_mae_deg),
            "spherical_median_deg": float(self.spherical_median_deg),
            "spherical_p90_deg": float(self.spherical_p90_deg),
            "spherical_p95_deg": float(self.spherical_p95_deg),
            "spherical_p99_deg": float(self.spherical_p99_deg),
            "spherical_rmse_deg": float(self.spherical_rmse_deg),
            "yaw_mae_deg": float(self.yaw_mae_deg),
            "yaw_rmse_deg": float(self.yaw_rmse_deg),
            "pitch_mae_deg": float(self.pitch_mae_deg),
            "pitch_rmse_deg": float(self.pitch_rmse_deg),
        }


def compute_spherical_error_vector(
    act_yaw_deg: np.ndarray,
    act_pitch_deg: np.ndarray,
    pred_yaw_deg: np.ndarray,
    pred_pitch_deg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes element-wise spherical error, circular yaw error, and pitch error in degrees.

    Parameters:
        act_yaw_deg: Ground truth yaw in degrees [-180, 180].
        act_pitch_deg: Ground truth pitch in degrees [-90, 90].
        pred_yaw_deg: Predicted yaw in degrees [-180, 180].
        pred_pitch_deg: Predicted pitch in degrees [-90, 90].

    Returns:
        Tuple of (spherical_error_deg, yaw_diff_deg, pitch_diff_deg).
    """
    act_y = np.asarray(act_yaw_deg, dtype=np.float64)
    act_p = np.asarray(act_pitch_deg, dtype=np.float64)
    pred_y = np.asarray(pred_yaw_deg, dtype=np.float64)
    pred_p = np.asarray(pred_pitch_deg, dtype=np.float64)

    # Circular yaw difference shortest path
    yaw_diff = np.abs((pred_y - act_y + 180.0) % 360.0 - 180.0)
    pitch_diff = np.abs(pred_p - act_p)

    # Great-circle angular distance
    p1 = np.radians(act_p)
    y1 = np.radians(act_y)
    p2 = np.radians(pred_p)
    y2 = np.radians(pred_y)

    dot = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(y1 - y2)
    dot_clamped = np.clip(dot, -1.0, 1.0)
    sph_err_deg = np.degrees(np.arccos(dot_clamped))

    return sph_err_deg, yaw_diff, pitch_diff


def compute_accuracy_metrics(
    act_yaw_deg: np.ndarray,
    act_pitch_deg: np.ndarray,
    pred_yaw_deg: np.ndarray,
    pred_pitch_deg: np.ndarray,
) -> ViewportAccuracyMetrics:
    """
    Computes summary accuracy metrics for a given prediction array.
    """
    sph_err, yaw_diff, pitch_diff = compute_spherical_error_vector(
        act_yaw_deg, act_pitch_deg, pred_yaw_deg, pred_pitch_deg
    )
    count = len(sph_err)
    if count == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")

    return ViewportAccuracyMetrics(
        count=count,
        spherical_mae_deg=float(np.mean(sph_err)),
        spherical_median_deg=float(np.median(sph_err)),
        spherical_p90_deg=float(np.percentile(sph_err, 90.0)),
        spherical_p95_deg=float(np.percentile(sph_err, 95.0)),
        spherical_p99_deg=float(np.percentile(sph_err, 99.0)),
        spherical_rmse_deg=float(np.sqrt(np.mean(sph_err ** 2))),
        yaw_mae_deg=float(np.mean(yaw_diff)),
        yaw_rmse_deg=float(np.sqrt(np.mean(yaw_diff ** 2))),
        pitch_mae_deg=float(np.mean(pitch_diff)),
        pitch_rmse_deg=float(np.sqrt(np.mean(pitch_diff ** 2))),
    )


class ViewportPredictionEvaluator:
    """
    Evaluates viewport trajectory predictions from DataFrame records across models,
    subjects, and video sequences.
    """

    def __init__(self, predictions_df: pd.DataFrame) -> None:
        self.df = self._validate_and_prepare_df(predictions_df)

    @staticmethod
    def _validate_and_prepare_df(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"predictions_df must be a pandas DataFrame, got {type(df).__name__}")
        if df.empty:
            raise ValueError("predictions_df cannot be empty.")

        required_cols = [
            "actual_yaw_deg",
            "actual_pitch_deg",
            "predicted_yaw_deg",
            "predicted_pitch_deg",
            "persistence_predicted_yaw_deg",
            "persistence_predicted_pitch_deg",
            "subject_id",
            "video",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in predictions DataFrame: {missing}")

        return df.copy()

    def evaluate_overall(self) -> Dict[str, ViewportAccuracyMetrics]:
        """
        Evaluates overall test-set accuracy for GRU Model, Persistence Baseline,
        and Constant Velocity Baseline (if available).
        """
        results: Dict[str, ViewportAccuracyMetrics] = {}

        # 1. Final GRU Model
        results["final_gru"] = compute_accuracy_metrics(
            act_yaw_deg=self.df["actual_yaw_deg"].values,
            act_pitch_deg=self.df["actual_pitch_deg"].values,
            pred_yaw_deg=self.df["predicted_yaw_deg"].values,
            pred_pitch_deg=self.df["predicted_pitch_deg"].values,
        )

        # 2. Persistence Baseline
        results["persistence"] = compute_accuracy_metrics(
            act_yaw_deg=self.df["actual_yaw_deg"].values,
            act_pitch_deg=self.df["actual_pitch_deg"].values,
            pred_yaw_deg=self.df["persistence_predicted_yaw_deg"].values,
            pred_pitch_deg=self.df["persistence_predicted_pitch_deg"].values,
        )

        return results

    def evaluate_by_subject(self) -> pd.DataFrame:
        """
        Evaluates prediction performance broken down by unseen subject.
        """
        rows = []
        for sub_id, group in self.df.groupby("subject_id"):
            gru_m = compute_accuracy_metrics(
                group["actual_yaw_deg"].values,
                group["actual_pitch_deg"].values,
                group["predicted_yaw_deg"].values,
                group["predicted_pitch_deg"].values,
            )
            pers_m = compute_accuracy_metrics(
                group["actual_yaw_deg"].values,
                group["actual_pitch_deg"].values,
                group["persistence_predicted_yaw_deg"].values,
                group["persistence_predicted_pitch_deg"].values,
            )
            mae_reduct_pct = ((pers_m.spherical_mae_deg - gru_m.spherical_mae_deg) / pers_m.spherical_mae_deg) * 100.0
            rows.append({
                "subject_id": int(sub_id),
                "sample_count": len(group),
                "gru_spherical_mae_deg": gru_m.spherical_mae_deg,
                "gru_spherical_median_deg": gru_m.spherical_median_deg,
                "gru_spherical_p90_deg": gru_m.spherical_p90_deg,
                "gru_spherical_rmse_deg": gru_m.spherical_rmse_deg,
                "gru_yaw_mae_deg": gru_m.yaw_mae_deg,
                "gru_pitch_mae_deg": gru_m.pitch_mae_deg,
                "persistence_spherical_mae_deg": pers_m.spherical_mae_deg,
                "persistence_spherical_p90_deg": pers_m.spherical_p90_deg,
                "spherical_mae_reduction_pct": mae_reduct_pct,
            })

        res_df = pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)
        return res_df

    def evaluate_by_video(self) -> pd.DataFrame:
        """
        Evaluates prediction performance broken down by video sequence.
        """
        rows = []
        for vid, group in self.df.groupby("video"):
            gru_m = compute_accuracy_metrics(
                group["actual_yaw_deg"].values,
                group["actual_pitch_deg"].values,
                group["predicted_yaw_deg"].values,
                group["predicted_pitch_deg"].values,
            )
            pers_m = compute_accuracy_metrics(
                group["actual_yaw_deg"].values,
                group["actual_pitch_deg"].values,
                group["persistence_predicted_yaw_deg"].values,
                group["persistence_predicted_pitch_deg"].values,
            )
            mae_reduct_pct = ((pers_m.spherical_mae_deg - gru_m.spherical_mae_deg) / pers_m.spherical_mae_deg) * 100.0
            rows.append({
                "video": str(vid),
                "sample_count": len(group),
                "gru_spherical_mae_deg": gru_m.spherical_mae_deg,
                "gru_spherical_median_deg": gru_m.spherical_median_deg,
                "gru_spherical_p90_deg": gru_m.spherical_p90_deg,
                "gru_spherical_rmse_deg": gru_m.spherical_rmse_deg,
                "persistence_spherical_mae_deg": pers_m.spherical_mae_deg,
                "spherical_mae_reduction_pct": mae_reduct_pct,
            })

        res_df = pd.DataFrame(rows).sort_values("video").reset_index(drop=True)
        return res_df

    def get_summary_comparison_table(self) -> pd.DataFrame:
        """
        Generates a consolidated comparison table between GRU Model and Persistence Baseline.
        """
        overall = self.evaluate_overall()
        gru = overall["final_gru"]
        pers = overall["persistence"]

        records = [
            {
                "Model / Baseline": "Final GRU-64 Model",
                "Spherical MAE (deg)": f"{gru.spherical_mae_deg:.4f}",
                "Median Error (deg)": f"{gru.spherical_median_deg:.4f}",
                "90th Percentile (deg)": f"{gru.spherical_p90_deg:.4f}",
                "95th Percentile (deg)": f"{gru.spherical_p95_deg:.4f}",
                "RMSE (deg)": f"{gru.spherical_rmse_deg:.4f}",
                "Yaw MAE (deg)": f"{gru.yaw_mae_deg:.4f}",
                "Pitch MAE (deg)": f"{gru.pitch_mae_deg:.4f}",
                "MAE Reduction vs Pers (%)": f"{((pers.spherical_mae_deg - gru.spherical_mae_deg) / pers.spherical_mae_deg * 100.0):+.2f}%",
            },
            {
                "Model / Baseline": "Persistence Baseline",
                "Spherical MAE (deg)": f"{pers.spherical_mae_deg:.4f}",
                "Median Error (deg)": f"{pers.spherical_median_deg:.4f}",
                "90th Percentile (deg)": f"{pers.spherical_p90_deg:.4f}",
                "95th Percentile (deg)": f"{pers.spherical_p95_deg:.4f}",
                "RMSE (deg)": f"{pers.spherical_rmse_deg:.4f}",
                "Yaw MAE (deg)": f"{pers.yaw_mae_deg:.4f}",
                "Pitch MAE (deg)": f"{pers.pitch_mae_deg:.4f}",
                "MAE Reduction vs Pers (%)": "—",
            },
        ]
        return pd.DataFrame(records)

