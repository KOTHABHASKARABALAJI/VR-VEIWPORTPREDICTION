"""
Script: evaluate_tile_prediction.py
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming

Phase 5: Evaluates tile-level prediction performance on the test set:
- Compares Final GRU Model predictions vs. Ground Truth Viewports.
- Compares against the Persistence Baseline using exact persistence coordinates.
- Evaluates the effect of 1-ring Guard Band Margin tiles.
- Metrics computed separately for all strategies:
  1. Tile Hit Ratio / Recall: Fraction of visible tiles accurately predicted and fetched.
  2. Tile Precision: Fraction of fetched tiles that are visible (avoiding wasted spatial area).
  3. Sampled Viewport Coverage: Percentage of user visual FoV area covered by fetched tiles (Mean, P10, P90).
  4. Average Active Tiles per Frame (out of 24).
  5. Tile Area Ratio (%): Fraction of 24 spatial ERP tiles fetched (spatial area ratio, not compressed network bandwidth).
- Granular subject-wise evaluation across all unseen test subjects.
"""

import sys
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tiling.grid_partitioner import TileGridPartitioner
from src.tiling.viewport_mapper import ViewportMapper

REQUIRED_COLUMNS = [
    "actual_yaw_deg",
    "actual_pitch_deg",
    "predicted_yaw_deg",
    "predicted_pitch_deg",
    "persistence_predicted_yaw_deg",
    "persistence_predicted_pitch_deg",
    "subject_id",
]


def validate_predictions_df(
    df: pd.DataFrame,
    sample_stride: int = 5,
    hfov_deg: float = 100.0,
    vfov_deg: float = 100.0
) -> None:
    """
    Validates the structure, types, and mathematical domains of the predictions DataFrame
    and evaluation configuration parameters.

    Raises:
        TypeError: If arguments are of invalid type or boolean where numeric is expected.
        ValueError: If arguments violate mathematical or domain constraints.
    """
    # Reject booleans explicitly for numeric parameters
    if isinstance(sample_stride, bool) or not isinstance(sample_stride, int) or sample_stride < 1:
        raise ValueError(
            f"sample_stride must be a positive integer >= 1, got {sample_stride} (type: {type(sample_stride).__name__})"
        )

    if isinstance(hfov_deg, bool) or not isinstance(hfov_deg, (int, float)) or not (0.0 < float(hfov_deg) < 180.0):
        raise ValueError(f"hfov_deg must be strictly between 0° and 180° exclusive, got: {hfov_deg}")

    if isinstance(vfov_deg, bool) or not isinstance(vfov_deg, (int, float)) or not (0.0 < float(vfov_deg) < 180.0):
        raise ValueError(f"vfov_deg must be strictly between 0° and 180° exclusive, got: {vfov_deg}")

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")

    if df.empty:
        raise ValueError("Prediction DataFrame is empty.")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in prediction DataFrame: {missing_cols}")

    # Validate numeric types before checking isfinite
    for col in REQUIRED_COLUMNS:
        if not np.issubdtype(df[col].dtype, np.number):
            raise TypeError(f"Column '{col}' must have numeric dtype, got {df[col].dtype}")
        vals = df[col].to_numpy()
        if not np.all(np.isfinite(vals)):
            raise ValueError(f"Column '{col}' contains non-finite values (NaN or Inf).")

    # Validate subject_id
    if (df["subject_id"] <= 0).any():
        raise ValueError("subject_id must contain positive subject identifiers.")

    # Range checks for pitch angles (equirectangular pitch must be in [-90, +90])
    pitch_cols = ["actual_pitch_deg", "predicted_pitch_deg", "persistence_predicted_pitch_deg"]
    for col in pitch_cols:
        if (df[col] < -90.001).any() or (df[col] > 90.001).any():
            raise ValueError(
                f"Column '{col}' has pitch values outside valid domain [-90°, 90°]: "
                f"min={df[col].min():.4f}, max={df[col].max():.4f}"
            )

    # Range checks for yaw angles (spherical domain [-360, +360])
    yaw_cols = ["actual_yaw_deg", "predicted_yaw_deg", "persistence_predicted_yaw_deg"]
    for col in yaw_cols:
        if (df[col] < -360.001).any() or (df[col] > 360.001).any():
            raise ValueError(
                f"Column '{col}' has yaw values outside valid spherical domain [-360°, 360°]: "
                f"min={df[col].min():.4f}, max={df[col].max():.4f}"
            )


def run_tile_evaluation(
    predictions_csv_path: Path | str,
    output_dir: Path | str,
    sample_stride: int = 5,
    hfov_deg: float = 100.0,
    vfov_deg: float = 100.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates tile intersection metrics on test predictions across all strategies and subjects.

    Parameters:
        predictions_csv_path: Path to final_model_test_predictions.csv.
        output_dir: Destination folder for output CSVs and summary reports (results/tiling).
        sample_stride: Subsampling stride to balance speed and statistical power (default: 5 -> ~9,025 samples).
        hfov_deg, vfov_deg: Viewport Field of View dimensions in degrees (strictly in (0, 180)).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (overall_summary_df, subject_wise_df)
    """
    predictions_csv_path = Path(predictions_csv_path)
    output_dir = Path(output_dir)

    if not predictions_csv_path.exists():
        raise FileNotFoundError(f"Predictions CSV file not found at: {predictions_csv_path}")

    if predictions_csv_path.stat().st_size == 0:
        raise ValueError(f"Predictions CSV file is empty: {predictions_csv_path}")

    print(f"Loading predictions from: {predictions_csv_path}")
    df = pd.read_csv(predictions_csv_path)

    # Validate inputs and data integrity
    validate_predictions_df(df, sample_stride=sample_stride, hfov_deg=hfov_deg, vfov_deg=vfov_deg)

    total_samples = len(df)
    eval_df = df.iloc[::sample_stride].copy().reset_index(drop=True)
    num_eval = len(eval_df)
    print(f"Evaluating {num_eval:,} samples (stride={sample_stride} from {total_samples:,} total test samples)...")

    grid = TileGridPartitioner()
    mapper = ViewportMapper(grid_partitioner=grid, hfov_deg=hfov_deg, vfov_deg=vfov_deg, ray_samples_per_axis=21)

    # Independent metrics storage per strategy
    model_recalls: List[float] = []
    model_precisions: List[float] = []
    model_coverages: List[float] = []
    model_active_counts: List[int] = []

    model_margin_recalls: List[float] = []
    model_margin_precisions: List[float] = []
    model_margin_coverages: List[float] = []
    model_margin_counts: List[int] = []

    persist_recalls: List[float] = []
    persist_precisions: List[float] = []
    persist_coverages: List[float] = []
    persist_active_counts: List[int] = []

    eval_subjects = eval_df["subject_id"].values

    start_time = time.time()

    for idx, row in eval_df.iterrows():
        act_y, act_p = float(row["actual_yaw_deg"]), float(row["actual_pitch_deg"])
        pred_y, pred_p = float(row["predicted_yaw_deg"]), float(row["predicted_pitch_deg"])
        pers_y, pers_p = float(row["persistence_predicted_yaw_deg"]), float(row["persistence_predicted_pitch_deg"])

        # 1. Final GRU Model (No Guard Band)
        res_model = mapper.evaluate_tile_prediction(act_y, act_p, pred_y, pred_p, include_margin=False)
        model_recalls.append(res_model["tile_hit_ratio_recall"])
        model_precisions.append(res_model["tile_precision"])
        model_coverages.append(res_model["viewport_coverage"])
        model_active_counts.append(res_model["num_predicted_tiles"])

        # 2. Final GRU Model (+1 Guard Band Margin)
        res_margin = mapper.evaluate_tile_prediction(act_y, act_p, pred_y, pred_p, include_margin=True)
        model_margin_recalls.append(res_margin["tile_hit_ratio_recall"])
        model_margin_precisions.append(res_margin["tile_precision"])
        model_margin_coverages.append(res_margin["viewport_coverage"])
        model_margin_counts.append(res_margin["num_predicted_tiles"])

        # 3. Persistence Baseline (Evaluated with exact persistence coordinates)
        res_persist = mapper.evaluate_tile_prediction(act_y, act_p, pers_y, pers_p, include_margin=False)
        persist_recalls.append(res_persist["tile_hit_ratio_recall"])
        persist_precisions.append(res_persist["tile_precision"])
        persist_coverages.append(res_persist["viewport_coverage"])
        persist_active_counts.append(res_persist["num_predicted_tiles"])

        if (idx + 1) % 1000 == 0 or (idx + 1) == num_eval:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else float("inf")
            print(f"  Processed {idx + 1:,}/{num_eval:,} samples ({rate:.1f} samples/s)...")

    # 1. Compile Overall Summary Table
    summary_data = [
        {
            "Strategy": "Persistence Baseline",
            "Mean Tile Hit Ratio (Recall)": float(np.mean(persist_recalls) * 100.0),
            "Mean Tile Precision": float(np.mean(persist_precisions) * 100.0),
            "Mean Viewport Coverage": float(np.mean(persist_coverages) * 100.0),
            "10th Percentile Coverage (P10)": float(np.percentile(persist_coverages, 10) * 100.0),
            "90th Percentile Coverage (P90)": float(np.percentile(persist_coverages, 90) * 100.0),
            "Avg Tiles Fetched (out of 24)": float(np.mean(persist_active_counts)),
            "Tile Area Ratio (%)": float((np.mean(persist_active_counts) / 24.0) * 100.0),
        },
        {
            "Strategy": "Final GRU Model (No Guard Band)",
            "Mean Tile Hit Ratio (Recall)": float(np.mean(model_recalls) * 100.0),
            "Mean Tile Precision": float(np.mean(model_precisions) * 100.0),
            "Mean Viewport Coverage": float(np.mean(model_coverages) * 100.0),
            "10th Percentile Coverage (P10)": float(np.percentile(model_coverages, 10) * 100.0),
            "90th Percentile Coverage (P90)": float(np.percentile(model_coverages, 90) * 100.0),
            "Avg Tiles Fetched (out of 24)": float(np.mean(model_active_counts)),
            "Tile Area Ratio (%)": float((np.mean(model_active_counts) / 24.0) * 100.0),
        },
        {
            "Strategy": "Final GRU Model (+1 Guard Band Margin)",
            "Mean Tile Hit Ratio (Recall)": float(np.mean(model_margin_recalls) * 100.0),
            "Mean Tile Precision": float(np.mean(model_margin_precisions) * 100.0),
            "Mean Viewport Coverage": float(np.mean(model_margin_coverages) * 100.0),
            "10th Percentile Coverage (P10)": float(np.percentile(model_margin_coverages, 10) * 100.0),
            "90th Percentile Coverage (P90)": float(np.percentile(model_margin_coverages, 90) * 100.0),
            "Avg Tiles Fetched (out of 24)": float(np.mean(model_margin_counts)),
            "Tile Area Ratio (%)": float((np.mean(model_margin_counts) / 24.0) * 100.0),
        },
    ]

    summary_df = pd.DataFrame(summary_data)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = output_dir / "tile_prediction_metrics.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved overall tile metrics to: {summary_csv_path}")

    # 2. Compile Granular Subject-Wise Metrics Table
    unique_subs = sorted(list(set(eval_subjects)))
    subject_rows = []

    arr_persist_rec = np.array(persist_recalls)
    arr_persist_prec = np.array(persist_precisions)
    arr_persist_cov = np.array(persist_coverages)
    arr_persist_cnt = np.array(persist_active_counts)

    arr_model_rec = np.array(model_recalls)
    arr_model_prec = np.array(model_precisions)
    arr_model_cov = np.array(model_coverages)
    arr_model_cnt = np.array(model_active_counts)

    arr_margin_rec = np.array(model_margin_recalls)
    arr_margin_prec = np.array(model_margin_precisions)
    arr_margin_cov = np.array(model_margin_coverages)
    arr_margin_cnt = np.array(model_margin_counts)

    for sub in unique_subs:
        mask = (eval_subjects == sub)
        sub_n = int(np.sum(mask))

        subject_rows.append({
            "subject_id": sub,
            "Strategy": "Persistence Baseline",
            "Mean Tile Hit Ratio (Recall)": float(np.mean(arr_persist_rec[mask]) * 100.0),
            "Mean Tile Precision": float(np.mean(arr_persist_prec[mask]) * 100.0),
            "Mean Viewport Coverage": float(np.mean(arr_persist_cov[mask]) * 100.0),
            "10th Percentile Coverage (P10)": float(np.percentile(arr_persist_cov[mask], 10) * 100.0),
            "90th Percentile Coverage (P90)": float(np.percentile(arr_persist_cov[mask], 90) * 100.0),
            "Avg Tiles Fetched (out of 24)": float(np.mean(arr_persist_cnt[mask])),
            "Tile Area Ratio (%)": float((np.mean(arr_persist_cnt[mask]) / 24.0) * 100.0),
            "Sample Count": sub_n
        })
        subject_rows.append({
            "subject_id": sub,
            "Strategy": "Final GRU Model (No Guard Band)",
            "Mean Tile Hit Ratio (Recall)": float(np.mean(arr_model_rec[mask]) * 100.0),
            "Mean Tile Precision": float(np.mean(arr_model_prec[mask]) * 100.0),
            "Mean Viewport Coverage": float(np.mean(arr_model_cov[mask]) * 100.0),
            "10th Percentile Coverage (P10)": float(np.percentile(arr_model_cov[mask], 10) * 100.0),
            "90th Percentile Coverage (P90)": float(np.percentile(arr_model_cov[mask], 90) * 100.0),
            "Avg Tiles Fetched (out of 24)": float(np.mean(arr_model_cnt[mask])),
            "Tile Area Ratio (%)": float((np.mean(arr_model_cnt[mask]) / 24.0) * 100.0),
            "Sample Count": sub_n
        })
        subject_rows.append({
            "subject_id": sub,
            "Strategy": "Final GRU Model (+1 Guard Band Margin)",
            "Mean Tile Hit Ratio (Recall)": float(np.mean(arr_margin_rec[mask]) * 100.0),
            "Mean Tile Precision": float(np.mean(arr_margin_prec[mask]) * 100.0),
            "Mean Viewport Coverage": float(np.mean(arr_margin_cov[mask]) * 100.0),
            "10th Percentile Coverage (P10)": float(np.percentile(arr_margin_cov[mask], 10) * 100.0),
            "90th Percentile Coverage (P90)": float(np.percentile(arr_margin_cov[mask], 90) * 100.0),
            "Avg Tiles Fetched (out of 24)": float(np.mean(arr_margin_cnt[mask])),
            "Tile Area Ratio (%)": float((np.mean(arr_margin_cnt[mask]) / 24.0) * 100.0),
            "Sample Count": sub_n
        })

    subject_df = pd.DataFrame(subject_rows)
    subject_csv_path = output_dir / "tile_prediction_metrics_by_subject.csv"
    subject_df.to_csv(subject_csv_path, index=False)
    print(f"Saved subject-wise tile metrics to: {subject_csv_path}")

    # 3. Dynamic Technical Summary Report Generation
    pers_row = summary_df[summary_df["Strategy"] == "Persistence Baseline"].iloc[0]
    gru_row = summary_df[summary_df["Strategy"] == "Final GRU Model (No Guard Band)"].iloc[0]
    margin_row = summary_df[summary_df["Strategy"] == "Final GRU Model (+1 Guard Band Margin)"].iloc[0]

    # Dynamically format comparative conclusions based on actual calculated values
    cov_diff = gru_row['Mean Viewport Coverage'] - pers_row['Mean Viewport Coverage']
    recall_diff = gru_row['Mean Tile Hit Ratio (Recall)'] - pers_row['Mean Tile Hit Ratio (Recall)']
    comp_statement = (
        f"The GRU model achieves higher visual coverage (+{cov_diff:.2f}%) and recall (+{recall_diff:.2f}%) "
        f"compared to the Persistence Baseline at ~0.5s horizon."
        if cov_diff >= 0 and recall_diff >= 0 else
        f"Comparative result: GRU coverage delta is {cov_diff:+.2f}%, recall delta is {recall_diff:+.2f}% vs. Persistence Baseline."
    )

    report_text = f"""================================================================================
          PHASE 5: VIEWPORT TILE PREDICTION EVALUATION REPORT
================================================================================
Evaluation Dataset: {predictions_csv_path.name} (Test Set, {len(unique_subs)} unseen subjects)
Number of Evaluated Samples: {num_eval:,} (subsample stride = {sample_stride} from {total_samples:,} total test samples)
Canvas Layout: 4 rows x 6 cols = 24 spatial tiles on 3840x1920 ERP Canvas
Tile Dimensions: 640 x 480 pixels (60° yaw x 45° pitch per tile)
Viewport FoV: {hfov_deg:.1f}° Horizontal x {vfov_deg:.1f}° Vertical (Ray grid: 21x21 = 441 rays)
Prediction Horizon: k = 5 frames (~0.5 seconds lead time)

SUMMARY BENCHMARK METRICS:
--------------------------------------------------------------------------------
{summary_df.to_string(index=False)}

KEY TECHNICAL FINDINGS (DYNAMICALLY COMPUTED):
--------------------------------------------------------------------------------
1. Tile Recall & Viewport Coverage:
   - Persistence Baseline:
     * Mean Tile Recall: {pers_row['Mean Tile Hit Ratio (Recall)']:.2f}%
     * Mean Tile Precision: {pers_row['Mean Tile Precision']:.2f}%
     * Sampled Viewport Coverage: {pers_row['Mean Viewport Coverage']:.2f}% (P10: {pers_row['10th Percentile Coverage (P10)']:.2f}%, P90: {pers_row['90th Percentile Coverage (P90)']:.2f}%)
     * Avg Tiles Fetched: {pers_row['Avg Tiles Fetched (out of 24)']:.2f} ({pers_row['Tile Area Ratio (%)']:.2f}% of canvas area)
   - Final GRU Model (No Guard Band):
     * Mean Tile Recall: {gru_row['Mean Tile Hit Ratio (Recall)']:.2f}%
     * Mean Tile Precision: {gru_row['Mean Tile Precision']:.2f}%
     * Sampled Viewport Coverage: {gru_row['Mean Viewport Coverage']:.2f}% (P10: {gru_row['10th Percentile Coverage (P10)']:.2f}%, P90: {gru_row['90th Percentile Coverage (P90)']:.2f}%)
     * Avg Tiles Fetched: {gru_row['Avg Tiles Fetched (out of 24)']:.2f} ({gru_row['Tile Area Ratio (%)']:.2f}% of canvas area)
   - {comp_statement}

2. Guard Band Margin Trade-off:
   - Final GRU Model (+1 Guard Band Margin):
     * Sampled Viewport Coverage: {margin_row['Mean Viewport Coverage']:.2f}% (P10: {margin_row['10th Percentile Coverage (P10)']:.2f}%, P90: {margin_row['90th Percentile Coverage (P90)']:.2f}%)
     * Tile Recall: {margin_row['Mean Tile Hit Ratio (Recall)']:.2f}%
     * Tile Precision: {margin_row['Mean Tile Precision']:.2f}%
     * Avg Tiles Fetched: {margin_row['Avg Tiles Fetched (out of 24)']:.2f} ({margin_row['Tile Area Ratio (%)']:.2f}% of canvas area)
   - Adding a 1-ring margin increases visual coverage to {margin_row['Mean Viewport Coverage']:.2f}%,
     providing a buffer against rapid saccades and unexpected user movements.
   - Trade-off: Precision decreases from {gru_row['Mean Tile Precision']:.2f}% to {margin_row['Mean Tile Precision']:.2f}%,
     fetching an additional {margin_row['Avg Tiles Fetched (out of 24)'] - gru_row['Avg Tiles Fetched (out of 24)']:.2f} tiles on average.

3. Spatial Area Efficiency Definition:
   - Tile Area Ratio = (Average active tiles fetched / 24 total tiles) * 100%.
   - Under equal-sized tile assumptions on ERP canvas, fetching {gru_row['Avg Tiles Fetched (out of 24)']:.2f} tiles (No Guard Band)
     corresponds to {gru_row['Tile Area Ratio (%)']:.2f}% of full-frame uncompressed pixel area.
   - With +1 Guard Band, fetching {margin_row['Avg Tiles Fetched (out of 24)']:.2f} tiles corresponds to {margin_row['Tile Area Ratio (%)']:.2f}% of area.
   - Note on Metrics: Tile Area Ratio represents the fraction of 24 ERP spatial tiles fetched.
     It does NOT represent measured compressed network bitrate savings or guarantee zero streaming stalls or complete viewport coverage.

4. Subject-Wise Generalization:
   - Evaluated across {len(unique_subs)} unseen test subjects ({', '.join(str(s) for s in unique_subs)}).
   - Detailed per-subject metrics are saved to: tile_prediction_metrics_by_subject.csv
================================================================================
"""
    summary_txt_path = output_dir / "tile_prediction_summary.txt"
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n" + report_text)
    return summary_df, subject_df


if __name__ == "__main__":
    pred_path = PROJECT_ROOT / "results" / "predictions" / "final_model_test_predictions.csv"
    out_dir = PROJECT_ROOT / "results" / "tiling"
    run_tile_evaluation(pred_path, out_dir, sample_stride=5)
