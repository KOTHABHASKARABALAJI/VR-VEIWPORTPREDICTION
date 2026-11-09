"""
Script: run_day3_evaluation.py
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming
Phase: Phase 5, Day 3 — Complete System Evaluation

Master CLI driver script that executes all Day 3 experiments:
1. Viewport prediction accuracy evaluation on held-out test set (overall, per-subject, per-video).
2. Tile prediction evaluation (recall, precision, coverage, boundary cases).
3. Quality allocation strategy benchmark across bandwidth budgets (Uniform, Viewport-Aware, Neighbourhood-Aware).
4. Controlled sudden head motion simulation and utility retention analysis.
5. Robustness and sensitivity experiments (FoV size, ray sampling resolution).
6. Generates publication-ready figures in results/day3/plots/.
7. Saves all consolidated CSV tables to results/day3/.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.viewport_evaluation import (
    ViewportPredictionEvaluator,
    compute_accuracy_metrics,
)
from src.evaluation.system_evaluator import SystemEvaluator
from src.evaluation.visualizer import (
    plot_utility_vs_bandwidth,
    plot_utility_retention_vs_head_turn,
    plot_realized_utility_vs_head_turn,
    plot_per_subject_performance,
    plot_tier_distribution_comparison,
    plot_error_distribution_cdf,
)


def run_evaluation(
    predictions_csv_path: str = "results/predictions/final_model_test_predictions.csv",
    output_dir: str = "results/day3",
    sample_stride: int = 10,
) -> None:
    """
    Executes the complete Day 3 evaluation pipeline.
    """
    start_time = time.time()
    out_dir = Path(output_dir)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" DAY 3: COMPLETE SYSTEM EVALUATION PIPELINE")
    print(" Machine Learning-Based Viewport Tile Prediction & Adaptive Quality Allocation")
    print("=" * 80)

    # 1. Load test predictions DataFrame
    print(f"\n[1/6] Loading held-out test predictions from: {predictions_csv_path}")
    df_preds = pd.read_csv(predictions_csv_path)
    total_samples = len(df_preds)
    print(f"      Loaded {total_samples:,} test samples across {df_preds['subject_id'].nunique()} subjects and {df_preds['video'].nunique()} videos.")

    # 2. Viewport Prediction Evaluation
    print("\n[2/6] Evaluating Viewport Prediction Accuracy (Spherical Error, Baselines, Subjects, Videos)...")
    vp_evaluator = ViewportPredictionEvaluator(df_preds)
    overall_acc = vp_evaluator.evaluate_overall()
    df_subj = vp_evaluator.evaluate_by_subject()
    df_vid = vp_evaluator.evaluate_by_video()
    summary_table = vp_evaluator.get_summary_comparison_table()

    # Save CSVs
    summary_table.to_csv(out_dir / "viewport_prediction_overall_metrics.csv", index=False)
    df_subj.to_csv(out_dir / "viewport_prediction_by_subject.csv", index=False)
    df_vid.to_csv(out_dir / "viewport_prediction_by_video.csv", index=False)
    print("      Saved: viewport_prediction_overall_metrics.csv, by_subject.csv, by_video.csv")
    print(summary_table.to_string(index=False))

    # 3. Tile Prediction Evaluation
    print(f"\n[3/6] Evaluating Tile Prediction Performance (stride={sample_stride})...")
    sys_evaluator = SystemEvaluator()
    tile_results = sys_evaluator.evaluate_tile_predictions(df_preds, sample_stride=sample_stride)
    df_tile_summary = pd.DataFrame([res.to_dict() for res in tile_results.values()])
    df_tile_summary.to_csv(out_dir / "tile_prediction_metrics.csv", index=False)

    df_boundaries = sys_evaluator.evaluate_canonical_boundary_orientations()
    df_boundaries.to_csv(out_dir / "tile_prediction_boundary_cases.csv", index=False)
    print("      Saved: tile_prediction_metrics.csv, tile_prediction_boundary_cases.csv")
    print(df_tile_summary[["strategy", "mean_tile_recall", "mean_tile_precision", "mean_viewport_coverage", "avg_tiles_fetched"]].to_string(index=False))

    # 4. Strategy Comparison Across Bandwidth Budgets
    print(f"\n[4/6] Comparing Quality Allocation Strategies Across Bandwidth Budgets (stride={sample_stride * 2})...")
    df_strat_comparison = sys_evaluator.evaluate_allocation_strategies_on_dataset(
        df_preds, sample_stride=sample_stride * 2
    )
    df_strat_comparison.to_csv(out_dir / "allocation_strategy_comparison.csv", index=False)
    print("      Saved: allocation_strategy_comparison.csv")

    # 5. Controlled Head Motion Simulation & Empirical Retention
    print("\n[5/6] Simulating Sudden Head Rotations & Utility Retention Analysis...")
    df_motion = sys_evaluator.simulate_controlled_head_movement(budget_kbps=7200.0)
    df_motion.to_csv(out_dir / "head_movement_retention_simulation.csv", index=False)

    df_retention_bins = sys_evaluator.evaluate_empirical_retention_by_error_bin(
        df_preds, budget_kbps=7200.0, sample_stride=sample_stride * 2
    )
    df_retention_bins.to_csv(out_dir / "empirical_retention_by_error_bin.csv", index=False)

    df_sens_ray = sys_evaluator.evaluate_sampling_sensitivity()
    df_sens_ray.to_csv(out_dir / "robustness_ray_density_sensitivity.csv", index=False)

    df_sens_fov = sys_evaluator.evaluate_fov_sensitivity()
    df_sens_fov.to_csv(out_dir / "robustness_fov_sensitivity.csv", index=False)
    print("      Saved: head_movement_retention_simulation.csv, empirical_retention_by_error_bin.csv, sensitivity CSVs")

    # 6. Generate Figures & Visualizations
    print(f"\n[6/6] Generating Publication-Quality Figures in: {plots_dir}")
    plot_utility_vs_bandwidth(df_strat_comparison, plots_dir / "utility_vs_bandwidth.png")
    plot_utility_retention_vs_head_turn(df_motion, plots_dir / "utility_retention_vs_head_turn.png")
    plot_realized_utility_vs_head_turn(df_motion, plots_dir / "realized_utility_vs_head_turn.png")
    plot_per_subject_performance(df_subj, plots_dir / "per_subject_performance.png")
    plot_tier_distribution_comparison(df_strat_comparison, plots_dir / "tier_distribution_comparison.png")

    gru_sph_err = df_preds["spherical_error_deg"].values
    pers_sph_err = df_preds["persistence_sph_err_deg"].values
    plot_error_distribution_cdf(gru_sph_err, pers_sph_err, plots_dir / "error_distribution_cdf.png")
    print("      Generated 6 figures successfully.")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f" DAY 3 EVALUATION COMPLETED IN {elapsed:.1f} SECONDS")
    print(f" Results directory: {out_dir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    run_evaluation()

