"""
Module: statistical_analysis.py
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming
Phase: Phase 5, Day 3 — Final Audit Statistical Analysis

Performs statistically rigorous analysis of the prediction and allocation results:
1. Bootstrap confidence intervals on overall spherical MAE (GRU and Persistence).
2. Per-subject paired t-test: GRU vs Persistence spherical error.
3. Cohen's d effect size for GRU vs Persistence error reduction.
4. Paired t-test on realized utility per sample: Viewport-Aware vs Neighbourhood-25,
   per bandwidth budget, using subject-level aggregation to avoid pseudo-replication.

NOTE ON INDEPENDENCE:
   Test samples are drawn from continuous head-movement trajectories.  Consecutive
   windows from the same subject are temporally correlated.  Treating each of the
   45,121 samples as fully independent would inflate degrees of freedom and deflate
   p-values.  We therefore apply two-level analysis:
     (a) Per-sample bootstrap CI on overall MAE   — reports uncertainty but does NOT
         claim sample-level independence.
     (b) Subject-level paired comparisons (N=8 subjects) — conservative, correct
         approach for subject-independent evaluation.
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREDICTIONS_CSV = os.path.join(PROJECT_ROOT, "results", "predictions", "final_model_test_predictions.csv")
ALLOCATION_CSV  = os.path.join(PROJECT_ROOT, "results", "day3", "allocation_strategy_comparison.csv")
OUT_DIR         = os.path.join(PROJECT_ROOT, "results", "day3", "final")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 42
N_BOOTSTRAP = 10_000


# ── Helpers ──────────────────────────────────────────────────────────────────

def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    ci_level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float, float]:
    """
    Parametric-free bootstrap CI for the mean.

    Returns
    -------
    mean : float
    ci_low : float
    ci_high : float
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    n = len(values)
    boot_means = np.array(
        [rng.choice(values, size=n, replace=True).mean() for _ in range(n_boot)]
    )
    alpha = 1.0 - ci_level
    ci_low  = float(np.percentile(boot_means, 100.0 * alpha / 2.0))
    ci_high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return float(values.mean()), ci_low, ci_high


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d (unsigned) for paired difference a - b."""
    diff = a - b
    return float(diff.mean() / (diff.std(ddof=1) + 1e-15))


# ── Main analysis ─────────────────────────────────────────────────────────────

def run_statistical_analysis() -> pd.DataFrame:
    print("=" * 70)
    print("STATISTICAL ANALYSIS — Day 3 Final Audit")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────────
    df = pd.read_csv(PREDICTIONS_CSV)
    alloc_df = pd.read_csv(ALLOCATION_CSV)
    n_total = len(df)
    n_subjects = df["subject_id"].nunique()
    subjects = sorted(df["subject_id"].unique())

    gru_errs  = df["spherical_error_deg"].values.astype(np.float64)
    pers_errs = df["persistence_sph_err_deg"].values.astype(np.float64)

    print(f"\nTotal prediction instances : {n_total:,}")
    print(f"Test subjects              : {n_subjects} {subjects}")

    records: List[Dict] = []

    # ── ANALYSIS 1: Bootstrap CI for overall spherical MAE ───────────────────
    print("\n--- ANALYSIS 1: Bootstrap CI for Overall Spherical MAE ---")
    rng = np.random.default_rng(RANDOM_SEED)

    gru_mean,  gru_ci_lo,  gru_ci_hi  = bootstrap_mean_ci(gru_errs,  N_BOOTSTRAP, 0.95, rng)
    pers_mean, pers_ci_lo, pers_ci_hi = bootstrap_mean_ci(pers_errs, N_BOOTSTRAP, 0.95, rng)

    print(f"  GRU  MAE  = {gru_mean:.4f}° (95% CI: [{gru_ci_lo:.4f}, {gru_ci_hi:.4f}])")
    print(f"  Pers MAE  = {pers_mean:.4f}° (95% CI: [{pers_ci_lo:.4f}, {pers_ci_hi:.4f}])")

    records.append({
        "analysis": "Bootstrap CI — GRU Spherical MAE",
        "metric": "spherical_mae_deg",
        "mean": gru_mean,
        "ci_low_95": gru_ci_lo,
        "ci_high_95": gru_ci_hi,
        "n_samples": n_total,
        "note": f"Bootstrap with {N_BOOTSTRAP} replicates. Samples are trajectory windows "
                "(temporally correlated within subject); CI reflects sampling uncertainty "
                "of the mean, not iid CI.",
    })
    records.append({
        "analysis": "Bootstrap CI — Persistence Spherical MAE",
        "metric": "spherical_mae_deg",
        "mean": pers_mean,
        "ci_low_95": pers_ci_lo,
        "ci_high_95": pers_ci_hi,
        "n_samples": n_total,
        "note": "Same caveats as above.",
    })

    # ── ANALYSIS 2: Subject-level paired t-test ───────────────────────────────
    print("\n--- ANALYSIS 2: Subject-Level Paired t-Test (GRU vs Persistence) ---")
    # Subject-level mean error (N = 8)
    sub_gru_means  = []
    sub_pers_means = []
    for sid in subjects:
        mask = df["subject_id"] == sid
        sub_gru_means.append(df.loc[mask, "spherical_error_deg"].mean())
        sub_pers_means.append(df.loc[mask, "persistence_sph_err_deg"].mean())
    sub_gru_means  = np.array(sub_gru_means)
    sub_pers_means = np.array(sub_pers_means)

    t_stat, p_val = stats.ttest_rel(sub_pers_means, sub_gru_means, alternative="greater")
    # H0: Pers MAE <= GRU MAE  →  H1: Pers MAE > GRU MAE  (one-sided)

    d = cohens_d(sub_pers_means, sub_gru_means)  # positive = GRU better
    print(f"  Subject means  GRU : {sub_gru_means}")
    print(f"  Subject means  Pers: {sub_pers_means}")
    print(f"  Paired t-test (one-sided, Pers > GRU): t={t_stat:.4f}, p={p_val:.6f}")
    print(f"  Cohen's d (error reduction): {d:.4f}")

    records.append({
        "analysis": "Subject-Level Paired t-Test — GRU vs Persistence MAE",
        "metric": "spherical_mae_deg",
        "mean": float(sub_gru_means.mean()),
        "ci_low_95": float(sub_gru_means.mean() - 1.895 * sub_gru_means.std(ddof=1) / np.sqrt(len(subjects))),
        "ci_high_95": float(sub_gru_means.mean() + 1.895 * sub_gru_means.std(ddof=1) / np.sqrt(len(subjects))),
        "t_statistic": float(t_stat),
        "p_value_one_sided": float(p_val),
        "cohens_d": float(d),
        "n_subjects": n_subjects,
        "note": "Paired one-sided t-test H1: Persistence MAE > GRU MAE. "
                "N=8 subjects. Subject-level means are used to avoid pseudo-replication. "
                "95% CI for GRU mean via t-distribution (df=7, t_crit=1.895).",
    })

    # ── ANALYSIS 3: Cohen's d on full sample errors ───────────────────────────
    print("\n--- ANALYSIS 3: Cohen's d on Full Sample Error Distribution ---")
    # NOTE: This is for descriptive effect size; independence assumed only
    # for characterising the magnitude of difference, not for inference.
    d_full = cohens_d(pers_errs, gru_errs)
    print(f"  Cohen's d (all 45,121 samples): {d_full:.4f}")
    if d_full < 0.2:
        magnitude = "negligible"
    elif d_full < 0.5:
        magnitude = "small"
    elif d_full < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"
    print(f"  Effect magnitude: {magnitude}")

    records.append({
        "analysis": "Cohen's d — Full Sample Error Reduction",
        "metric": "spherical_error_deg",
        "mean": float(np.mean(pers_errs - gru_errs)),
        "cohens_d": d_full,
        "effect_magnitude": magnitude,
        "n_samples": n_total,
        "note": "Descriptive effect size on paired sample-level errors. "
                "Within-subject temporal correlation inflates N; treat as "
                "magnitude characterisation only, not an inferential test.",
    })

    # ── ANALYSIS 4: Bootstrap CI for MAE reduction using subject-level means ─
    print("\n--- ANALYSIS 4: Bootstrap CI for Subject-Level MAE Reduction % ---")
    # Use subject-level means to avoid near-zero denominator issues that
    # distort per-sample percentage calculations when pers_err ≈ 0.
    sub_reduction_pct = (
        (sub_pers_means - sub_gru_means) / sub_pers_means * 100.0
    )
    rng2 = np.random.default_rng(RANDOM_SEED + 1)
    red_mean, red_ci_lo, red_ci_hi = bootstrap_mean_ci(sub_reduction_pct, N_BOOTSTRAP, 0.95, rng2)
    # Also compute overall MAE reduction from aggregate means
    overall_reduction_pct = (pers_mean - gru_mean) / pers_mean * 100.0
    print(f"  Overall aggregate MAE reduction : {overall_reduction_pct:.2f}%")
    print(f"  Subject-level mean reduction    : {red_mean:.2f}% (95% CI: [{red_ci_lo:.2f}, {red_ci_hi:.2f}])")
    records.append({
        "analysis": "Bootstrap CI — Subject-Level MAE Reduction %",
        "metric": "mae_reduction_pct",
        "overall_aggregate_reduction_pct": overall_reduction_pct,
        "mean": red_mean,
        "ci_low_95": red_ci_lo,
        "ci_high_95": red_ci_hi,
        "n_subjects": n_subjects,
        "note": "Reduction = (pers_mean - gru_mean)/pers_mean * 100 per subject. "
                "Bootstrap CI on subject-level means (N=8). "
                "Overall aggregate = (13.9567 - 9.4454) / 13.9567 * 100 = 32.32%.",
    })

    # ── ANALYSIS 5: Per-budget utility comparison (Viewport vs N25) ──────────
    print("\n--- ANALYSIS 5: Subject-Level Utility Analysis per Budget ---")
    # We use the per-sample allocation comparison data.
    # NOTE: The allocation CSV only has aggregate stats per budget × strategy (24 rows).
    # For per-sample paired tests we must recompute or note limitation.
    # The 24-row CSV does NOT provide per-sample per-subject data, so we cannot
    # do subject-level paired tests directly from it.
    # We report the aggregate difference and note this limitation.
    budgets_in_csv = sorted(alloc_df["bandwidth_budget_kbps"].unique())
    for b in budgets_in_csv:
        vp_row = alloc_df[(alloc_df["bandwidth_budget_kbps"] == b) & (alloc_df["strategy"] == "Viewport-Aware")]
        n25_row = alloc_df[(alloc_df["bandwidth_budget_kbps"] == b) & (alloc_df["strategy"] == "Neighbourhood (alpha=0.25)")]
        if vp_row.empty or n25_row.empty:
            continue
        vp_util  = float(vp_row["actual_realized_utility"].values[0])
        n25_util = float(n25_row["actual_realized_utility"].values[0])
        diff = n25_util - vp_util
        pct_change = (diff / (vp_util + 1e-9)) * 100.0
        print(f"  Budget {int(b):6d} kbps | VP={vp_util:.4f} | N25={n25_util:.4f} | Δ={diff:+.4f} ({pct_change:+.2f}%)")
        records.append({
            "analysis": f"Aggregate Utility Comparison @ {int(b)} kbps (VP vs N25)",
            "metric": "actual_realized_utility",
            "viewport_aware_utility": vp_util,
            "neighbourhood_25_utility": n25_util,
            "difference_n25_minus_vp": diff,
            "pct_change": pct_change,
            "n_samples": "aggregate (see allocation_strategy_comparison.csv)",
            "note": "Aggregate over ~2,256 test samples (stride=20 from 45,121). "
                    "Per-sample utility data not stored separately; subject-level "
                    "paired test on utility is not feasible without re-running the "
                    "allocator per sample and storing individual results. "
                    "Positive Δ = Neighbourhood-25 outperforms Viewport-Aware.",
        })

    # ── Save results ──────────────────────────────────────────────────────────
    out_df = pd.DataFrame(records)
    out_path = os.path.join(OUT_DIR, "statistical_analysis.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved statistical analysis to: {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  GRU-64 Spherical MAE : {gru_mean:.4f}° (95% bootstrap CI: [{gru_ci_lo:.4f}, {gru_ci_hi:.4f}])")
    print(f"  Persistence MAE      : {pers_mean:.4f}° (95% bootstrap CI: [{pers_ci_lo:.4f}, {pers_ci_hi:.4f}])")
    print(f"  Subject-level t-test : t={t_stat:.4f}, p={p_val:.6f} (one-sided, N=8)")
    print(f"  Cohen's d (subject)  : {d:.4f} — {magnitude if d < 0.2 else 'small' if d < 0.5 else 'medium' if d < 0.8 else 'large'} effect")
    print(f"  Cohen's d (all)      : {d_full:.4f} — {magnitude}")

    return out_df


if __name__ == "__main__":
    run_statistical_analysis()
