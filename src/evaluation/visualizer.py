"""
Module: visualizer.py
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming
Phase: Phase 5, Day 3 — Complete System Evaluation

Generates publication-quality figures and plots for Day 3 evaluation:
1. Utility vs. Bandwidth Budget (Realized in-FoV Utility curves for all 4 allocation strategies).
2. Utility Retention vs. Head-Turn Offset (Quantifying spatial guard-band resilience under head motion).
3. Realized Utility vs. Head-Turn Offset (Absolute in-FoV utility degradation comparison).
4. Per-Subject Performance (Spherical MAE and realized utility across unseen test subjects).
5. Quality Tier Distribution (Stacked bar chart showing tile counts at Q1-Q4 across budgets).
6. Error Distribution CDF (Empirical Cumulative Distribution Function for GRU vs Baselines).
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt


def set_clean_plot_style() -> None:
    """Applies a clean, readable scientific plotting style."""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["grid.color"] = "#E0E0E0"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.7


def plot_utility_vs_bandwidth(
    df_comparison: pd.DataFrame,
    output_path: Union[str, Path]
) -> None:
    """
    Plots Realized In-FoV Weighted Utility vs. Bandwidth Budget (kbps)
    for Uniform, Viewport-Aware, and Neighbourhood-Aware (alpha=0.25, alpha=0.50).
    """
    set_clean_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    strategies = [
        ("Uniform", "#555555", "s", "--", "Uniform Allocation (Module 2)"),
        ("Viewport-Aware", "#1f77b4", "o", "-", "Viewport-Aware (Module 3)"),
        ("Neighbourhood (alpha=0.25)", "#2ca02c", "^", "-", "Neighbourhood (alpha=0.25, Module 4)"),
        ("Neighbourhood (alpha=0.50)", "#d62728", "v", "-.", "Neighbourhood (alpha=0.50, Module 4)"),
    ]

    for strat_key, color, marker, linestyle, label in strategies:
        strat_df = df_comparison[df_comparison["strategy"] == strat_key].sort_values("bandwidth_budget_kbps")
        if not strat_df.empty:
            ax.plot(
                strat_df["bandwidth_budget_kbps"] / 1000.0,
                strat_df["actual_realized_utility"],
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=2.0,
                markersize=7,
                label=label,
            )

    ax.set_title("Realized In-Viewport Utility vs. Bandwidth Budget", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Downlink Bandwidth Budget (Mbps)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Realized Viewport-Weighted Utility", fontsize=11, fontweight="bold")
    ax.set_ylim(0.8, 3.6)
    ax.set_yticks([1.0, 1.5, 2.0, 2.5, 2.8, 3.0, 3.4])
    ax.axhline(3.4, color="#AAAAAA", linestyle=":", label="Maximum Utility (Q4 Ultra = 3.40)")
    ax.grid(True)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=9.5)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_utility_retention_vs_head_turn(
    df_motion: pd.DataFrame,
    output_path: Union[str, Path]
) -> None:
    """
    Plots Utility Retention (%) vs. Head-Turn Offset Magnitude (degrees),
    highlighting the spatial guard-band resilience under rotational saccades.
    """
    set_clean_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    # Average across rotation directions for each delta_theta
    grouped = df_motion.groupby("delta_theta_deg").mean(numeric_only=True).reset_index()

    ax.plot(
        grouped["delta_theta_deg"],
        grouped["viewport_retention_pct"],
        color="#1f77b4",
        marker="o",
        linewidth=2.2,
        markersize=7,
        label="Viewport-Aware (No Guard Band)",
    )
    ax.plot(
        grouped["delta_theta_deg"],
        grouped["neighbourhood_25_retention_pct"],
        color="#2ca02c",
        marker="^",
        linewidth=2.2,
        markersize=7,
        label="Neighbourhood-Aware (alpha=0.25 Guard Band)",
    )
    ax.plot(
        grouped["delta_theta_deg"],
        grouped["neighbourhood_50_retention_pct"],
        color="#d62728",
        marker="v",
        linestyle="-.",
        linewidth=2.0,
        markersize=7,
        label="Neighbourhood-Aware (alpha=0.50 Guard Band)",
    )
    ax.plot(
        grouped["delta_theta_deg"],
        grouped["uniform_retention_pct"],
        color="#555555",
        marker="s",
        linestyle="--",
        linewidth=1.8,
        markersize=6,
        label="Uniform Baseline (Constant Low Utility)",
    )

    ax.set_title("Utility Retention vs. Head-Turn Magnitude (Simulated Saccades)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Rotational Angular Offset delta theta (Degrees)", fontsize=11, fontweight="bold")
    ax.set_ylabel("In-FoV Utility Retention (% of Predicted)", fontsize=11, fontweight="bold")
    ax.set_ylim(20, 105)
    ax.set_xlim(-1, 46)
    ax.axvline(15.0, color="#888888", linestyle=":", label="Typical Head Saccade (15°)")
    ax.grid(True)
    ax.legend(loc="lower left", frameon=True, framealpha=0.9, fontsize=9.5)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_realized_utility_vs_head_turn(
    df_motion: pd.DataFrame,
    output_path: Union[str, Path]
) -> None:
    """
    Plots Absolute Realized In-FoV Utility vs. Head-Turn Magnitude (degrees).
    """
    set_clean_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    grouped = df_motion.groupby("delta_theta_deg").mean(numeric_only=True).reset_index()

    ax.plot(
        grouped["delta_theta_deg"],
        grouped["viewport_realized_utility"],
        color="#1f77b4",
        marker="o",
        linewidth=2.2,
        markersize=7,
        label="Viewport-Aware (Module 3)",
    )
    ax.plot(
        grouped["delta_theta_deg"],
        grouped["neighbourhood_25_realized_utility"],
        color="#2ca02c",
        marker="^",
        linewidth=2.2,
        markersize=7,
        label="Neighbourhood-Aware (alpha=0.25, Module 4)",
    )
    ax.plot(
        grouped["delta_theta_deg"],
        grouped["neighbourhood_50_realized_utility"],
        color="#d62728",
        marker="v",
        linestyle="-.",
        linewidth=2.0,
        markersize=7,
        label="Neighbourhood-Aware (alpha=0.50, Module 4)",
    )
    ax.plot(
        grouped["delta_theta_deg"],
        grouped["uniform_realized_utility"],
        color="#555555",
        marker="s",
        linestyle="--",
        linewidth=1.8,
        markersize=6,
        label="Uniform Baseline (Q2 = 2.00)",
    )

    ax.set_title("Absolute Realized In-FoV Utility vs. Head Rotation Angle (at 7.2 Mbps)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Rotational Angular Offset delta theta (Degrees)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Absolute Realized In-FoV Utility", fontsize=11, fontweight="bold")
    ax.set_ylim(0.8, 3.6)
    ax.grid(True)
    ax.legend(loc="lower left", frameon=True, framealpha=0.9, fontsize=9.5)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_per_subject_performance(
    df_subject: pd.DataFrame,
    output_path: Union[str, Path]
) -> None:
    """
    Plots Per-Subject Spherical Error and MAE reduction.
    """
    set_clean_plot_style()
    fig, ax1 = plt.subplots(figsize=(10, 5.5), dpi=300)

    subjects = [f"Sub {s}" for s in df_subject["subject_id"]]
    x = np.arange(len(subjects))
    width = 0.35

    rects1 = ax1.bar(x - width/2, df_subject["gru_spherical_mae_deg"], width, label="GRU-64 Model", color="#1f77b4")
    rects2 = ax1.bar(x + width/2, df_subject["persistence_spherical_mae_deg"], width, label="Persistence Baseline", color="#aec7e8")

    ax1.set_ylabel("Spherical MAE (Degrees)", fontsize=11, fontweight="bold")
    ax1.set_title("Spherical Prediction Accuracy Across 8 Unseen Test Subjects", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(subjects, fontsize=10)
    ax1.set_ylim(0, max(df_subject["persistence_spherical_mae_deg"]) * 1.25)
    ax1.grid(True, axis="y")
    ax1.legend(loc="upper right", frameon=True, fontsize=10)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_tier_distribution_comparison(
    df_comparison: pd.DataFrame,
    output_path: Union[str, Path]
) -> None:
    """
    Plots stacked bar chart of Quality Tier distribution (Q1, Q2, Q3, Q4)
    across bandwidth budgets for Viewport-Aware and Neighbourhood-Aware (alpha=0.25).
    """
    set_clean_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300, sharey=True)

    budgets = df_comparison["bandwidth_budget_kbps"].unique()
    budget_labels = [f"{b/1000:.1f} Mbps" for b in sorted(budgets)]

    for ax, strat_name, title in [
        (ax1, "Viewport-Aware", "Viewport-Aware Allocation (Module 3)"),
        (ax2, "Neighbourhood (alpha=0.25)", "Neighbourhood-Aware (alpha=0.25, Module 4)"),
    ]:
        sub = df_comparison[df_comparison["strategy"] == strat_name].sort_values("bandwidth_budget_kbps")
        q1 = sub["avg_q1_tiles"].values
        q2 = sub["avg_q2_tiles"].values
        q3 = sub["avg_q3_tiles"].values
        q4 = sub["avg_q4_tiles"].values

        x = np.arange(len(budget_labels))
        width = 0.55

        ax.bar(x, q1, width, label="Q1 (Low, 100 kbps)", color="#7f7f7f")
        ax.bar(x, q2, width, bottom=q1, label="Q2 (Med, 300 kbps)", color="#ffbb78")
        ax.bar(x, q3, width, bottom=q1 + q2, label="Q3 (High, 600 kbps)", color="#1f77b4")
        ax.bar(x, q4, width, bottom=q1 + q2 + q3, label="Q4 (Ultra, 1000 kbps)", color="#2ca02c")

        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(budget_labels, fontsize=9.5)
        ax.set_ylim(0, 24)
        ax.set_ylabel("Tile Count (Total = 24)", fontsize=10, fontweight="bold")
        ax.grid(True, axis="y")
        ax.legend(loc="upper left", frameon=True, fontsize=8.5)

    fig.suptitle("Canvas Quality Tier Distribution Across Bandwidth Budgets", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_error_distribution_cdf(
    gru_errors: np.ndarray,
    pers_errors: np.ndarray,
    output_path: Union[str, Path]
) -> None:
    """
    Plots Cumulative Distribution Function (CDF) of spherical angular error.
    """
    set_clean_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    sorted_gru = np.sort(gru_errors)
    cdf_gru = np.arange(1, len(sorted_gru) + 1) / len(sorted_gru) * 100.0

    sorted_pers = np.sort(pers_errors)
    cdf_pers = np.arange(1, len(sorted_pers) + 1) / len(sorted_pers) * 100.0

    ax.plot(sorted_gru, cdf_gru, color="#1f77b4", linewidth=2.2, label=f"Final GRU Model (MAE={np.mean(gru_errors):.2f}°)")
    ax.plot(sorted_pers, cdf_pers, color="#aec7e8", linewidth=2.0, linestyle="--", label=f"Persistence Baseline (MAE={np.mean(pers_errors):.2f}°)")

    ax.set_title("Empirical CDF of Spherical Angular Error on Test Set", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Spherical Error (Degrees)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cumulative Percentage of Frames (%)", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 102)
    ax.axvline(10.0, color="#888888", linestyle=":", label="10° Error Boundary")
    ax.grid(True)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=10)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)

