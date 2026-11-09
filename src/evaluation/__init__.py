"""
Package: src.evaluation
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming
Phase: Phase 5, Day 3 — Complete System Evaluation

Exposes comprehensive evaluation engines:
- Viewport prediction accuracy and spherical error benchmarking (ViewportPredictionEvaluator).
- End-to-end tile prediction and adaptive quality allocation system evaluation (SystemEvaluator).
- Publication-quality visualization routines (visualizer).
"""

from .viewport_evaluation import (
    ViewportAccuracyMetrics,
    ViewportPredictionEvaluator,
    compute_spherical_error_vector,
    compute_accuracy_metrics,
)
from .system_evaluator import (
    SystemEvaluator,
    TilePredictionBenchmarkResult,
    CANONICAL_BUDGETS_KBPS,
)

__all__ = [
    "ViewportAccuracyMetrics",
    "ViewportPredictionEvaluator",
    "compute_spherical_error_vector",
    "compute_accuracy_metrics",
    "SystemEvaluator",
    "TilePredictionBenchmarkResult",
    "CANONICAL_BUDGETS_KBPS",
]
