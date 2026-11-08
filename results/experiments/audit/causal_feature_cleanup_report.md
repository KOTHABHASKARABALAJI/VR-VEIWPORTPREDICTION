# Causal Feature Engineering Cleanup Report

**Project Title:** Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming  
**Cleanup Date:** November 4, 2026  
**Scope:** Elimination of lookahead data leakage in feature engineering, mathematical enforcement of strictly causal backward differences, boundary and irregular timestamp handling, repository-wide gradient audit, and causality unit test verification.  
**Report Location:** [`results/experiments/audit/causal_feature_cleanup_report.md`](file:///d:/ml%20final%20project/results/experiments/audit/causal_feature_cleanup_report.md)  
**Test Suite:** [`tests/test_causal_features.py`](file:///d:/ml%20final%20project/tests/test_causal_features.py)  

---

## 1. Executive Summary

During the Phase 3 audit, an implementation bug was identified in [`src/feature_engineering/features.py`](file:///d:/ml%20final%20project/src/feature_engineering/features.py), where `np.gradient()` was used across full video trajectories prior to sequence slicing. This introduced future-information lookahead leakage into velocity ($t+1$) and acceleration ($t+2$) features at the final observed window frames.

To guarantee complete scientific integrity prior to Phase 4 (Viewport-to-Tile Mapping and Adaptive Quality Allocation), a comprehensive **causal feature engineering cleanup** was performed across the codebase:
1. **Strictly Causal Backward Differences:** Rewrote `compute_derivatives()` and `extract_7_features()` in `features.py` to use exclusively backward differences.
2. **Circular Yaw Wrap-around Protection:** Implemented shortest signed circular difference across the $\pm 180^\circ$ boundary to eliminate angular wrap discontinuities without introducing derivative spikes.
3. **Boundary & Timestamp Robustness:** Engineered safe fallbacks for edge cases: empty trajectories ($T=0$), single-frame initial states ($T=1$), irregular sampling intervals, duplicate timestamps ($\Delta t = 0$), and negative timestamp jitter.
4. **Repository-Wide Eradication of `np.gradient`:** Scanned the entire repository and eliminated all secondary usages of centered differences in viewport feature calculation (specifically in [`src/evaluation/detailed_error_analysis.py`](file:///d:/ml%20final%20project/src/evaluation/detailed_error_analysis.py)).
5. **Automated Causality Unit Test Suite:** Developed [`tests/test_causal_features.py`](file:///d:/ml%20final%20project/tests/test_causal_features.py) comprising 7 unit tests that formally prove that mutating future trajectory frames has zero effect on features calculated for earlier timestamps. All 7 tests pass with 100% success.
6. **Strict Non-Interference:** Preserved all existing model checkpoints (`saved_models/`, `results/experiments/models/`) and historical metric logs. No Phase 4 logic was initiated, and no new models were trained.

---

## 2. Mathematical Formulation of the Causal Pipeline

### 2.1 Velocity: Causal First-Order Backward Difference
For a discrete sequence of kinematic observations $x_0, x_1, \dots, x_{T-1}$ sampled at timestamps $t_0, t_1, \dots, t_{T-1}$:
- At initial frame $t = 0$: No prior observation exists. Velocity is strictly defined as initial stationary state:
  $$v_0 = 0.0$$
- For subsequent frames $t \ge 1$: Velocity is computed using exclusively the current observation $x_t$ and the immediately preceding observation $x_{t-1}$:
  $$v_t = \frac{\Delta x_t}{\Delta t_{t-1}}, \quad \text{where } \Delta t_{t-1} = t_t - t_{t-1}$$

### 2.2 Shortest Signed Circular Yaw Difference
Direct subtraction of circular angles produces artificial spikes at the rear meridian ($+180^\circ / -180^\circ$). If a user turns clockwise from $+179^\circ$ to $-179^\circ$, naive subtraction yields $-179^\circ - 179^\circ = -358^\circ$, generating a spurious $-3580^\circ/\text{s}$ velocity artifact.

To resolve this causally without multi-pass unwrapping, the shortest signed circular arc difference is computed:
$$\Delta \text{yaw}_t = \left( (\text{yaw}_t - \text{yaw}_{t-1} + 180^\circ) \bmod 360^\circ \right) - 180^\circ$$
- For $+179^\circ \to -179^\circ$: $\Delta \text{yaw}_t = (+2^\circ)$ (Clockwise)
- For $-179^\circ \to +179^\circ$: $\Delta \text{yaw}_t = (-2^\circ)$ (Counter-clockwise)
- Bounded strictly within $[-180^\circ, +180^\circ]$.

Because pitch is physically constrained to $[-90^\circ, +90^\circ]$ and does not wrap around:
$$\Delta \text{pitch}_t = \text{pitch}_t - \text{pitch}_{t-1}$$

### 2.3 Acceleration: Causal Second-Order Backward Difference
Angular velocity is a continuous Euclidean quantity in $\text{deg/s}$ (not an angle on a circle):
- At initial frame $t = 0$:
  $$a_0 = 0.0$$
- For subsequent frames $t \ge 1$:
  $$a_t = \frac{v_t - v_{t-1}}{\Delta t_{t-1}}$$

### 2.4 Mathematical Proof of Causality and Prefix Invariance
Let $\mathbf{F}_{0:t}$ denote the $(t+1) \times 7$ feature matrix computed from observations $x_{0:t}$.
Because each row $\mathbf{f}_t$ is defined as:
$$\mathbf{f}_t = g(x_t, x_{t-1}, t_t, t_{t-1}, v_{t-1})$$
It follows that:
$$\frac{\partial \mathbf{f}_t}{\partial x_{t+k}} = 0 \quad \forall k \ge 1$$
Therefore, mutating any future value $x_{t+k}$ has an identically zero partial derivative and zero numerical impact on all historical features $\mathbf{f}_{0:t}$.

---

## 3. Code Modifications & Implementation Details

### 3.1 [`src/feature_engineering/features.py`](file:///d:/ml%20final%20project/src/feature_engineering/features.py)
The following enhancements were implemented:
1. **`compute_circular_difference(angles_degrees)`**:
   Implements pointwise shortest circular difference on $[-180^\circ, +180^\circ]$.
2. **`compute_derivatives(values, dt, is_angular)`**:
   - Computes causal backward difference for velocity and acceleration.
   - Initial boundaries ($t=0$) evaluate cleanly to $0.0$.
   - Fully supports scalar `dt` (float) or 1D array of `timestamps` / sampling intervals.
   - Protects against division by zero: non-positive, zero, NaN, or Inf intervals fall back to default sampling interval ($0.10091\text{ s}$).
3. **`extract_7_features(yaw_degrees, pitch_degrees, dt, timestamps)`**:
   - Casts inputs to `float64` to prevent integer-division truncation issues during differentiation.
   - Handles boundary trajectory lengths:
     - $T=0$: Returns empty array of shape `(0, 7)`.
     - $T=1$: Returns `(1, 7)` array with valid trigonometric coordinates and zero velocity/acceleration.
     - $T \ge 2$: Full causal feature extraction.
   - Outputs `float32` matrix of shape `(T, 7)` preserving exact feature order:
     - Feature 0: `sin_yaw`
     - Feature 1: `cos_yaw`
     - Feature 2: `pitch`
     - Feature 3: `yaw_vel`
     - Feature 4: `pitch_vel`
     - Feature 5: `yaw_acc`
     - Feature 6: `pitch_acc`

### 3.2 [`src/feature_engineering/__init__.py`](file:///d:/ml%20final%20project/src/feature_engineering/__init__.py)
- Exported `compute_circular_difference` in `__all__` alongside `extract_7_features`, `compute_derivatives`, `compute_sin_cos_yaw`, `unwrap_yaw_degrees`, and `standardize_features`.

### 3.3 [`src/evaluation/detailed_error_analysis.py`](file:///d:/ml%20final%20project/src/evaluation/detailed_error_analysis.py)
- Replaced trajectory-level `np.gradient(y_unwrapped, dt_step)` and `np.gradient(p, dt_step)` in Analysis B (lines 232–237) with strictly causal backward differences.
- Ensures all multi-horizon baseline evaluations use strictly causal kinematics.

---

## 4. Repository-Wide Audit for `np.gradient` and Centered Differences

A search across all Python files in the workspace confirmed:
```powershell
Get-ChildItem -Path . -Recurse -Filter "*.py" | Select-String -Pattern "np\.gradient"
# Result: 0 matches found.
```
```powershell
Get-ChildItem -Path . -Recurse -Filter "*.py" | Select-String -Pattern "\.gradient"
# Result: 0 matches found.
```

**Audit Status:** `np.gradient` and centered difference operators have been **100% eradicated** from all feature engineering, preprocessing, and evaluation pipelines across the repository.

---

## 5. Automated Causality & Regression Test Suite

A comprehensive test suite was established in [`tests/test_causal_features.py`](file:///d:/ml%20final%20project/tests/test_causal_features.py).

### 5.1 Test Results Summary
Command: `py -3.14 -m unittest tests/test_causal_features.py`
```
.......
----------------------------------------------------------------------
Ran 7 tests in 0.092s

OK
```

### 5.2 Test Specifications & Verification Matrix

| Test Method | Objective | Verification Condition | Result |
| :--- | :--- | :--- | :---: |
| `test_strict_causality_under_future_mutation` | Verifies that injecting radical future perturbations (noise, spikes $\pm 10,000^\circ$) at $t > k$ does not alter earlier features. | `np.testing.assert_allclose(F_clean[:k+1], F_mutated[:k+1], atol=1e-12)` across cutoffs $k \in \{1, 5, 10, 25, 50, 75, 90\}$. | **PASS** |
| `test_prefix_invariance` | Verifies that extracting features on prefix `[:k]` exactly matches slice `[:k]` of full trajectory features. | `extract_7_features(y[:k], p[:k]) == extract_7_features(y, p)[:k]` for $k \in \{2, 10, 20, 50\}$. | **PASS** |
| `test_circular_yaw_wraparound` | Verifies circular shortest distance across $+180^\circ$ and $-180^\circ$ boundaries. | $+179^\circ \to -179^\circ$ at $\Delta t = 0.1\text{ s}$ yields $+20.0^\circ/\text{s}$, not $-3580^\circ/\text{s}$. | **PASS** |
| `test_boundary_conditions_and_shapes` | Verifies boundary handling on empty ($T=0$), single-frame ($T=1$), and standard trajectories. | Output shapes $(0, 7)$, $(1, 7)$, $(T, 7)$ with `dtype=float32` and zero initial derivatives. | **PASS** |
| `test_irregular_timestamps_and_zero_interval` | Tests robustness against zero-interval duplicates ($\Delta t = 0$) and negative jitter. | Features remain strictly finite with zero NaNs and zero Infs. | **PASS** |
| `test_feature_order_and_encoding_integrity` | Validates trigonometric encoding ($\sin^2 + \cos^2 = 1.0$) and exact column alignment. | Feature columns match $[ \sin, \cos, \text{pitch}, v_y, v_p, a_y, a_p ]$. | **PASS** |
| `test_streaming_inference_equivalence` | Simulates online real-time streaming client receiving 1 frame per step. | Streaming window `feats[:t+1][-10:]` identically matches offline batch slice `feats[t-9:t+1]`. | **PASS** |

---

## 6. Verification of Online Model Inference

To ensure that the causal feature pipeline integrates seamlessly with trained neural models, an online streaming inference simulation was conducted:
1. Trajectory data from test subject 9 (`2.mp4`) was streamed frame-by-frame.
2. At frame $t = 20$, features were computed using strictly historical observations:
   ```python
   feats_history = extract_7_features(yaw[: t + 1], pitch[: t + 1], timestamps=ts[: t + 1])
   input_window = feats_history[-10:]  # Last 10 causal frames
   input_norm = (input_window - norm_mean) / norm_std
   ```
3. Passed through winning model `ModelC_GRU64` (`saved_models/best_viewport_model.pth`).
4. Output:
   - Last observed orientation at $t=20$: $\text{yaw} = -13.00^\circ$, $\text{pitch} = 2.00^\circ$
   - Predicted orientation at $t+5$ ($0.5\text{ s}$ ahead): $\widehat{\text{yaw}} = -40.03^\circ$, $\widehat{\text{pitch}} = 2.07^\circ$
   - Actual ground truth at $t+5$: $\text{yaw} = -20.00^\circ$, $\text{pitch} = 1.00^\circ$

**Inference Verification:** The recurrent model executes without errors, consuming strictly historical window inputs and producing valid continuous viewport predictions.

---

## 7. Artifact Integrity & Readiness for Phase 4

- **No Overwriting:** Historical models in `saved_models/` and `results/experiments/models/` remain untouched.
- **Git Status:** All changes remain in the local working directory (no commits or pushes executed).
- **Clean State:** The feature engineering foundation is verified as leak-free and mathematically sound.
- **Next Phase:** The project is prepared to proceed to Phase 4 (Viewport-to-Tile Mapping and Adaptive Quality Allocation) upon user authorization.

