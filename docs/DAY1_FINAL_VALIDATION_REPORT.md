# Day 1 Final Technical Validation Report

**Project:** Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming  
**Stage:** Phase 5, Day 1 — Technical Audit, Bug Fix, and Systemic Validation  
**Date:** November 2026  
**Status:** FULLY VERIFIED & DETERMINISTICALLY REPRODUCIBLE  

---

## 1. Executive Summary

This report establishes the verified technical state of the 360° viewport prediction and tile mapping pipeline following the complete resolution of all data pipeline, feature extraction, streaming inference, model training, and spatial tile evaluation bugs diagnosed during the Day 1 audit.

The entire pipeline was audited against fundamental physical, kinematic, mathematical, and machine learning principles:
1. **Zero Information Leakage:** Strict subject-independent partitioning (33 train, 7 validation, 8 test subjects) with zero subject or temporal overlap. All Z-score normalization parameters are fitted exclusively on the training split.
2. **Strict Causality:** Mathematical proof of causal feature construction where future data corruption has zero impact on past or current features. Fixed acceleration boundary policy ($a_0 = 0.0, a_1 = 0.0$) eliminates spurious initial spikes.
3. **Auditable Sensor Telemetry:** Logged deduplication of 34 initial calibration duplicates at $t = 0.0\text{s}$, verifying strictly positive sampling intervals ($\min dt = 0.002\text{s}$, nominal $9.91\text{ Hz}$).
4. **Optimal Model Architecture:** A lightweight single-layer GRU (64 hidden units, 16,195 trainable parameters) achieves **9.45° Spherical MAE** on unseen test subjects, outperforming the Persistence baseline (13.96°) by **+32.32%** and the Constant Angular Velocity baseline (11.56°) by **+18.27%**.
5. **Real-Time Streaming Latency:** Benchmarked mean inference latency of **271.67 µs (0.27 ms)** per sample, which is two orders of magnitude below the 100 ms video frame budget.
6. **Tile Mapping Parity & Trade-offs:** Evaluated on a $4\times 6$ (24 tile) equirectangular projection grid with $100^\circ \times 100^\circ$ FoV. The GRU model achieves **97.99% mean viewport coverage** (P10: 94.33%) fetching 8.87 tiles (36.98% area ratio). With a $+1$ guard band margin, coverage reaches **99.98%** (P10: 100.0%) fetching 19.47 tiles (81.11% area ratio).

---

## 2. Dataset Architecture & Split Integrity

### 2.1 Raw Telemetry Audit & Deduplication
The project utilizes viewport head trajectories from the AVTrack360 benchmark dataset (`data/processed/viewport_trajectories.csv`). An exhaustive sweep over all 284,247 rows across 960 trajectories identified 34 records where consecutive timestamps were identical ($dt = 0.0\text{s}$). All 34 cases occurred at the exact start ($t = 0.0\text{s}$) during VR headset zero-calibration across 29 trajectories.

- **Preprocessing Policy:** The initial duplicate observation was deduplicated to retain the settled orientation at $t = 0.0\text{s}$.
- **Post-Cleaning Verification:** The cleaned dataset contains 284,213 rows. All consecutive intervals satisfy $\Delta t > 0$ strictly:
  - $\Delta t_{\min} = 0.0020\text{ s}$
  - $\Delta t_{\max} = 0.4330\text{ s}$
  - $\Delta t_{\text{mean}} = 0.10091\text{ s}$
  - $\Delta t_{\text{median}} = 0.10100\text{ s}$
  - Nominal Sampling Frequency: $9.91\text{ Hz}$ (~10 Hz)
- All 34 deduplications are logged with trajectory ID, subject, video, and coordinates in `data/processed/dataset_audit_metadata.json`.

### 2.2 Subject-Independent Partitioning
To guarantee valid generalization to novel viewers, data partitioning is enforced strictly by subject ID:
- **Total Unique Subjects:** 48
- **Training Set (33 subjects, 68.75%):**  
  `[2, 3, 5, 6, 7, 8, 10, 11, 13, 14, 15, 17, 18, 19, 21, 23, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 39, 41, 42, 45, 46, 47, 48]`
- **Validation Set (7 subjects, 14.58%):**  
  `[4, 12, 24, 25, 37, 38, 43]`
- **Test Set (8 subjects, 16.67%):**  
  `[9, 16, 20, 22, 30, 40, 44, 49]`
- **Leakage Audit:**
  $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
  Subject leakage is strictly **0.0%**.

### 2.3 Supervised Sliding Window Generation
- **Input History Length ($L$):** 10 steps (nominal 1.009 seconds).
- **Prediction Horizon ($k$):** 5 steps ahead (nominal 0.505 seconds lead time).
- **Target Indexing Formulation:**
  For window starting at index $i$:
  - Historical observations: indices $[i, i + 9]$ (inclusive).
  - Last observed input timestamp: $t_{\text{last}} = \text{timestamps}[i + 9]$.
  - Target future index: $t_{\text{target\_idx}} = i + 14$ ($i + L + k - 1$).
  - Target future timestamp: $t_{\text{target}} = \text{timestamps}[i + 14]$.
  - Exact prediction lead time: $\Delta t_{\text{lead}} = t_{\text{target}} - t_{\text{last}} \approx 0.505\text{ s}$.
- **Boundary Discard Policy:** Trajectory boundaries are strictly guarded; incomplete windows spanning the end of a trajectory are dropped ($N = T - (L + k - 1)$).
- **Generated Dataset Sizes:**
  - `lstm_train.npz`: 186,156 samples
  - `lstm_validation.npz`: 39,496 samples
  - `lstm_test.npz`: 45,121 samples
  - **Total Samples:** 270,773 samples
- **Stored Sample Metadata:** Every NPZ file embeds 9 metadata arrays (`subject_ids`, `video_names`, `trajectory_ids`, `window_start_indices`, `target_indices`, `target_timestamps`, `last_input_timestamps`, `target_yaws`, `target_pitches`), ensuring traceability for downstream evaluations.

---

## 3. Causal Feature Engineering & Normalization

### 3.1 7-Dimensional Feature Vector
At each time step $t$, the state is represented by 7 features:
$$\mathbf{f}_t = \begin{bmatrix} \sin(\theta_{\text{yaw}, t}), & \cos(\theta_{\text{yaw}, t}), & \theta_{\text{pitch}, t}, & v_{\text{yaw}, t}, & v_{\text{pitch}, t}, & a_{\text{yaw}, t}, & a_{\text{pitch}, t} \end{bmatrix}^T$$

1. **Continuous Yaw Representation:**  
   $$\sin(\text{radians}(\theta_{\text{yaw}})), \quad \cos(\text{radians}(\theta_{\text{yaw}}))$$
   Resolves the $0^\circ \leftrightarrow 360^\circ$ circular discontinuity.
2. **Pitch Coordinate:** $\theta_{\text{pitch}} \in [-90^\circ, 90^\circ]$.
3. **Causal Backward Velocity ($t \ge 1$):**
   $$v_{\text{yaw}, t} = \frac{\Delta \theta_{\text{yaw}, t}}{\Delta t_t}, \quad v_{\text{pitch}, t} = \frac{\theta_{\text{pitch}, t} - \theta_{\text{pitch}, t-1}}{\Delta t_t}$$
   where $\Delta \theta_{\text{yaw}, t} = ((\theta_{\text{yaw}, t} - \theta_{\text{yaw}, t-1} + 180^\circ) \pmod{360^\circ}) - 180^\circ$.  
   Boundary: $v_0 = 0.0^\circ/\text{s}$.
4. **Causal Acceleration Policy ($t \ge 2$):**
   $$a_{\text{yaw}, t} = \frac{v_{\text{yaw}, t} - v_{\text{yaw}, t-1}}{\Delta t_t}, \quad a_{\text{pitch}, t} = \frac{v_{\text{pitch}, t} - v_{\text{pitch}, t-1}}{\Delta t_t}$$
   Boundary: $a_0 = 0.0^\circ/\text{s}^2, \quad a_1 = 0.0^\circ/\text{s}^2$.  
   Computing acceleration only when two physical velocity measurements exist removes the unphysical $1000^\circ/\text{s}^2$ spike artifact at $t=1$.

### 3.2 Formal Verification of Causality
A unit test (`test_strict_causality_under_future_mutation`) systematically mutates frames at indices $t > k$ with massive artificial disturbances. For all past and current frames $t \le k$, features remain bitwise identical, proving zero future leakage.

### 3.3 Normalization Fitted Strictly on Training Set
- **Mean Vector ($\boldsymbol{\mu}$):** `[-0.0287, 0.2771, -0.0579, 0.3946, 0.0187, 0.0568, -0.0137]`
- **Std Vector ($\boldsymbol{\sigma}$):** `[0.6410, 0.7145, 18.8318, 42.1055, 19.0014, 165.6360, 111.0417]`
- **Target Pitch Normalization:** $\mu_{\text{pitch}} = -0.0580^\circ, \quad \sigma_{\text{pitch}} = 18.9973^\circ$.
- Test and validation sets are standardized using these exact training parameters.

---

## 4. Final Model Architecture & Training Convergence

### 4.1 Architecture Specifications
- **Model:** `ConfigurableGRU`
- **Recurrent Layer:** 1 GRU layer, 64 hidden units, batch-first.
- **Linear Projection:** Fully connected layer $64 \to 32$ with ReLU activation.
- **Output Head:** Linear layer $32 \to 3$ producing $[\widehat{\sin}(\text{yaw}), \widehat{\cos}(\text{yaw}), \widehat{\text{norm\_pitch}}]$.
- **Dropout:** 0.0 (experimentation demonstrated optimal validation performance with zero dropout).
- **Trainable Parameters:** 16,195
- **Checkpoint Size:** 66.9 KB (`saved_models/final_viewport_model.pth`).

### 4.2 Training Hyperparameters & Convergence
- **Random Seed:** 42 (`random`, `numpy`, `torch`, `torch.use_deterministic_algorithms(True, warn_only=True)`).
- **Optimizer:** Adam ($\text{lr} = 0.0005$).
- **Batch Size:** 256.
- **Loss Function:** MSE Loss.
- **Early Stopping:** Patience = 5 epochs on validation loss.
- **Total Training Duration:** 132.99 seconds (20 epochs).
- **Optimal Checkpoint:** Epoch 18 (Validation Loss: 0.052574, Validation Spherical MAE: 9.6833°).

---

## 5. Verified Experimental Results & Baseline Comparison

### 5.1 Test Set Angular Accuracy Benchmark
Evaluated across all 45,121 unseen test samples from 8 test subjects:

| Model / Baseline Strategy | Type | Parameters | Spherical MAE | Spherical Median | Spherical P90 | Spherical RMSE | Yaw MAE | Pitch MAE | Error Reduction vs Persistence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline (Zero-Order Hold)** | Heuristic | 0 | 13.96° | 10.25° | 30.73° | 19.36° | 12.62° | 4.34° | 0.00% |
| **Constant Velocity Baseline** | Kinematic | 0 | 11.56° | 8.18° | 24.91° | 16.29° | 9.70° | 4.88° | 17.20% |
| **Model A — LSTM-64** | Deep RNN | 20,835 | 9.53° | 6.88° | 20.04° | 13.27° | 8.25° | 3.65° | 31.71% |
| **Model C — GRU-64** | Deep RNN | 16,195 | 9.55° | 6.88° | 20.08° | 13.30° | 8.25° | 3.68° | 31.59% |
| **FINAL OPTIMAL GRU-64** | Deep RNN | **16,195** | **9.45°** | **6.81°** | **19.93°** | **13.19°** | **8.17°** | **3.64°** | **+32.32%** |

### 5.2 Key Accuracy Insights
1. The Final GRU-64 model reduces average spherical error from **13.96° to 9.45°**, achieving a **32.32% reduction** over the Persistence baseline and **18.27% reduction** over Constant Angular Velocity.
2. The 90th percentile error drops from **30.73° down to 19.93°** (-10.80°), demonstrating robust suppression of large tracking deviations during rapid head motions.
3. Decoupled errors show that yaw is the primary source of angular movement ($\text{MAE}_{\text{yaw}} = 8.17^\circ$), while pitch is much more tightly constrained ($\text{MAE}_{\text{pitch}} = 3.64^\circ$).

---

## 6. Real-Time Inference Latency Benchmark

Latency was benchmarked across 2,000 continuous single-sample inferences with 100 warm-up runs under `torch.inference_mode()` on the host CPU:

| Method | Mean Latency | Median Latency | Feasibility Under 10 Hz / 60 Hz Budget |
| :--- | :---: | :---: | :---: |
| Persistence Baseline | 3.67 µs | 3.60 µs | Negligible overhead |
| Constant Velocity Baseline | 11.63 µs | 11.60 µs | Negligible overhead |
| Model A — LSTM-64 | 322.26 µs | 284.85 µs | 0.32% of 100 ms frame interval |
| Model C — GRU-64 | 290.56 µs | 242.90 µs | 0.29% of 100 ms frame interval |
| **FINAL OPTIMAL GRU-64** | **271.67 µs (0.27 ms)** | **249.70 µs (0.25 ms)** | **0.27% of 100 ms frame interval** |

With an inference time of ~0.27 ms, the GRU model consumes only **0.27%** of the 100 ms telemetry sampling period and **1.63%** of a 60 fps video display frame budget (16.67 ms), confirming real-time embedded streaming viability.

---

## 7. Viewport Tile Prediction & Spatial Area Efficiency

### 7.1 Setup & Canvas Layout
- **Canvas Projection:** Equirectangular Panorama (ERP), $3840 \times 1920$ resolution (2:1 aspect ratio).
- **Tile Grid:** 4 rows $\times$ 6 columns = 24 spatial tiles.
- **Tile Dimensions:** $640 \times 480$ pixels ($60^\circ$ yaw $\times 45^\circ$ pitch per tile).
- **Viewport FoV:** $100^\circ$ Horizontal $\times 100^\circ$ Vertical.
- **Ray Sampling Density:** $21 \times 21 = 441$ camera rays.
- **Evaluated Test Subsamples:** 9,025 samples (stride = 5 from 45,121 test samples).

### 7.2 Benchmark Comparison Across Strategies

| Strategy | Tile Recall (Hit Ratio) | Tile Precision | Viewport Coverage (Mean) | 10th Percentile Coverage (P10) | 90th Percentile Coverage (P90) | Avg Tiles Fetched (out of 24) | Tile Area Ratio (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline** | 89.79% | 90.16% | 95.92% | 85.71% | 100.00% | 8.88 | 37.00% |
| **Final GRU Model (No Guard Band)** | 92.55% | 92.99% | **97.99%** | **94.33%** | 100.00% | 8.87 | 36.98% |
| **Final GRU (+1 Guard Band Margin)** | **99.95%** | 45.93% | **99.98%** | **100.00%** | 100.00% | 19.47 | 81.11% |

### 7.3 Guard Band & Area Ratio Analysis
1. **Raw Model vs Baseline:**  
   Without guard bands, the GRU model fetches virtually the same tile area as persistence (8.87 vs 8.88 tiles, ~37.0% of canvas) while boosting visual coverage from **95.92% to 97.99%** (+2.07%) and raising the 10th percentile worst-case coverage from **85.71% to 94.33%** (+8.62%).
2. **Guard Band Trade-off:**  
   Adding a $+1$ spatial ring margin guarantees near-complete coverage (**99.98% mean, 100.00% P10**), buffering against saccades. However, precision drops to 45.93% as 19.47 tiles (81.11% area) are fetched.
3. **Metric Definitions & Distinctions:**
   - **Tile Recall (Hit Ratio):** Fraction of actual ground-truth viewport tiles that were successfully predicted and fetched.
   - **Tile Precision:** Fraction of fetched tiles that actually overlap the user's viewport.
   - **Sampled Viewport Coverage:** Fraction of discrete rays in the user's viewport that land inside fetched tiles.
   - **Tile Area Ratio:** $\frac{\text{Tiles Fetched}}{24} \times 100\%$. Represents the uncompressed spatial pixel area fetched. It does **not** represent compressed video bitrate savings or guarantee zero rebuffering stalls.

---

## 8. Viewport Ray Sampling Sensitivity Analysis

To verify that ray sampling density does not distort tile selection, `ViewportMapper.run_sampling_sensitivity_experiment()` was executed across 5 canonical head orientations across 3 resolutions:
- $21 \times 21$ (441 rays)
- $41 \times 41$ (1,681 rays)
- $81 \times 81$ (6,561 rays)

### Results:
1. **Active Tile Set Consistency:** Across all orientations (equatorial, quadrant centers, boundary seams, near-polar pitch at $+70^\circ$), the set of active tiles was **100% identical** (`active_tiles_match_baseline = True`).
2. **Primary Tile Invariance:** The primary tile (maximum weight) was identical across all resolutions.
3. **Weight Discretization Error:** The maximum observed weight difference between $21\times 21$ and $81\times 81$ was **0.0201** (2.0% weight variance), confirming that $21 \times 21$ provides high fidelity with fast computation.

---

## 9. Known Limitations, Operating Bounds, & Transition to Day 2

### 9.1 Known Limitations
1. **Projection Distortion Near Poles:** Under equirectangular projection, horizontal lines stretch towards $\pm 90^\circ$. Viewports pointing directly at poles activate all 6 tiles in the top or bottom row due to longitudinal convergence.
2. **Prediction Horizon Bounds:** The current model is trained and validated specifically for $k = 5$ frames (~0.5 seconds lead time). For horizons $> 1.0\text{ s}$, multi-step recursive drift will degrade coverage.
3. **Dataset Horizon Discretization:** The AVTrack360 trajectories are sampled at ~10 Hz; fast saccades occurring within $< 100\text{ ms}$ cannot be captured by the telemetry stream.

### 9.2 Day 2 Readiness & Recommendations
The predictive viewport engine is validated, fully causal, and ready for Day 2 (Adaptive Quality Allocation):
1. **Quality Tier Allocation:** Use the predicted tile weights from `ViewportMapper` to drive multi-tier bitrate optimization (e.g., High Quality for primary/inner tiles, Medium for boundary tiles, Low for out-of-viewport fallback).
2. **Network Adaptation Integration:** Combine tile importance weights with network bandwidth estimations to dynamically determine guard band depth.
3. **Strict Codebase Discipline:** Maintain the deterministic testing and assertion suite to prevent regression as quality adaptation scripts are developed.

---

## 10. Audit Conclusion

**Final Verdict:** DAY 1 COMPLETE TECHNICAL BUG FIX AND VALIDATION IS **APPROVED AND READY FOR DAY 2**.

- All 9 audit bugs fixed and verified.
- 23 of 23 unit tests passing (100%).
- All experimental outputs, model weights, and evaluation metrics deterministically validated and committed to canonical documentation.

