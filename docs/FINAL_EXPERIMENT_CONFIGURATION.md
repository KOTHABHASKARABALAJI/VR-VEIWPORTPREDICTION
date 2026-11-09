# Final Experiment Configuration & System Specification

**Project:** Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming  
**Phase:** Phase 5, Day 1 — Final Configuration Archive  
**Date:** November 2026  
**Status:** CANONICAL REFERENCE  

---

## 1. Hardware & Software Environment

- **Operating System:** Windows 11 Pro 64-bit (Build 10.0.26300)
- **Host CPU:** Intel Core i7-12700H (14 cores, 20 logical threads)
- **Python Version:** 3.14.2 64-bit (`pythoncore-3.14-64`)
- **PyTorch Version:** 2.14.0+cpu
- **NumPy Version:** 2.4.4
- **Pandas Version:** 2.3.0
- **Random Seed:** 42 (`random`, `numpy`, `torch.manual_seed(42)`, `torch.use_deterministic_algorithms(True, warn_only=True)`)

---

## 2. Dataset Specifications

- **Benchmark Dataset:** AVTrack360 (Head-mounted VR viewport trajectories)
- **Source File:** `data/processed/viewport_trajectories.csv`
- **Raw Telemetry Rows:** 284,247 across 960 trajectories
- **Cleaned Telemetry Rows:** 284,213 across 960 trajectories (34 initialization duplicates deduplicated at $t = 0.0\text{s}$)
- **Unique Subjects:** 48 subjects (IDs 2 through 49)
- **Sampling Frequency (Nominal):** $9.91\text{ Hz}$ (~10 Hz)
- **Sampling Interval ($\Delta t$):** Mean = $0.10091\text{ s}$, Median = $0.10100\text{ s}$, Min = $0.00200\text{ s}$, Max = $0.43300\text{ s}$
- **Sliding Window History Length ($L$):** 10 steps (nominal 1.009 seconds)
- **Prediction Horizon ($k$):** 5 steps ahead (nominal 0.505 seconds lead time)
- **Target Index Formula:** $\text{target\_idx} = \text{window\_start\_idx} + 14$

### Subject Split Mapping (0.0% Overlap)
- **Train (33 subjects, 186,156 samples):**  
  `[2, 3, 5, 6, 7, 8, 10, 11, 13, 14, 15, 17, 18, 19, 21, 23, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 39, 41, 42, 45, 46, 47, 48]`
- **Validation (7 subjects, 39,496 samples):**  
  `[4, 12, 24, 25, 37, 38, 43]`
- **Test (8 subjects, 45,121 samples):**  
  `[9, 16, 20, 22, 30, 40, 44, 49]`
- **Total Supervised Samples:** 270,773 samples

---

## 3. Feature Definitions & Normalization

At each time step $t$, the 7-dimensional causal feature vector is:
$$\mathbf{f}_t = \begin{bmatrix} \sin(\theta_{\text{yaw}, t}), & \cos(\theta_{\text{yaw}, t}), & \theta_{\text{pitch}, t}, & v_{\text{yaw}, t}, & v_{\text{pitch}, t}, & a_{\text{yaw}, t}, & a_{\text{pitch}, t} \end{bmatrix}^T$$

1. $\sin(\text{radians}(\theta_{\text{yaw}}))$: Continuous yaw coordinate.
2. $\cos(\text{radians}(\theta_{\text{yaw}}))$: Continuous yaw coordinate.
3. $\theta_{\text{pitch}}$: Pitch in degrees, bounded to $[-90^\circ, 90^\circ]$.
4. $v_{\text{yaw}, t}$: Causal backward angular velocity ($\Delta \theta_{\text{yaw}, t} / \Delta t_t$), circular difference tie-breaking at $-180.0^\circ$. Boundary: $v_0 = 0.0$.
5. $v_{\text{pitch}, t}$: Causal backward pitch velocity ($(\theta_{\text{pitch}, t} - \theta_{\text{pitch}, t-1}) / \Delta t_t$). Boundary: $v_0 = 0.0$.
6. $a_{\text{yaw}, t}$: Causal backward yaw acceleration ($(v_{\text{yaw}, t} - v_{\text{yaw}, t-1}) / \Delta t_t$). Boundary: $a_0 = 0.0, a_1 = 0.0$ (evaluated for $t \ge 2$).
7. $a_{\text{pitch}, t}$: Causal backward pitch acceleration ($(v_{\text{pitch}, t} - v_{\text{pitch}, t-1}) / \Delta t_t$). Boundary: $a_0 = 0.0, a_1 = 0.0$ (evaluated for $t \ge 2$).

### Normalization Statistics (Fitted Exclusively on Train Set)
- **Feature Means ($\boldsymbol{\mu}$):**  
  `[-0.02867465, 0.27714762, -0.05785309, 0.39461410, 0.01870017, 0.05675898, -0.01371846]`
- **Feature Standard Deviations ($\boldsymbol{\sigma}$):**  
  `[0.64103818, 0.71445644, 18.83176804, 42.10546494, 19.00144958, 165.63604736, 111.04169464]`
- **Target Pitch Statistics:** $\mu_{\text{pitch}} = -0.05795139^\circ, \quad \sigma_{\text{pitch}} = 18.99729919^\circ$

---

## 4. Model Architecture & Hyperparameters

- **Architecture Class:** `ConfigurableGRU`
- **Input Dimension:** 7
- **Recurrent Layer:** 1 layer GRU, 64 hidden units, batch-first
- **Linear Projection:** Fully connected layer $64 \to 32$, ReLU activation
- **Output Layer:** Linear layer $32 \to 3$ ($[\widehat{\sin}(\text{yaw}), \widehat{\cos}(\text{yaw}), \widehat{\text{norm\_pitch}}]$)
- **Dropout:** 0.0
- **Trainable Parameters:** 16,195
- **Optimizer:** Adam ($\text{lr} = 0.0005$, $\beta_1 = 0.9, \beta_2 = 0.999$)
- **Batch Size:** 256
- **Loss Criterion:** MSE Loss
- **Max Epochs:** 20 (Early stopping patience = 5 on validation loss)
- **Best Validation Epoch:** Epoch 18 (Validation Loss: 0.052574, Validation Spherical MAE: 9.6833°)
- **Model Checkpoint:** `saved_models/final_viewport_model.pth` (SHA-256: `45a7dbec9063411cb80074130d6391fccc7036f4fe34164b93954326c1356c38`)

---

## 5. Tile Grid & Viewport Projection Specifications

- **Canvas Format:** Equirectangular Panorama (ERP)
- **Canvas Resolution:** $3840 \times 1920$ pixels (2:1 aspect ratio)
- **Grid Layout:** 4 rows $\times$ 6 columns = 24 spatial tiles
- **Tile Dimensions:** $640 \times 480$ pixels ($60^\circ$ yaw span $\times 45^\circ$ pitch span per tile)
- **Solid Angle per Tile:**
  - Rows 0 & 3 (polar regions): $0.306717\text{ sr}$
  - Rows 1 & 2 (equatorial regions): $0.740480\text{ sr}$
  - Total Canvas Solid Angle: $\sum \Omega = 12.566364\text{ sr} \approx 4\pi\text{ sr}$
- **Viewport Field of View (FoV):** $100.0^\circ$ Horizontal $\times 100.0^\circ$ Vertical
- **Ray Tracing Grid:** $21 \times 21 = 441$ camera rays (discrete ray weights sum to $1.000000$)

---

## 6. Official Verified Performance Metrics

### Test Set Prediction Accuracy (45,121 Unseen Samples, 8 Test Subjects)
- **Final Optimal GRU-64 Spherical MAE:** **9.4454°**
- **Final Optimal GRU-64 Spherical Median:** **6.8054°**
- **Final Optimal GRU-64 Spherical P90:** **19.9330°**
- **Final Optimal GRU-64 Spherical RMSE:** **13.1949°**
- **Final Optimal GRU-64 Yaw MAE:** **8.1654°**
- **Final Optimal GRU-64 Pitch MAE:** **3.6428°**
- **Persistence Baseline Spherical MAE:** 13.9567° (+32.32% improvement)
- **Constant Velocity Baseline Spherical MAE:** 11.5562° (+18.27% improvement)
- **Model A (LSTM-64) Baseline Spherical MAE:** 9.5304°
- **Model C (GRU-64) Baseline Spherical MAE:** 9.5474°

### Latency Benchmark (2,000 Repetitions, Host CPU)
- **Final GRU-64 Mean Inference Latency:** **271.67 µs (0.27 ms)**
- **Final GRU-64 Median Inference Latency:** **249.70 µs (0.25 ms)**

### Tile Evaluation Metrics (9,025 Subsampled Test Samples)
- **Persistence Baseline:**
  - Recall: 89.66% | Precision: 90.26% | Coverage: 95.78% (P10: 85.49%, P90: 100.0%) | Avg Tiles: 8.87 (36.96% Area)
- **Final GRU (No Guard Band):**
  - Recall: 92.42% | Precision: 92.98% | Coverage: **97.93%** (P10: **94.33%**, P90: 100.0%) | Avg Tiles: 8.87 (36.97% Area)
- **Final GRU (+1 Guard Band Margin):**
  - Recall: **99.96%** | Precision: 45.99% | Coverage: **99.99%** (P10: **100.0%**, P90: 100.0%) | Avg Tiles: 19.47 (81.11% Area)

