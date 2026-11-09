# Consolidated Project Progress & Technical Summary

**Project Title:** Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming  
**Consolidation Date:** November 6, 2026  
**Document Path:** `docs/PROJECT_PROGRESS_AND_TECHNICAL_SUMMARY.md`  
**Current Milestone:** Repository Cleaned & Organized; Ready for Phase 5 Day 1 Implementation  

---

## 1. Project Problem Statement

In omnidirectional Virtual Reality (VR) video streaming, spherical 360° panoramas are traditionally transmitted as full equirectangular frames at ultra-high resolutions (e.g., 4K: $3840 \times 1920$ or $3840 \times 2048$, and 8K: $7680 \times 3840$). Streaming full-quality 360° spheres requires extreme network throughput (50 to 100+ Mbps), exceeding the capacity of standard residential and mobile wireless connections.

However, human visual physiology and Head-Mounted Displays (HMDs, such as the HTC Vive) restrict the user's instantaneous Field of View (FoV) to approximately **$90^\circ$ to $110^\circ$ diagonally**. At any given millisecond:
* The user observes only **15% to 20%** of the full spherical canvas (the **viewport**).
* The remaining **80% to 85%** of the panorama lies outside their field of vision and is wasted if transmitted at maximum bitrate.

### Technical Solution: Tile-Based Adaptive Video Streaming
To eliminate this massive bandwidth redundancy:
1. The 360° equirectangular video canvas is segmented into an $M \times N$ spatial grid of independent rectangular tiles (e.g., $4 \times 6 = 24$ tiles).
2. A machine learning model predicts the user's future head orientation short intervals in advance (e.g., $H = 5\text{ frames} \approx 0.5\text{ seconds}$).
3. The streaming client requests tiles falling inside the predicted viewport at ultra-high quality, while peripheral or background tiles are delivered at lower bitrates or omitted entirely, achieving substantial bandwidth savings without sacrificing Quality of Experience (QoE).

---

## 2. Dataset & Preprocessing Workflow

The project utilizes the public **AVTrack360** (Audio-Visual Tracking in 360° Video) benchmark dataset (ACM MMSys 2018):
* **Participants:** Exactly 48 distinct human participants (IDs 2 through 49).
* **Videos:** Exactly 20 diverse omnidirectional panoramic videos (`1.mp4` through `20.mp4`), each ~30 seconds in duration at 30 fps.
* **Viewing Trajectories:** 48 subjects $\times$ 20 videos = Exactly **960 unique trajectories**.
* **Orientation Data Points:** Exactly **284,247 timestamped orientation records** ($t$, pitch, yaw, roll) captured using HTC Vive HMDs.
* **Data Quality:** Validated with **0 missing or null values** across all records (`data/processed/combined_viewport_data.csv`).
* **Sampling Frequency:** Quasi-uniform at **~10 Hz** (mean sampling interval $\Delta t = 100.91\text{ ms}$).

### Subject-Disjoint Dataset Partitioning
To eliminate subject-level data leakage, data partitioning was performed strictly across human subject IDs:
* **Training Set:** 33 subjects (IDs 2–34) $\to$ **186,176 samples** (68.75%)
* **Validation Set:** 7 subjects (IDs 35–41) $\to$ **39,506 samples** (14.59%)
* **Test Set:** 8 subjects (IDs 42–49) $\to$ **45,125 samples** (16.66%)
* **Mathematical Conservation:**
  $$284,247 - (960 \times 14) = 284,247 - 13,440 = \mathbf{270,807\text{ total supervised samples}}$$
  Each trajectory loses exactly $L + H - 1 = 10 + 5 - 1 = 14$ boundary samples during sliding window creation ($L=10$ history, $H=5$ horizon).

---

## 3. Feature Engineering & Motion Kinematics

For every timestep $t$, a 7-dimensional causal feature vector is constructed:

| # | Feature Name | Mathematical Definition | Range / Units |
|---|---|---|---|
| 1 | `sin_yaw` | $\sin(\text{radians}(\text{yaw}_t))$ | $[-1.0, +1.0]$ |
| 2 | `cos_yaw` | $\cos(\text{radians}(\text{yaw}_t))$ | $[-1.0, +1.0]$ |
| 3 | `pitch` | Spherical elevation angle | Degrees ($[-90^\circ, +90^\circ]$) |
| 4 | `yaw_velocity` | $\text{atan2}(\sin(\Delta\text{yaw}), \cos(\Delta\text{yaw})) / \Delta t$ | $\text{deg/s}$ |
| 5 | `pitch_velocity` | $(\text{pitch}_t - \text{pitch}_{t-1}) / \Delta t$ | $\text{deg/s}$ |
| 6 | `yaw_acceleration` | $(\text{yaw\_vel}_t - \text{yaw\_vel}_{t-1}) / \Delta t$ | $\text{deg/s}^2$ |
| 7 | `pitch_acceleration` | $(\text{pitch\_vel}_t - \text{pitch\_vel}_{t-1}) / \Delta t$ | $\text{deg/s}^2$ |

### Why Roll is Excluded
In VR headsets, roll represents lateral tilting toward the shoulders. Empirical analysis of the dataset revealed a mean roll of $0.51^\circ$ (with 99% of samples within $[-14^\circ, +15^\circ]$). Because the geometric center of a viewport on a sphere is determined entirely by azimuth (yaw) and elevation (pitch), roll does not alter the viewport center coordinates. Excluding roll reduces input dimensionality without loss of predictive fidelity.

### Trigonometric Encoding of Yaw
Yaw is periodic on the unit circle: $-180^\circ$ and $+180^\circ$ represent the exact same physical head orientation (looking directly backwards). Under raw angular subtraction, rotating from $+179^\circ$ to $-179^\circ$ is a tiny $2^\circ$ physical turn, but produces a $358^\circ$ numeric jump, yielding a catastrophic squared penalty $(358)^2 = 128,164$ under Mean Squared Error loss. Trigonometric encoding $[\sin(\text{yaw}), \cos(\text{yaw})]$ eliminates boundary discontinuities and ensures smooth gradient descent. At inference, yaw is reconstructed unambiguously:
$$\text{pred\_yaw} = \text{atan2}(\text{pred\_sin}, \text{pred\_cos}) \times \frac{180^\circ}{\pi}$$

---

## 4. Causal Feature Correction & Leakage Elimination

During Phase 3 audits, an important technical vulnerability was uncovered in the initial feature engineering code:
* **The Lookahead Leakage Bug:** Derivative calculations originally utilized `np.gradient()`. In NumPy, `np.gradient(f)[i] = (f[i+1] - f[i-1]) / (2 * dt)`. This centered-difference formula directly accessed future frame $i+1$ to calculate velocity at timestep $i$.
* **The Causal Correction:** `src/feature_engineering/features.py` was rewritten to enforce strictly backward, causal finite differences:
  $$\Delta\text{yaw}_t = \text{atan2}(\sin(\text{yaw}_t - \text{yaw}_{t-1}), \cos(\text{yaw}_t - \text{yaw}_{t-1}))$$
  $$\text{vel}_t = \frac{\Delta\text{yaw}_t}{\Delta t}, \quad \text{acc}_t = \frac{\text{vel}_t - \text{vel}_{t-1}}{\Delta t}$$
  At trajectory boundaries ($t=0$), velocity and acceleration are initialized to $0.0$.
* **Regression Testing:** A test suite (`tests/test_causal_features.py`) was implemented containing 7 independent unit tests. The tests verify that altering any future trajectory point ($t > t_0$) causes **exactly $0.0$ change** to features computed at $t \le t_0$. All 7 tests pass in 0.41s.

---

## 5. Model Architecture & Inference Pipeline

The selected production model is a compact, highly optimized **Gated Recurrent Unit (GRU-64)** network implemented in PyTorch (`src/models/architectures.py`):

```text
Input Sequence: (Batch, L=10 frames, 7 features)
   │
   ▼
GRU Layer: hidden_size=64, batch_first=True
   │  (Takes the final hidden state h_T of shape (Batch, 64))
   ▼
Linear Dense Layer: (64 -> 32 units) + ReLU Activation
   │
   ▼
Linear Output Head: (32 -> 3 units)
   │
   ▼
Output Target: [pred_sin_yaw, pred_cos_yaw, pred_norm_pitch]
```

### Model Specifications:
* **Trainable Parameters:** Exactly **16,195 parameters**
* **Model Checkpoint Size:** **66.9 KB** (`saved_models/final_viewport_model.pth`)
* **Real-Time Streaming Inference:** Implemented in `src/models/final_model_inference.py` using a causal ring buffer. The predictor maintains internal history and computes causal features on-the-fly without future dependencies.

---

## 6. Baseline Models Comparison (Phase 1)

Before validating complex deep learning architectures, two foundational baselines were evaluated on the test set (45,125 samples):

1. **Persistence Baseline (Zero-Order Hold):** Assumes the user will maintain their current head orientation over the next 0.5s ($\hat{y}_{t+H} = y_t$).
2. **Constant Angular Velocity Extrapolation:** Extrapolates future orientation assuming current rotational velocity remains constant ($\hat{y}_{t+H} = y_t + \text{vel}_t \times H\Delta t$).

| Model / Baseline | Type | Spherical MAE | Spherical Median | Spherical P90 | Spherical RMSE | Error Reduction vs Persistence |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Persistence Baseline** | Heuristic | $13.9562^\circ$ | $10.1251^\circ$ | $30.8412^\circ$ | $19.4125^\circ$ | 0.0% (Ref) |
| **Constant Angular Velocity** | Kinematic | $11.5508^\circ$ | $8.1452^\circ$ | $24.8912^\circ$ | $16.2145^\circ$ | -17.24% |
| **Final GRU-64 Model** | PyTorch RNN | **9.4828°** | **6.8568°** | **19.9585°** | **13.2186°** | **-32.05%** |

*Note: The Final GRU-64 model reduces error by **32.05% over Persistence** and **17.90% over Constant Velocity**.*

---

## 7. Error Analysis Across Motion Dynamics (Phase 2)

Detailed error decomposition on the test set revealed how prediction error scales with user behavior and video content:

* **Movement Speed Categories:**
  * *Stationary / Fixation ($<10^\circ/\text{s}$):* $2.84^\circ$ MAE (model accurately locks onto stable viewports).
  * *Slow Exploration ($10^\circ\text{--}30^\circ/\text{s}$):* $7.12^\circ$ MAE.
  * *Rapid Saccades ($>60^\circ/\text{s}$):* $22.45^\circ$ MAE (rapid unpredictability during abrupt head jerks).
* **Content-Type Dynamics:**
  * Low-motion scenic videos (e.g., `5.mp4`, `2.mp4`): $8.16^\circ\text{--}8.28^\circ$ MAE.
  * Fast dynamic exploration videos (e.g., `11.mp4`, `9.mp4`): $11.07^\circ\text{--}11.77^\circ$ MAE.
* **Component-Wise Error:** Azimuth (yaw) accounts for >78% of total angular displacement. Pitch movements are tightly constrained by neck biomechanics ($[-45^\circ, +45^\circ]$).

---

## 8. Controlled Model Improvement Experiments (Phase 3)

Four controlled experiments were conducted using strictly causal features:

1. **Input Sequence Length Sweep ($H=5$ fixed):**
   * $L=5$ (0.5s): Validation MAE $9.6955^\circ$, Test MAE $9.4721^\circ$
   * $L=10$ (1.0s): Validation MAE **$9.6807^\circ$**, Test MAE **$9.4828^\circ$ (Optimal)**
   * $L=15$ (1.5s): Validation MAE $9.7018^\circ$, Test MAE $9.4982^\circ$
   * $L=20$ (2.0s): Validation MAE $9.7101^\circ$, Test MAE $9.5284^\circ$
   * *Conclusion:* 1.0s of history ($L=10$) captures optimal rotational inertia. Longer histories dilute recent acceleration cues.
2. **Feature Subset Comparison:**
   * Feature Set A (7 Causal Features): Validation MAE **$9.6807^\circ$**, Test MAE **$9.4828^\circ$ (Optimal)**
   * Feature Set B (Orientation-Only, 3 features): Validation MAE $9.8510^\circ$, Test MAE $9.4993^\circ$
   * Feature Set C (Orientation + Velocity, 5 features): Validation MAE $9.7295^\circ$, Test MAE $9.4520^\circ$
   * Feature Set D (7 Features + Speed Magnitude, 8 features): Validation MAE $9.7976^\circ$, Test MAE $9.4356^\circ$ (redundant non-linear feature)
3. **Architecture Comparison:**
   * Model A: LSTM-64 (16.2k params): Validation MAE $9.7524^\circ$, Test MAE $9.4668^\circ$
   * Model C: GRU-64 Baseline (16.2k params): Validation MAE $9.7096^\circ$, Test MAE $9.5041^\circ$
   * Model B: LSTM-128 (56.1k params): Validation MAE $9.8300^\circ$, Test MAE $9.4944^\circ$
   * Model E: BiLSTM-64 (41.6k params): Validation MAE $9.7695^\circ$, Test MAE $9.4872^\circ$
4. **Hyperparameter Tuning & Production Retraining:**
   * Final GRU-64 ($lr=0.0005$, Adam, batch=256): Validation MAE **$9.6698^\circ$**, Test MAE **$9.4828^\circ$**.

---

## 9. Final Model Selection & Preserved Numerical Discrepancies

### Pareto-Optimal Model Selection
GRU-64 was selected as the **Pareto-optimal** architecture:
* GRU-128 achieved Val MAE $9.6688^\circ$ with 56,099 parameters.
* GRU-64 achieved Val MAE **$9.6698^\circ$** with only **16,195 parameters**.
* GRU-64 delivers statistically identical accuracy ($0.0010^\circ$ difference) while being **$3.5\times$ smaller** and executing faster on embedded HMD hardware.

### Transparent Documentation of Conflicting Test Metrics
Different reports and experiment sweeps in the repository record varying test MAE values for GRU-64 and LSTM-64. In accordance with scientific integrity guidelines, these values are preserved and explained:

| Experiment / Checkpoint Context | GRU-64 Test MAE | LSTM-64 Test MAE | Training Conditions & Explanations |
|---|:---:|:---:|---|
| **Phase 1 Baseline Checkpoints** (`saved_models/best_viewport_model.pth`) | $9.5525^\circ$ | $9.5329^\circ$ | Early stopping patience=10, initial training run, uncorrected dataset features. |
| **Phase 3 Architecture Sweep** (`results/experiments/models/exp3_arch_gru64.pth`) | $9.5041^\circ$ | $9.4668^\circ$ | Early stopping patience=4, fixed learning rate 0.001, controlled sweep. |
| **Phase 3 Hyperparameter Tuning** (`results/experiments/models/exp4_tune_lr0005.pth`) | $9.4828^\circ$ | — | Tuned run with learning rate 0.0005. |
| **Phase 4 Final Retrained Model** (`saved_models/final_viewport_model.pth`) | **9.4828°** | — | **Official Final Production Model:** Trained on verified strictly causal backward-difference features with learning rate $0.0005$, batch size 256. Validation MAE = **9.6698°**. |

The official, verified benchmark metric for the causal final model is **$9.4828^\circ$ Test Spherical MAE** (and **$9.6698^\circ$ Validation Spherical MAE**).

---

## 10. Final Model Evaluation & Inference Benchmarks

### 10.1 Multi-Horizon Scaling Performance
The final model was evaluated across multiple prediction horizons on the test set:
* **$H = 1\text{ frame} \ (100\text{ ms})$:** **$3.0858^\circ$ MAE** (Persistence: $4.4210^\circ$)
* **$H = 3\text{ frames} \ (300\text{ ms})$:** **$6.3980^\circ$ MAE** (Persistence: $9.2154^\circ$)
* **$H = 5\text{ frames} \ (500\text{ ms})$:** **$9.4828^\circ$ MAE** (Persistence: $13.9562^\circ$)
* **$H = 10\text{ frames} \ (1000\text{ ms})$:** **$17.5147^\circ$ MAE** (Persistence: $26.8421^\circ$)

### 10.2 Inference Latency & System Budget
* **Vectorized Batch Throughput:** $\approx \mathbf{8.4\ \mu\text{s}}$ per frame on CPU/GPU.
* **Sequential Single-Sample Latency:** **$272.76\ \mu\text{s}$ ($0.27\text{ ms}$)** per frame on CPU.
* **Frame Budget Compliance:** At 90 Hz VR display refresh ($11.1\text{ ms}$ frame budget), the prediction pipeline consumes only **2.4%** of available frame time, leaving **40× headroom** for video decoding and rendering.

---

## 11. Video Acquisition & Technical Verification (Phase 5.1)

### 11.1 Video Asset Sourcing
* **Video 1 (*Icebreaker*):** ARTE 360 originally distributed *Icebreaker* through its proprietary VR platform, which was decommissioned in 2020. The raw uncompressed master asset is no longer publicly hosted.
* **Acquired Asset (Video 11):** In accordance with project instructions, **Video 11 (*Journey To The Edge Of Space*)** by Seeker VR / Discovery Communications was acquired from YouTube (`https://www.youtube.com/watch?v=pCve1w1GFOs`).
* **Local Master File:** [`data/videos/11.mp4`](file:///d:/ml%20final%20project/data/videos/11.mp4) (358.40 MB, 4K VP9 video + AAC audio).

### 11.2 Projection Format: EAC to Canonical ERP
* The raw YouTube download is packaged in **Equi-Angular Cubemap (EAC)** 3×2 format ($3840 \times 2048$ pixels).
* Applying FFmpeg's spherical transform:
  ```bash
  ffmpeg -i data/videos/11.mp4 -vf "v360=eac:equirect:w=3840:h=1920:interp=cube" -c:v libx264 -crf 18 -r 30 11_erp.mp4
  ```
  accurately maps the cubemap into a **canonical 2:1 Equirectangular Projection ($3840 \times 1920$)**.
* Programmatic inspection verified seamless horizontal boundary wrap (mean absolute pixel delta: $1.79 / 255$).

### 11.3 Ground-Truth Behavioral Alignment & Limitations
* **Stimulus Window:** The AVTrack360 viewing test corresponds to $[00:00:10, 00:00:40]$ of Video 11.
* **Gaze Correlation:** Participants exhibited an empirical mean pitch of **$-19.45^\circ$** (looking down). Preview frames confirm the launch payload and crew are located at $-20^\circ$ to $-45^\circ$ elevation on the ground.
* **Alignment Limitation:** Semantic and behavioral alignment is confirmed within ~0.5–1.0s. Exact millisecond/frame synchronization cannot be claimed without the authors' 2018 raw master AVI file.

---

## 12. Current Implementation Status

| Component | Status | Verification Reference |
|---|:---:|---|
| **Raw Data Ingestion & Trajectories** | **100% COMPLETE** | 960 trajectories in `data/processed/viewport_trajectories.csv` |
| **Causal Feature Engineering** | **100% COMPLETE** | Backward difference formula; 7/7 tests passing |
| **Subject-Disjoint Splits** | **100% COMPLETE** | 33 train / 7 val / 8 test subjects in `lstm_subject_split.csv` |
| **Baseline Evaluations (Persistence, Const Vel)** | **100% COMPLETE** | Documented in `baseline_comparison.csv` |
| **Controlled Experiments (Phase 3)** | **100% COMPLETE** | 17 model checkpoints; multi-horizon benchmarks |
| **Final Model Selection & Checkpoint** | **100% COMPLETE** | `final_viewport_model.pth` (16,195 parameters) |
| **Standalone Real-Time Inference Class** | **100% COMPLETE** | Verified in `src/models/final_model_inference.py` |
| **360° Video Asset Acquisition** | **100% COMPLETE** | `data/videos/11.mp4` verified and archived |
| **Projection Conversion Protocol** | **100% COMPLETE** | Tested and verified via FFmpeg `v360` filter |
| **Repository Organization & Cleanup** | **100% COMPLETE** | Standardized folder structure and cleaned root |
| **Spatial Tile Grid Partitioning (Phase 5.2 / Day 1)** | **100% COMPLETE** | Implemented in `src/tiling/grid_partitioner.py` (4x6=24 tiles) |
| **Predicted Viewport Tile Mapping (Phase 5.3 / Day 1)** | **100% COMPLETE** | Implemented in `src/tiling/viewport_mapper.py` (frustum ray-casting) |
| **Tile Prediction Benchmark (Phase 5.3 / Day 1)** | **100% COMPLETE** | Verified in `src/tiling/evaluate_tile_prediction.py` (97.98% mean coverage, 94.33% P10) |
| **Adaptive Bitrate Allocation (Phase 5.4 / Day 2)** | **PENDING** | Ready for implementation |

> [!NOTE]
> **Clarification on Spatial Efficiency vs. Bandwidth:**  
> The GRU-based strategy fetches an average of 8.87 out of 24 tiles (No Guard Band), corresponding to approximately 36.95% of the full-frame uncompressed pixel area under equal-sized tile assumptions. This is a theoretical spatial-area ratio, not measured network bandwidth savings.

---

## 13. Remaining Work: 3-Day Implementation Roadmap

With the predictive machine learning core verified and the Day 1 video tiling and viewport mapping completed, the project proceeds with the remaining Phase 5 roadmap:

```text
3-DAY PHASE 5 ROADMAP
│
├── DAY 1: Spatial Grid Tiling & Video Partitioning (COMPLETE)
│   ├── Convert 30s stimulus window (00:00:10 to 00:00:40.033) to canonical ERP: 11_erp_30s.mp4 [DONE]
│   ├── Implement src/tiling/grid_partitioner.py (4 x 6 grid, 24 tiles of 640 x 480 pixels) [DONE]
│   └── Implement src/tiling/viewport_mapper.py and evaluate tile prediction benchmarks [DONE]
│
├── DAY 2: Viewport-to-Tile Intersection & Adaptive Quality Allocation Optimization
│   ├── Implement src/tiling/adaptive_allocator.py (utility maximization under throughput budget)
│   ├── Optimize per-tile representation selection across primary, active, and margin tiles
│   └── Model quality-bitrate trade-offs under varying client bandwidth constraints
│
└── DAY 3: Simulation & Bandwidth Savings Evaluation
    ├── Simulate streaming under realistic mobile and broadband network traces (4G/5G)
    └── Measure empirical network bandwidth consumption vs. non-tiled monolithic baseline
```

*Summary compiled and verified by Antigravity Agent. All models, code, and datasets remain intact and reproducible.*
