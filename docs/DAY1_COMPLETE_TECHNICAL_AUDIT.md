# Complete Day 1 Technical Audit: Viewport Prediction & Video Tiling Pipeline

**Project:** Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming  
**Date:** November 8, 2026  
**Auditor:** Autonomous Senior Research Engineering & Quality Assurance Agent  
**Scope:** Independent technical audit of all work completed through Day 1 of Phase 5 (and foundational Phases 1–4)  
**Deliverable File:** `docs/DAY1_COMPLETE_TECHNICAL_AUDIT.md`  

---

## 1. Executive Summary

This independent technical audit was conducted on Day 1 of Phase 5 to rigorously evaluate whether the existing dataset preparation, predictive machine learning models, video conversion pipeline, spatial tile partitioner, viewport mapper, and empirical evaluation metrics are scientifically sound, mathematically correct, and fully reproducible.

### Audit Summary:
1. **Machine Learning Pipeline (Phases 1–4):** The underlying dataset preprocessing, subject-disjoint splits (33 train, 7 val, 8 test), causal backward-difference feature engineering, and PyTorch `ConfigurableGRU` checkpoint (`saved_models/final_viewport_model.pth`) are **100% verified and free of data leakage**.
2. **Canonical Test MAE Discrepancy Resolved:** Direct evaluation of the final production checkpoint on the 45,125 test samples produces a **Test Spherical MAE of $9.4828^\circ$** (matching `final_model_evaluation_metrics.csv` and `final_model_test_predictions.csv` to 6 decimal places). The previously reported figure of $9.7088^\circ$ in `PROJECT_PROGRESS_AND_TECHNICAL_SUMMARY.md` was a documentation transcription error.
3. **Video Conversion (Phase 5.1):** The conversion of the 30-second stimulus segment from `data/videos/11.mp4` to canonical 2:1 Equirectangular Projection (`data/videos/11_erp_30s.mp4`) at $3840 \times 1920$ resolution yields **exactly 901 frames** at 29.97 fps, perfectly matching the duration specified in the AVTrack360 metadata (`SRC_info/11.json`). Semantic and behavioral alignment is verified; exact millisecond-level frame synchronization is properly bounded as an unverified limitation.
4. **Spatial 24-Tile Partitioner (`grid_partitioner.py`):** Verified. All 24 tiles have exact $640 \times 480$ pixel bounds, $60^\circ \times 45^\circ$ angular spans, exact spherical solid angles summing to $4\pi$ steradians ($12.566364\text{ sr}$), seamless circular horizontal wraparound across $\pm 180^\circ$, and zero-error coordinate round-trip mapping.
5. **3D Viewport-to-Tile Mapper (`viewport_mapper.py`):** Verified. Ray-frustum projection correctly rotates viewing vectors into spherical space. All tile weights are in $(0, 1]$, sum to $1.0$, and primary line-of-sight tiles are consistently contained in the active set. The implementation is identified as a numerical ray-sampling approximation (not exact analytical polygon clipping).
6. **Critical Metric Labeling Bug Identified (`evaluate_tile_prediction.py`):** The reported metric `"90th Percentile Coverage: 94.33%"` was caused by calling `np.percentile(model_coverages, 10)` in the code. $94.33\%$ is mathematically the **10th percentile** ($P_{10}$, representing a 90% coverage lower bound), whereas the true 90th percentile ($P_{90}$) is **$100.00\%$** (with a median of $100.00\%$ and mean of $98.00\%$).
7. **Scientific Overstatement Corrected:** The reported ratio $8.87 / 24 = 36.97\%$ is a **spatial tile-area / raw fetched-pixel ratio**, NOT measured compressed network bandwidth savings.
8. **Test Suite & Reproducibility:** All 13 unit tests pass cleanly in $0.099\text{ seconds}$. Polar singular edge cases ($\pm 90^\circ$) execute without NaN or division errors. All modules import and run cleanly.

### Final Readiness Verdict:
**B. READY AFTER CORRECTIONS** — The technical implementation, mathematical geometry, video assets, and model checkpoints are robust and functioning properly; however, specific documentation errors (test MAE discrepancy, coverage percentile labeling, and bandwidth claims) must be formally corrected before proceeding to Day 2 adaptive quality allocation.

---

## 2. Dataset and GRU Prediction Audit

### 2.1 Subject-Independent Dataset Splitting
- **Source File:** [`data/processed/lstm_subject_split.csv`](file:///d:/ml%20final%20project/data/processed/lstm_subject_split.csv)
- **Train Subjects (33):** `[2, 3, 5, 6, 7, 8, 10, 11, 13, 14, 15, 17, 18, 19, 21, 23, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 39, 41, 42, 45, 46, 47, 48]`
- **Validation Subjects (7):** `[4, 12, 24, 25, 37, 38, 43]`
- **Test Subjects (8):** `[9, 16, 20, 22, 30, 40, 44, 49]`
- **Disjointness Check:**
  - $\text{Train} \cap \text{Validation} = \emptyset$
  - $\text{Train} \cap \text{Test} = \emptyset$
  - $\text{Validation} \cap \text{Test} = \emptyset$
- **Sequence Counts & Shapes:**
  - Train: `(186176, 10, 7)` sequences, targets `(186176, 3)`
  - Validation: `(39506, 10, 7)` sequences, targets `(39506, 3)`
  - Test: `(45125, 10, 7)` sequences, targets `(45125, 3)`
- **Audit Status:** **VERIFIED** (No subject overlap exists).

### 2.2 Feature Engineering Causality & Target Construction
- **Source File:** [`src/feature_engineering/features.py`](file:///d:/ml%20final%20project/src/feature_engineering/features.py)
- **Features Extracted (7):**
  1. $\sin(\text{yaw})$
  2. $\cos(\text{yaw})$
  3. $\text{pitch}$ (degrees)
  4. $\text{yaw\_vel} = \frac{\Delta_{\text{circ}} \text{yaw}}{\Delta t}$ (causal backward difference)
  5. $\text{pitch\_vel} = \frac{\Delta \text{pitch}}{\Delta t}$ (causal backward difference)
  6. $\text{yaw\_acc} = \frac{\Delta \text{yaw\_vel}}{\Delta t}$ (causal backward difference)
  7. $\text{pitch\_acc} = \frac{\Delta \text{pitch\_vel}}{\Delta t}$ (causal backward difference)
- **Future Leakage Check:** Verified via [`tests/test_causal_features.py`](file:///d:/ml%20final%20project/tests/test_causal_features.py): altering future frames at $t > k$ causes zero change in features at $t \le k$. Centered differences and `np.gradient` were completely removed.
- **Normalization Isolation:**
  - `norm_mean` and `norm_std` are computed strictly from the 186,176 training sequences (`data/processed/lstm_train.npz`).
  - Validation and test splits in NPZ are standardized using the training split's statistics.
  - Target pitch normalization uses index 2 parameters: $\mu = -0.058101$, $\sigma = 18.830423$.
- **Audit Status:** **VERIFIED**.

### 2.3 Model Architecture & Production Inference
- **Model Checkpoint:** [`saved_models/final_viewport_model.pth`](file:///d:/ml%20final%20project/saved_models/final_viewport_model.pth) (66.9 KB, 16,195 trainable parameters).
- **Architecture:** `ConfigurableGRU` (`input_dim=7, hidden_dim=64, proj_dim=32, output_dim=3, dropout=0.0`).
- **Inference Pipeline:** [`src/models/final_model_inference.py`](file:///d:/ml%20final%20project/src/models/final_model_inference.py) implements `FinalViewportPredictor`:
  - Consumes 10 historical frames ($1.0\text{ s}$ context).
  - Uses strictly causal `extract_7_features()`.
  - Normalizes using `saved_models/final_model_config.json`.
  - Decodes $\hat{\text{yaw}} = \text{arctan2}(\hat{\sin}, \hat{\cos})$ in degrees.
  - Decodes $\hat{\text{pitch}} = \text{clip}(\hat{p}_{\text{norm}} \times \sigma_{\text{pitch}} + \mu_{\text{pitch}}, -90^\circ, +90^\circ)$.
- **Audit Status:** **VERIFIED**.

### 2.4 Investigation of Test MAE Discrepancy (9.4828° vs 9.7088°)
A major question investigated in this audit was why `docs/PROJECT_PROGRESS_AND_TECHNICAL_SUMMARY.md` reported $9.7088^\circ$ Test MAE while other files cited $9.4828^\circ$.

#### Audit Findings:
1. **Direct Model Execution on Test Set:**
   Executing `saved_models/final_viewport_model.pth` directly on the 45,125 test sequences in `data/processed/lstm_test.npz` produces:
   - **Mean Spherical Angular MAE:** **$9.482777^\circ$ ($\approx 9.4828^\circ$)**
   - **Median Spherical Error:** **$6.856809^\circ$ ($\approx 6.8568^\circ$)**
   - **90th Percentile Error ($P_{90}$):** **$19.958485^\circ$ ($\approx 19.9585^\circ$)**
   - **Root Mean Square Error (RMSE):** **$13.218580^\circ$ ($\approx 13.2186^\circ$)**
   - **Yaw MAE:** **$8.179353^\circ$ ($\approx 8.1794^\circ$)**
   - **Pitch MAE:** **$3.645955^\circ$ ($\approx 3.6460^\circ$)**
2. **Matching Artifacts:**
   The exact values above match the generated CSV files:
   - [`results/metrics/final_model_evaluation_metrics.csv`](file:///d:/ml%20final%20project/results/metrics/final_model_evaluation_metrics.csv) (Line 16: `Test Spherical MAE,9.4828,degrees`)
   - [`results/predictions/final_model_test_predictions.csv`](file:///d:/ml%20final%20project/results/predictions/final_model_test_predictions.csv) (Mean of `spherical_error_deg` column = `9.482777°`)
   - [`docs/model_evaluation/FINAL_MODEL_EVALUATION_REPORT.md`](file:///d:/ml%20final%20project/docs/model_evaluation/FINAL_MODEL_EVALUATION_REPORT.md) (Line 31: `9.4828° MAE`)
3. **Origin of the $9.7088^\circ$ Discrepancy:**
   The value $9.7088^\circ$ does **NOT** exist in any model prediction CSV, training log, or evaluation file. It was accidentally introduced as a manual transcription typo in `docs/PROJECT_PROGRESS_AND_TECHNICAL_SUMMARY.md` during repository summarization, likely conflating the validation error scale (~$9.70^\circ$) or a single sample index with the test metric.
4. **Conclusion:**
   The **canonical, verified Test Spherical MAE is $9.4828^\circ$**.
- **Audit Status:** **VERIFIED (DISCREPANCY RESOLVED)**.

---

## 3. Video Conversion Audit

### 3.1 Media Stream Inspection
Stream properties for both assets were extracted using `ffprobe`:

| Video Property | Master Asset (`data/videos/11.mp4`) | Converted Stimulus (`data/videos/11_erp_30s.mp4`) | AVTrack360 Stimulus Spec (`11.json`) | Audit Status |
|:---|:---:|:---:|:---:|:---:|
| **Container** | MP4 (`isom`) | MP4 (`isom`) | AVI (`tmp\edgeofspace.avi`) | **VERIFIED** |
| **Video Codec** | VP9 (Profile 0) | H.264 / AVC (`libx264`, High 5.1) | Raw I420 | **VERIFIED** |
| **Pixel Format** | `yuv420p` | `yuv420p` | `yuv420p` | **VERIFIED** |
| **Resolution** | $3840 \times 2048$ | $3840 \times 1920$ | $3840 \times 1920$ | **VERIFIED** |
| **Aspect Ratio** | $15:8$ (EAC $3\times2$ packing) | $2:1$ (Canonical ERP) | $2:1$ (Canonical ERP) | **VERIFIED** |
| **Frame Rate** | $29.970\text{ fps}$ ($30000/1001$) | $29.970\text{ fps}$ ($30000/1001$) | $30.000\text{ fps}$ ($30/1$) | **WARNING** (0.1% drift) |
| **Duration** | $317.92\text{ s}$ ($5\text{ min } 18\text{ s}$) | $30.063\text{ s}$ | $30.033\text{ s}$ | **VERIFIED** |
| **Total Frames** | 9,528 frames | **901 frames** | **901 frames** | **VERIFIED** |
| **File Size** | 358.4 MB (375,811,868 bytes) | 32.88 MB (34,475,602 bytes) | 9.95 GB (uncompressed) | **VERIFIED** |

### 3.2 EAC to ERP Transformation Correctness
- **Filter Applied:** `v360=eac:equirect:w=3840:h=1920:interp=cube`
- **Output Geometry:**
  - $3840 \times 1920$ is the exact 2:1 aspect ratio required for spherical latitude/longitude mapping.
  - Horizontal seam wrap test (Column 0 vs Column 3839) on preview frames shows a mean pixel difference of $1.79 / 255$ (< 1% variance, consistent with bicubic filtering).
  - Vertical cube face seams show smooth gradient continuity ($1.81$–$2.81$ vs $0.94$ global baseline), with zero tearing or black lines.
  - Polar rows exhibit natural spherical convergence (zenith standard deviation: $3.09$; nadir payload bracket: $10.41$).
- **Audit Status:** **VERIFIED**.

### 3.3 Dataset Alignment Scope & Scientific Boundaries
- **Stimulus Window:** Extracted from $t = 10.0\text{ s}$ (`00:00:10`) to $t = 40.033\text{ s}$, yielding exactly 901 frames.
- **Visual Scene & Gaze Alignment:** Verified. In this segment, the Seeker VR crew prepares the telemetry payload beneath the deflated weather balloon on the asphalt lot. The $-19.45^\circ$ average downward pitch bias recorded across all 48 test participants aligns directly with the ground-level equipment located at $-20^\circ$ to $-45^\circ$ elevation.
- **Scientific Limitation (Temporal Sync):**
  - Claiming "frame-accurate synchronization ($\pm 1$ frame)" is **unverified and scientifically unsustainable** because the authors' 2018 raw master AVI file (`tmp\edgeofspace.avi`) is unavailable and YouTube re-encoding may shift initial GOP timestamps by several hundred milliseconds.
  - The frame rate difference ($29.970\text{ fps}$ vs $30.000\text{ fps}$) introduces an accumulated drift of approximately $1$ frame over 30 seconds ($30 \times 0.03\text{ s} \approx 0.03\text{ s}$).
  - **Verdict:** Alignment is verified at the **semantic and behavioral level (~within 0.5–1.0s)**.
- **Audit Status:** **VERIFIED WITH WARNING (Approximate Behavioral Alignment Only)**.

---

## 4. Tile Partitioner Audit

### 4.1 Grid Specifications & Boundary Checks
- **Source File:** [`src/tiling/grid_partitioner.py`](file:///d:/ml%20final%20project/src/tiling/grid_partitioner.py)
- **Grid Configuration:** 4 rows $\times$ 6 columns = 24 spatial tiles.
- **Frame Canvas:** $3840 \times 1920$ pixels.
- **Tile Dimensions:** Exactly $640 \times 480$ pixels per tile.
- **Angular Spans:** Exactly $60.0^\circ$ yaw $\times$ $45.0^\circ$ pitch per tile.
- **Pixel & Angular Bounds Table:**

| Row Index | Col Index | Tile IDs | Pixel Box $(X_{\min}, Y_{\min}) \to (X_{\max}, Y_{\max})$ | Yaw Range ($\lambda$) | Pitch Range ($\phi$) | Analytical Solid Angle | Grid Solid Angle |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Row 0 (Zenith)** | 0 to 5 | 0 – 5 | $(c \cdot 640, 0) \to ((c+1) \cdot 640, 480)$ | $[-180^\circ + c \cdot 60^\circ, -180^\circ + (c+1) \cdot 60^\circ]$ | $[+45.0^\circ, +90.0^\circ]$ | $0.306717\text{ sr}$ | $0.306717\text{ sr}$ |
| **Row 1 (Upper Eq)** | 0 to 5 | 6 – 11 | $(c \cdot 640, 480) \to ((c+1) \cdot 640, 960)$ | $[-180^\circ + c \cdot 60^\circ, -180^\circ + (c+1) \cdot 60^\circ]$ | $[0.0^\circ, +45.0^\circ]$ | $0.740480\text{ sr}$ | $0.740480\text{ sr}$ |
| **Row 2 (Lower Eq)** | 0 to 5 | 12 – 17 | $(c \cdot 640, 960) \to ((c+1) \cdot 640, 1440)$ | $[-180^\circ + c \cdot 60^\circ, -180^\circ + (c+1) \cdot 60^\circ]$ | $[-45.0^\circ, 0.0^\circ]$ | $0.740480\text{ sr}$ | $0.740480\text{ sr}$ |
| **Row 3 (Nadir)** | 0 to 5 | 18 – 23 | $(c \cdot 640, 1440) \to ((c+1) \cdot 640, 1920)$ | $[-180^\circ + c \cdot 60^\circ, -180^\circ + (c+1) \cdot 60^\circ]$ | $[-90.0^\circ, -45.0^\circ]$ | $0.306717\text{ sr}$ | $0.306717\text{ sr}$ |

### 4.2 Mathematical Solid Angle Verification
- **Analytical Formula:**
  $$\Omega = \Delta \lambda \times (\sin \phi_{\max} - \sin \phi_{\min})$$
  $$\Delta \lambda = \frac{360^\circ}{6} = \frac{\pi}{3}\text{ radians} \approx 1.04719755\text{ rad}$$
  - **Polar Cap (Rows 0 & 3):** $\sin(90^\circ) - \sin(45^\circ) = 1 - \frac{\sqrt{2}}{2} \approx 0.2928932 \implies \Omega_{\text{polar}} = 0.3067185\text{ sr}$.
  - **Equatorial Band (Rows 1 & 2):** $\sin(45^\circ) - \sin(0^\circ) = \frac{\sqrt{2}}{2} \approx 0.7071068 \implies \Omega_{\text{equat}} = 0.7404790\text{ sr}$.
- **Sum Over All 24 Tiles:**
  $$\sum_{i=0}^{23} \Omega_i = 6 \times 0.306717 + 6 \times 0.740480 + 6 \times 0.740480 + 6 \times 0.306717 = 12.566364\text{ sr}$$
  $$\text{Theoretical Sphere Area } 4\pi = 12.566371\text{ sr} \implies \text{Difference} = 0.000007\text{ sr} \ (< 0.0001\%)$$
- **Audit Status:** **VERIFIED**.

### 4.3 Coordinate Round-Trip & Boundary Wraparound
- **Coordinate Roundtrip Test:** 1,000 random points $(\lambda \in [-179.9^\circ, 179.9^\circ], \phi \in [-89.9^\circ, 89.9^\circ])$ were mapped to pixel space via `spherical_to_pixel()` and inverted via `pixel_to_spherical()`.
  - Max Yaw Inversion Error: **$0.00000000^\circ$**
  - Max Pitch Inversion Error: **$0.00000000^\circ$**
- **Circular Wraparound:** Tested on boundary tile 6 (Row 1, Column 0, $[-180^\circ, -120^\circ]$).
  - Neighbors returned: `[0, 7, 11, 12]`
  - Correctly links right neighbor (Tile 7, Col 1), top neighbor (Tile 0), bottom neighbor (Tile 12), and **left-wrapping neighbor across $\pm 180^\circ$ (Tile 11, Row 1, Col 5)**.
- **Audit Status:** **VERIFIED**.

---

## 5. Viewport Mapper Audit

### 5.1 3D Viewing Geometry & Ray Transformation
- **Source File:** [`src/tiling/viewport_mapper.py`](file:///d:/ml%20final%20project/src/tiling/viewport_mapper.py)
- **Frustum Model:** Pinhole camera frustum facing $+Z$ in camera-local space:
  $$u \in [-\tan(\text{HFoV}/2), +\tan(\text{HFoV}/2)], \quad v \in [-\tan(\text{VFoV}/2), +\tan(\text{VFoV}/2)], \quad w = 1$$
- **Rotation Matrix:**
  $$\mathbf{R} = \mathbf{R}_{\text{yaw}}(\lambda) \times \mathbf{R}_{\text{pitch}}(\phi)$$
  Rays are transformed to world coordinates $\mathbf{w} = \mathbf{r}_{\text{local}} \mathbf{R}^T$, then converted to spherical angles:
  $$\phi_{\text{ray}} = \arcsin(w_y), \quad \lambda_{\text{ray}} = \text{arctan2}(w_x, w_z)$$
- **Orientation Correctness Check:**
  - Viewport at $(0^\circ, 0^\circ)$ produces active tiles centered around Tile 15/8/9/14 (equatorial center).
  - Primary line-of-sight tile is always contained within the active tile set (tested across 50 diverse orientations).
  - Tile weights $w_i \in (0, 1]$ and strictly satisfy $\sum_{i} w_i = 1.0$ (within $10^{-5}$).
  - Margin tiles (1-ring neighbor expansion) are strictly disjoint from the active tile set.
- **Audit Status:** **VERIFIED**.

### 5.2 Methodological Nature & Limitations
1. **Ray-Sampling Approximation:**
   - The viewport-to-tile overlap is computed via **numerical ray sampling** ($21 \times 21 = 441$ rays or $25 \times 25 = 625$ rays), NOT via exact analytical spherical polygon clipping.
   - **Limitation:** Rays are distributed uniformly on the tangent viewing plane, creating slightly denser sampling near the optical center than at the extreme periphery. Very small boundary intersections (covering $< 0.1\%$ of the frustum) could theoretically be omitted if no ray intersects that corner.
2. **Clarification on Tile Weights:**
   - Tile weights $w_i$ represent **fractional visual ray coverage of tile $i$ within the frustum**, NOT calibrated probabilistic viewing likelihoods.
3. **100° x 100° Field of View Justification:**
   - The $100^\circ \times 100^\circ$ FoV assumption is consistent with the binocular viewing optics of consumer VR headsets (such as the HTC Vive used in AVTrack360, which has an approximate $110^\circ$ diagonal FoV).
   - However, FoV is a **configurable engineering parameter**, not an immutable dataset feature.
- **Audit Status:** **VERIFIED WITH WARNING (Numerical Discretization Approximation)**.

---

## 6. Tile Prediction Evaluation Audit

### 6.1 Critical Issue: Resolution of the Percentile Discrepancy
The previous report stated:
$$\text{Mean Viewport Coverage} = 98.00\%, \quad \text{90th Percentile Coverage} = 94.33\%$$

This discrepancy was audited against the raw distributions generated from all 4,513 evaluated test frames (stride=10 from `results/predictions/final_model_test_predictions.csv` across 8 test subjects and 20 videos).

#### Audit Findings:
In [`src/tiling/evaluate_tile_prediction.py`](file:///d:/ml%20final%20project/src/tiling/evaluate_tile_prediction.py) line 95:
```python
"Mean Viewport Coverage": np.mean(model_coverages) * 100.0,
"90th Percentile Coverage": np.percentile(model_coverages, 10) * 100.0,  # 10th percentile worst-case
```
1. **Root Cause:** The code called `np.percentile(model_coverages, 10)` — the **10th percentile** — but stored it under the column key `"90th Percentile Coverage"`.
2. **Context:** The author intended to express a **"90% coverage lower bound"** (the coverage value that 90% of frames exceed, i.e., $P(\text{coverage} \ge 94.33\%) = 0.90$).
3. **Statistical Fact:** In standard descriptive statistics, the 90th percentile of a distribution is the value below which 90% of observations fall.
   - **True 10th Percentile ($P_{10}$):** **$94.3311\%$** (the 90% reliability lower bound)
   - **True 50th Percentile (Median):** **$100.0000\%$**
   - **True 90th Percentile ($P_{90}$):** **$100.0000\%$**
   - **True Mean:** **$98.0023\%$**
   - **Frames with 100% Full Visual Coverage:** **$58.79\%$**
   - **Frames with $\ge 95\%$ Visual Coverage:** **$89.32\%$**
   - **Frames with $\ge 90\%$ Visual Coverage:** **$94.17\%$**
- **Audit Status:** **FAILED (LABELING BUG IN SCRIPT & REPORT)**.

### 6.2 Recalculated Metric Benchmark Table

| Strategy | Mean Tile Hit Ratio (Recall) | Mean Tile Precision | Mean Viewport Coverage | **10th Pct ($P_{10}$, 90% Bound)** | **True 90th Pct ($P_{90}$)** | Avg Tiles Fetched | Fetched Tile Area Ratio |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Persistence Baseline** | 91.93% | 92.21% | 97.82% | 92.74% | 100.00% | 8.87 / 24 | 36.97% |
| **Final GRU Model (Exact)** | **92.43%** | **92.81%** | **98.00%** | **94.33%** | **100.00%** | **8.87 / 24** | **36.97%** |
| **Final GRU Model (+1 Guard Band)** | **99.94%** | 45.92% | **99.97%** | **100.00%** | **100.00%** | **19.45 / 24** | **81.03%** |

### 6.3 Scientific Audit of the "36.97% Bandwidth" Claim
- **Previous Claim:** "Yielding ~63-70% theoretical bandwidth savings before quality adaptation."
- **Audit Finding:** The claim that fetching 8.87 out of 24 tiles equals $36.97\%$ bandwidth is **scientifically invalid**:
  1. $8.87 / 24 = 36.958\% \approx 36.97\%$ is strictly a **spatial tile-count / uncompressed pixel-area ratio**, NOT measured compressed video transmission bandwidth.
  2. Spatial video complexity is non-uniform across the spherical panorama (e.g., polar sky tiles compress to drastically fewer bits than equatorial action tiles).
- **Corrected Statement:** "The GRU-based strategy fetches an average of 8.87 out of 24 tiles, corresponding to approximately 36.97% of the full-frame uncompressed pixel area under equal-sized tile assumptions. This is a theoretical spatial-area ratio, not measured network bandwidth savings." Actual network bandwidth reduction can only be measured after video encoding and bitrate allocation simulation in Day 3.
- **Audit Status:** **WARNING (SCIENTIFIC OVERSTATEMENT)**.

---

## 7. Metric Recalculations & Data Sanity

### 7.1 Final Model Kinematic Benchmarks (Test Set: 45,125 Samples)
All metrics recalculated directly from `results/predictions/final_model_test_predictions.csv` and verified against model weights:

- **Spherical Mean Absolute Error (MAE):** $9.4828^\circ$
- **Spherical Median Error:** $6.8568^\circ$
- **Spherical 90th Percentile Error ($P_{90}$):** $19.9585^\circ$
- **Spherical Root Mean Square Error (RMSE):** $13.2186^\circ$
- **Circular Yaw MAE:** $8.1794^\circ$
- **Pitch MAE:** $3.6460^\circ$
- **Persistence Baseline Spherical MAE:** $13.9562^\circ$ (Improvement: $+32.05\%$)
- **Constant Angular Velocity MAE:** $11.5508^\circ$ (Improvement: $+17.90\%$)

### 7.2 Memory and Inference Footprint
- **Trainable Parameters:** 16,195 (GRU: 14,016; FC1: 2,080; FC2: 99).
- **Model Checkpoint Disk Footprint:** 66.9 KB.
- **Single-Sample Inference Latency:** $272.76\text{ }\mu\text{s}$ on CPU (enables up to 3,600 predictions per second, comfortably exceeding 10 Hz requirements).

---

## 8. Test Suite Audit

Executed command:
```powershell
py -3.14 -m unittest discover tests -v
```

### 8.1 Results Summary
- **Tests Discovered:** 13
- **Tests Passed:** 13
- **Tests Failed:** 0
- **Tests Skipped:** 0
- **Warnings:** 0
- **Execution Time:** $0.099\text{ seconds}$

### 8.2 Test Coverage Analysis
1. **`tests/test_causal_features.py` (7 tests):**
   - `test_boundary_conditions_and_shapes`: Empty arrays and single-frame inputs.
   - `test_circular_yaw_wraparound`: $\pm 180^\circ$ boundary crossing without derivative spikes.
   - `test_feature_order_and_encoding_integrity`: Verifies exact 7-column ordering.
   - `test_irregular_timestamps_and_zero_interval`: Handles $dt \le 0$ duplicates safely.
   - `test_prefix_invariance`: Prefixes produce identical features to full arrays.
   - `test_streaming_inference_equivalence`: Online streaming buffer matches offline batch.
   - `test_strict_causality_under_future_mutation`: **Zero leakage proof.**
2. **`tests/test_tiling.py` (6 tests):**
   - `test_grid_initialization`: 24 tiles, $640 \times 480$ bounds.
   - `test_solid_angle_sum`: Verifies $\sum \Omega_i = 4\pi\text{ sr}$.
   - `test_pixel_to_spherical_roundtrip`: Forward and inverse coordinate projection.
   - `test_circular_neighbors`: Longitude boundary wraparound from Col 0 to Col 5.
   - `test_intersection_weights_sum_to_one`: Viewing frustum ray weights sum to $1.0$.
   - `test_perfect_prediction_recall`: Ground truth prediction yields recall = 1.0.

### 8.3 Polar Boundary Edge Case Verification
To test unhandled edge cases, the viewport mapper was audited at extreme singular poles ($\text{pitch} = \pm 90.0^\circ$ and $\pm 89.99^\circ$):
- **Zenith ($+90.0^\circ$):** Correctly activates all 6 polar tiles in Row 0 plus upper equatorial tiles (Row 1), weights sum to $1.000000$, zero NaN/Inf errors.
- **Nadir ($-90.0^\circ$):** Correctly activates all 6 polar tiles in Row 3 plus lower equatorial tiles (Row 2), weights sum to $1.000000$, zero NaN/Inf errors.
- **Audit Status:** **VERIFIED**.

---

## 9. Reproducibility Audit

A clean environment execution check was performed across all public interfaces:

| Pipeline Step | Target Command | Execution Status | Output Artifact Generated |
|:---|:---|:---:|:---|
| **Module Imports** | `python -c "from src.feature_engineering import ...; from src.models import ...; from src.tiling import ..."` | **PASSED** | Exit code 0, all classes imported |
| **Model Inference** | `python -m src.models.final_model_inference` | **PASSED** | Input $(22^\circ, -1^\circ) \to (+0.5\text{s}): (36.20^\circ, 0.88^\circ)$ |
| **Grid Partitioner** | `python -m src.tiling.grid_partitioner` | **PASSED** | 24-tile geometry table printed |
| **Viewport Mapper** | `python -m src.tiling.viewport_mapper` | **PASSED** | Ray intersection weights printed |
| **Tile Evaluation** | `python -m src.tiling.evaluate_tile_prediction` | **PASSED** | `results/tiling/tile_prediction_metrics.csv` |
| **Unit Test Suite** | `python -m unittest discover tests` | **PASSED** | 13/13 tests OK in 0.099s |

All file paths in scripts are dynamically resolved using `Path(__file__).resolve().parents[...]`, ensuring path portability across development environments. No stale results are being misrepresented.

- **Audit Status:** **VERIFIED**.

---

## 10. Scientific Limitations

The following scientific limitations are explicitly declared:
1. **Ray-Sampling Discretization:** The viewport-to-tile overlap is computed via numerical ray sampling ($21 \times 21$ rays), not via exact analytical spherical polygon clipping.
2. **Approximate Temporal Synchronization:** The downloaded Video 11 segment ($[00:00:10, 00:00:40]$) matches the AVTrack360 stimulus content and gaze bias at a semantic level (~within 0.5–1.0s). Exact millisecond-level or frame-for-frame synchronization cannot be established without the authors' 2018 raw master AVI file.
3. **Motion-Only Viewport Model:** The final GRU model relies purely on past head motion kinematics ($L=10$ frames, 7 causal features). It does not incorporate visual saliency maps or optical flow features.
4. **Tile-Area vs. Bandwidth Distinction:** The spatial ratio ($8.87 / 24 = 36.97\%$) represents the raw pixel rendering fraction of the viewport, not compressed bitstream transmission bandwidth savings.

---

## 11. Bugs and Corrections Required

The following specific bugs and documentation inconsistencies were identified and must be corrected:

| # | Severity | File Involved | Identified Bug / Inconsistency | Required Correction |
|:---:|:---:|:---|:---|:---|
| 1 | **Medium** | `src/tiling/evaluate_tile_prediction.py` (Line 95) & `results/tiling/tile_prediction_metrics.csv` | Labeled `np.percentile(model_coverages, 10)` as `"90th Percentile Coverage"`. | Relabel to `"10th Percentile (90% Coverage Lower Bound)"` and add the true 90th percentile column ($100.00\%$). |
| 2 | **Medium** | `docs/PROJECT_PROGRESS_AND_TECHNICAL_SUMMARY.md` (Lines 120, 147, 182, 184) | Stated that the Final Retrained GRU Model achieved $9.7088^\circ$ Test MAE. | Correct the text to state the canonical ground-truth Test MAE: **$9.4828^\circ$** ($9.6698^\circ$ Validation MAE). |
| 3 | **Low** | `results/metrics/final_model_evaluation_metrics.csv` (Line 12) | Unquoted comma inside description string caused pandas parsing error: `Loss Function,MSELoss,-,Mean Squared Error on [sin(yaw), cos(yaw), norm_pitch]`. | Wrap description in quotation marks: `"Mean Squared Error on [sin(yaw), cos(yaw), norm_pitch]"`. |
| 4 | **Low** | `README.md` & `PROJECT_PROGRESS_AND_TECHNICAL_SUMMARY.md` | Claimed ~63–70% bandwidth savings based solely on the 8.87/24 tile ratio. | Clarify that $36.97\%$ is an uncompressed pixel area ratio, reserving bandwidth savings claims for Day 3 simulation traces. |

---

## 12. Final Readiness Verdict

### Verdict: **B. READY AFTER CORRECTIONS**

### Justification:
- **Why Not C (Not Ready):** The core implementation is technically sound. The dataset splits are strictly disjoint, feature engineering is causally clean, the PyTorch GRU checkpoint executes inference accurately ($9.4828^\circ$ Test MAE), the 30-second ERP stimulus video is verified at 901 frames, the $4 \times 6$ tile grid is mathematically consistent ($\sum \Omega_i = 4\pi$), and all 13 unit tests pass.
- **Why Not A (Ready As-Is):** The documentation contains a lingering transcription discrepancy ($9.7088^\circ$ vs $9.4828^\circ$), a misleading statistical label in `evaluate_tile_prediction.py` (10th percentile labeled as 90th percentile), an unquoted CSV parsing bug, and an overstatement regarding bandwidth savings.
- **Action Plan:** Once these documentation and labeling corrections are applied, Day 1 is 100% verified and the project is fully cleared to begin **Day 2 (Adaptive Quality Allocation Optimization)**.

