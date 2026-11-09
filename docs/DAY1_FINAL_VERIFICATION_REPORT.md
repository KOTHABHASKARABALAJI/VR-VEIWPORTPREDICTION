# Day 1 Final Technical Verification Report

**Project:** Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming  
**Phase:** Phase 5, Day 1 — Final System Verification & Acceptance  
**Date:** November 2026  
**Status:** FULLY VERIFIED, CONTRADICTIONS RESOLVED, 100% TEST PASS RATE  

---

## 1. Executive Summary & Verification Purpose

This report delivers the authoritative, evidence-based verification of the Day 1 viewport prediction and tile partitioning codebase. All contradictions identified in previous documentation drafts—including the test count discrepancy (reported 23 vs 33 in table vs 40 actual discovered tests), metric discrepancies between older uncleaned dataset runs (9.4828°) and retrained clean runs (9.4454°), and online streaming feature parity—have been investigated, root-caused, resolved, and empirically verified.

The codebase satisfies all acceptance criteria for transition to Day 2 (Adaptive Quality Allocation).

---

## 2. Test Discovery & Execution Audit (Resolving Test Count Contradiction)

### 2.1 Root Cause of Previous Contradiction
Previous reports contained an apparent contradiction where text stated 23 tests while an embedded table listed 33 test names. Investigation reveals:
1. During an earlier test run, an import error in `test_dataset_generation.py` caused discovery to halt partway, executing only 22 passing tests.
2. The author subsequently drafted a table with 33 proposed/historical test cases without running a complete, clean `unittest discover`.
3. When the complete test suite is discovered across the 5 active test modules in `tests/`, there are **exactly 40 distinct unit test methods**.

### 2.2 Actual Discovered Tests & Terminal Verification
Running `py -3.14 -m unittest discover -s tests -v`:
- **Total Discovered Tests:** 40
- **Passed:** 40
- **Failed:** 0
- **Skipped:** 0
- **Errors:** 0
- **Pass Rate:** **100.0%**
- **Execution Time:** ~3.3 seconds

### 2.3 Detailed Module Breakdown

| Test Module File | Test Class | Discovered Test Methods | Result |
| :--- | :--- | :---: | :---: |
| [`tests/test_causal_features.py`](file:///d:/ml%20final%20project/tests/test_causal_features.py) | `TestCausalFeatures` | 10 | **10 / 10 PASS** |
| [`tests/test_dataset_generation.py`](file:///d:/ml%20final%20project/tests/test_dataset_generation.py) | `TestDatasetGeneration` | 4 | **4 / 4 PASS** |
| [`tests/test_inference.py`](file:///d:/ml%20final%20project/tests/test_inference.py) | `TestFinalModelInference` | 5 | **5 / 5 PASS** |
| [`tests/test_tile_evaluation.py`](file:///d:/ml%20final%20project/tests/test_tile_evaluation.py) | `TestTileEvaluation` | 11 | **11 / 11 PASS** |
| [`tests/test_tiling.py`](file:///d:/ml%20final%20project/tests/test_tiling.py) | `TestTileGridPartitioner` (6) & `TestViewportMapper` (4) | 10 | **10 / 10 PASS** |
| **Total Test Suite** | **6 Test Classes across 5 Files** | **40** | **40 / 40 PASS (100%)** |

All 40 test cases execute and verify real properties: strict causality under future frame perturbation, causal boundary acceleration ($a_0 = 0, a_1 = 0$), duplicate calibration timestamp deduplication, disjoint 33/7/8 subject partitioning, streaming vs offline feature parity, ray weight sum to 1.0, and solid angles summing to $4\pi$ steradians.

---

## 3. Authoritative Model Metrics & Checkpoint Lineage

### 3.1 Resolving Model Metric Discrepancies
Previous project documents referenced three slightly differing values:
- Older verified test MAE: **9.4828°**
- Retrained clean GRU-64: **9.4454°** (rounded to 9.45°)
- Model C (GRU-64 Baseline): **9.5474°** (rounded to 9.55°)
- Model A (LSTM-64 Baseline): **9.5304°** (rounded to 9.53°)

#### Physical & Mathematical Cause:
1. **The 9.4828° result** was computed on an older version of `lstm_test.npz` containing **45,125 samples**. That dataset still included the 34 sensor-initialization calibration duplicates at $t = 0.0\text{s}$ (specifically, 4 duplicates were in test subject trajectories: subjects 9, 20, and 22).
2. **The 9.4454° result** is the true reproducible metric of the retrained model on the cleaned test set `lstm_test.npz` containing **45,121 samples**. With the unphysical calibration duplicates removed, test spherical MAE improved from 9.4828° to **9.4454°**.
3. **Model A (LSTM-64)** and **Model C (GRU-64)** are earlier baseline checkpoints (`viewport_lstm_model.pth` and `best_viewport_model.pth`). When evaluated on the exact same 45,121 test samples, they achieve **9.5304°** and **9.5474°** respectively.
4. **No Test Data Was Used for Model Selection:** Early stopping in `train_final_model.py` was monitored strictly on the validation set (`lstm_validation.npz`, 39,496 samples from 7 held-out validation subjects). Best validation loss (0.052574) occurred at **Epoch 18**. The test set was evaluated strictly once after training concluded.

### 3.2 Canonical Final Checkpoint Identity & Hashes

- **Primary Checkpoint File:** [`saved_models/final_viewport_model.pth`](file:///d:/ml%20final%20project/saved_models/final_viewport_model.pth)
- **Model Architecture:** `ConfigurableGRU` (1 GRU layer, 64 hidden units, 1 linear projection $64 \to 32$, output head $32 \to 3$, dropout = 0.0)
- **Trainable Parameters:** 16,195
- **File Size:** 68,547 bytes (66.9 KB)
- **SHA-256 Checksum:** `45a7dbec9063411cb80074130d6391fccc7036f4fe34164b93954326c1356c38`
- **Configuration File:** [`saved_models/final_model_config.json`](file:///d:/ml%20final%20project/saved_models/final_model_config.json)
- **Test Predictions CSV:** [`results/predictions/final_model_test_predictions.csv`](file:///d:/ml%20final%20project/results/predictions/final_model_test_predictions.csv) (45,121 rows, 14 columns)

### 3.3 Authoritative Test Set Benchmark (45,121 Samples, 8 Unseen Subjects)

| Model / Strategy | Method Type | Params | Spherical MAE | Spherical Median | Spherical P90 | Spherical RMSE | Yaw MAE | Pitch MAE | Error Reduction vs Persistence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline (Zero-Order Hold)** | Heuristic | 0 | 13.9567° | 10.2489° | 30.7285° | 19.3567° | 12.6233° | 4.3373° | Baseline (0.00%) |
| **Constant Velocity Baseline** | Kinematic | 0 | 11.5562° | 8.1762° | 24.9077° | 16.2863° | 9.6968° | 4.8763° | +17.20% |
| **Model A — LSTM-64 Baseline** | Deep RNN | 20,835 | 9.5304° | 6.8771° | 20.0353° | 13.2655° | 8.2524° | 3.6508° | +31.71% |
| **Model C — GRU-64 Baseline** | Deep RNN | 16,195 | 9.5474° | 6.8809° | 20.0800° | 13.3048° | 8.2484° | 3.6830° | +31.59% |
| **FINAL OPTIMAL GRU-64** | Deep RNN | **16,195** | **9.4454°** | **6.8054°** | **19.9330°** | **13.1949°** | **8.1654°** | **3.6428°** | **+32.32%** |

---

## 4. Dataset & Split Verification

### 4.1 Telemetry Cleaning & Deduplication Audit
- **Raw Telemetry Rows:** 284,247
- **Cleaned Telemetry Rows:** 284,213
- **Deduplicated Records:** Exactly 34 duplicate timestamp rows ($dt = 0.0\text{s}$) at $t = 0.0\text{s}$ calibration settled orientation retained.
- **Sampling Interval Bounds (Cleaned):**
  - $\Delta t_{\min} = 0.0020\text{ s}$ (all $\Delta t > 0$ strictly positive)
  - $\Delta t_{\max} = 0.4330\text{ s}$
  - $\Delta t_{\text{median}} = 0.1010\text{ s}$
  - $\Delta t_{\text{mean}} = 0.10091\text{ s}$
  - Nominal Frequency: $9.91\text{ Hz}$

### 4.2 Split Disjointness & Sample Counts
- **Total Subjects:** 48
- **Train (33 subjects):** 186,156 samples (`[2, 3, 5, 6, 7, 8, 10, 11, 13, 14, 15, 17, 18, 19, 21, 23, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 39, 41, 42, 45, 46, 47, 48]`)
- **Validation (7 subjects):** 39,496 samples (`[4, 12, 24, 25, 37, 38, 43]`)
- **Test (8 subjects):** 45,121 samples (`[9, 16, 20, 22, 30, 40, 44, 49]`)
- **Subject Leakage:** $\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$. (0.0% leakage).
- **Target Indexing Formula:** Verified across all 270,773 samples: $\text{target\_index} = \text{window\_start\_index} + 14$ ($L=10, k=5$).
- **Normalization:** $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$ fitted strictly on the 186,156 training samples.

---

## 5. Streaming & Feature Parity Verification

In [`src/models/final_model_inference.py`](file:///d:/ml%20final%20project/src/models/final_model_inference.py), the online streaming predictor maintains causal state (`prev_yaw`, `prev_pitch`, `prev_ts`, `prev_yaw_vel`, `prev_pitch_vel`, `streaming_step`) and a sliding feature window:
- Step $t=0$: $v_0 = 0.0, a_0 = 0.0$
- Step $t=1$: $v_1 = \Delta \theta / \Delta t, a_1 = 0.0$
- Step $t \ge 2$: $v_t = \Delta \theta / \Delta t, a_t = (v_t - v_{t-1}) / \Delta t$

**Empirical Parity Verification:**
Evaluating a 25-frame incoming stream through `push_and_predict_streaming` vs batch extraction across the entire sequence via `extract_7_features`:
- Max Yaw Difference: **0.000000°**
- Max Pitch Difference: **0.000000°**
- Online streaming feature calculation is **bit-for-bit identical** to offline batch processing.

---

## 6. Viewport Tile Evaluation & Area Efficiency

Evaluated on $3840 \times 1920$ ERP canvas, $4\times 6$ grid (24 tiles, $640\times 480$ px each), $100^\circ \times 100^\circ$ FoV, $21\times 21$ rays across 9,025 test instances (8 unseen subjects):

| Prediction Strategy | Mean Tile Hit Ratio (Recall) | Mean Tile Precision | Mean Viewport Coverage | P10 Viewport Coverage | P90 Viewport Coverage | Avg Tiles Fetched (out of 24) | Tile Area Ratio (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline** | 89.66% | 90.26% | 95.78% | 85.49% | 100.00% | 8.87 | 36.96% |
| **Final GRU (No Guard Band)** | 92.42% | 92.98% | **97.93%** | **94.33%** | 100.00% | 8.87 | 36.97% |
| **Final GRU (+1 Guard Band Margin)** | **99.96%** | 45.99% | **99.99%** | **100.00%** | 100.00% | 19.47 | 81.11% |

### Clarifications on Terminology:
- **Sampled Viewport Coverage:** Percentage of discrete ray intersections landing inside fetched tiles.
- **P10 & P90 Percentiles:** Computed empirically from the 9,025 evaluation sample distributions using `np.percentile`.
- **Tile Area Ratio:** Ratio of spatial tiles fetched to total canvas tiles ($8.87 / 24 \approx 36.97\%$). It represents uncompressed spatial area fetched on the ERP canvas; it does **not** equal compressed network bitrate savings or guarantee zero streaming stalls.

---

## 7. Reproducibility Findings

Controlled training runs were conducted under identical seed (42) using Python 3.14.2, PyTorch 2.14.0+cpu, and deterministic algorithms enabled:
1. Two consecutive training runs initialized with seed 42 yielded **exact identical epoch losses** (`[8.6966, 6.7412]`) and **max weight difference of 0.0**.
2. **Environment Caveat:** `os.environ["PYTHONHASHSEED"]` is set inside `set_seed()`; however, in Python, hash randomization is fixed at process start. For full cross-process determinism, `PYTHONHASHSEED=42` must be set in the shell before Python launches.
3. Claims of "100% determinism" have been moderated to reflect standard PyTorch CPU deterministic boundaries under fixed thread scheduling.

---

## 8. Known Limitations & Acceptance Verdict

### Known Limitations:
1. **Polar Distortion:** Under equirectangular projection, coordinate singularities at $\text{pitch} = \pm 90^\circ$ activate all 6 longitudinal tiles in row 0 or row 3.
2. **Single-Horizon Training:** The model is optimized specifically for $k=5$ (~0.5s lead time). Extrapolations beyond 1.0s incur multi-step divergence.
3. **Bandwidth Decoupling:** Fetching 36.97% of ERP tiles does not directly translate to a 63% bitrate saving due to inter-frame video compression overheads.

### Final Acceptance Verdict:
**DAY 1 IS OFFICIALLY COMPLETE, VERIFIED, AND ACCEPTED.**  
The pipeline is verified ready for Day 2: Adaptive Quality Allocation.

