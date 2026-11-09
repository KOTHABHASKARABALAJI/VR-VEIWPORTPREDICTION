# Day 1 Final Correction & Quality Assurance Report

**Project:** Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming  
**Date:** November 8, 2026  
**Auditor / Engineer:** Autonomous Senior Research Engineering & Quality Assurance Agent  
**Status:** All Corrections Successfully Implemented & Verified  
**Deliverable File:** `docs/DAY1_FINAL_CORRECTION_REPORT.md`  

---

## 1. Executive Summary

Following the independent technical audit documented in [`docs/DAY1_COMPLETE_TECHNICAL_AUDIT.md`](file:///d:/ml%20final%20project/docs/DAY1_COMPLETE_TECHNICAL_AUDIT.md), which rendered a conditional verdict of **READY AFTER CORRECTIONS**, all four identified defects were systematically corrected, re-evaluated against the raw data, and verified.

### Scope of Corrections Completed:
1. **Percentile Calculation & Label Correction:** Inverted statistical percentile labeling in `src/tiling/evaluate_tile_prediction.py` was corrected. The script now calculates and reports both the true 10th percentile ($P_{10} = 94.33\%$) and the true 90th percentile ($P_{90} = 100.00\%$), verified directly from 4,513 test samples.
2. **Canonical Model Metric Harmonization:** The erroneous $9.7088^\circ$ figure was purged from all active documentation. The canonical ground truth established across checkpoint execution, prediction logs, and evaluation CSVs is now consistently reported as **$9.4828^\circ$ Test Spherical MAE** and **$9.6698^\circ$ Validation Spherical MAE**.
3. **CSV Formatting Fix:** An unquoted comma in `results/metrics/final_model_evaluation_metrics.csv` (Line 12) was wrapped in quotation marks. The CSV now loads cleanly into pandas (26 rows $\times$ 4 columns).
4. **Bandwidth Terminology Alignment:** Overstated bandwidth savings claims were replaced with precise spatial-area wording in all documentation and summaries:
   > *"The GRU-based strategy fetches an average of 8.87 out of 24 tiles, corresponding to approximately 36.97% of the full-frame uncompressed pixel area under equal-sized tile assumptions. This is a theoretical spatial-area ratio, not measured network bandwidth savings."*
   The unsupported 10–25% encoding overhead claim was also removed.
5. **Full Pipeline Regression Validation:** All 13 unit tests passed in $0.103\text{ seconds}$ with zero failures, zero skipped tests, and zero warnings.

---

## 2. Detailed Audit Corrections

### Correction 1: Percentile Calculations in `src/tiling/evaluate_tile_prediction.py`
- **Files Modified:**
  - [`src/tiling/evaluate_tile_prediction.py`](file:///d:/ml%20final%20project/src/tiling/evaluate_tile_prediction.py) (Lines 111–178)
  - [`results/tiling/tile_prediction_metrics.csv`](file:///d:/ml%20final%20project/results/tiling/tile_prediction_metrics.csv)
  - [`results/tiling/tile_prediction_summary.txt`](file:///d:/ml%20final%20project/results/tiling/tile_prediction_summary.txt)
- **Defect Description:**
  The evaluation script called `np.percentile(model_coverages, 10)` (the 10th percentile, intended as a 90% coverage lower bound) but stored it under the column label `"90th Percentile Coverage"`. In standard statistics, the 90th percentile of this distribution is $100.00\%$, while the median is $100.00\%$, and the mean is $98.00\%$.
- **Correction Applied:**
  - Added dedicated columns for both `"10th Percentile Coverage (P10)"` and `"90th Percentile Coverage (P90)"`.
  - Renamed `"Tile Bandwidth Ratio"` to `"Tile Area Ratio (%)"`.
  - Re-ran the evaluation script over all 4,513 test frames (subsample stride 10 from 45,125 total test frames) to regenerate CSV and summary text directly from raw calculations.
- **Before-and-After Metric Values:**

| Strategy | Metric | Before Value | After Value (Corrected & Verified) |
|:---|:---|:---:|:---:|
| **Final GRU Model (Exact)** | Mean Viewport Coverage | 98.00% | **98.0023%** ($\approx 98.00\%$) |
| **Final GRU Model (Exact)** | 10th Percentile ($P_{10}$, 90% Bound) | *Misattributed as 90th Pct* | **94.3311%** ($\approx 94.33\%$) |
| **Final GRU Model (Exact)** | 90th Percentile ($P_{90}$) | *Unreported* | **100.0000%** ($100.00\%$) |
| **Final GRU Model (Exact)** | 50th Percentile (Median) | *Unreported* | **100.0000%** ($100.00\%$) |
| **Final GRU Model (Exact)** | Avg Tiles Fetched (out of 24) | 8.87 | **8.8721** ($\approx 8.87$) |
| **Final GRU Model (Exact)** | Tile / Pixel Area Ratio | 36.97% (*labeled as bandwidth*) | **36.97%** (*labeled as Tile Area Ratio*) |
| **GRU (+1 Guard Band)** | Mean Viewport Coverage | 99.97% | **99.9705%** ($\approx 99.97\%$) |
| **GRU (+1 Guard Band)** | 10th Percentile ($P_{10}$) | 100.00% (*labeled as 90th Pct*) | **100.0000%** ($100.00\%$) |
| **GRU (+1 Guard Band)** | 90th Percentile ($P_{90}$) | *Unreported* | **100.0000%** ($100.00\%$) |

---

### Correction 2: Canonical Model Metrics in Documentation
- **Files Modified:**
  - [`docs/PROJECT_PROGRESS_AND_TECHNICAL_SUMMARY.md`](file:///d:/ml%20final%20project/docs/PROJECT_PROGRESS_AND_TECHNICAL_SUMMARY.md) (Lines 116–122, 145–163, 175–185, 190–196)
  - [`docs/project_overview/PROJECT_PROGRESS_AND_TECHNICAL_SUMMARY.md`](file:///d:/ml%20final%20project/docs/project_overview/PROJECT_PROGRESS_AND_TECHNICAL_SUMMARY.md) (Lines 116–122, 145–163, 175–185, 190–196)
  - [`docs/REPOSITORY_CLEANUP_REPORT.md`](file:///d:/ml%20final%20project/docs/REPOSITORY_CLEANUP_REPORT.md) (Lines 106, 161)
- **Defect Description:**
  The figure $9.7088^\circ$ was erroneously documented as the Final Production Model's Test Spherical MAE in `PROJECT_PROGRESS_AND_TECHNICAL_SUMMARY.md`. In reality, direct execution of the saved model weights (`saved_models/final_viewport_model.pth`) on `data/processed/lstm_test.npz` yields exactly $9.4828^\circ$ Test MAE and $9.6698^\circ$ Validation MAE.
- **Correction Applied:**
  Purged all occurrences of $9.7088^\circ$ from active documentation and aligned all tables, multi-horizon scaling descriptions, and summary text to the verified canonical figures:
  - **Canonical Test Spherical MAE:** **$9.4828^\circ$**
  - **Canonical Validation Spherical MAE:** **$9.6698^\circ$**
  - **Test Median Great-Circle Error:** **$6.8568^\circ$**
  - **Test 90th Percentile Error ($P_{90}$):** **$19.9585^\circ$**
  - **Test Root Mean Square Error (RMSE):** **$13.2186^\circ$**
- **Verification Scan:** Automated search confirms **zero occurrences** of $9.7088$ remain in any active markdown documentation.

---

### Correction 3: CSV Formatting in `results/metrics/final_model_evaluation_metrics.csv`
- **Files Modified:**
  - [`results/metrics/final_model_evaluation_metrics.csv`](file:///d:/ml%20final%20project/results/metrics/final_model_evaluation_metrics.csv) (Line 12)
- **Defect Description:**
  Line 12 contained unquoted commas in the description field:
  `Loss Function,MSELoss,-,Mean Squared Error on [sin(yaw), cos(yaw), norm_pitch]`
  This caused standard CSV parsers (including `pandas.read_csv`) to raise `ParserError: Expected 4 fields in line 12, saw 6`.
- **Correction Applied:**
  Enclosed the description field in double quotes:
  `Loss Function,MSELoss,-,"Mean Squared Error on [sin(yaw), cos(yaw), norm_pitch]"`
- **Pandas Verification:**
  - File parses cleanly into a `pandas.DataFrame`.
  - Shape: exactly **26 rows $\times$ 4 columns**.
  - All columns (`Metric`, `Value`, `Unit`, `Description`) are preserved with zero data loss.

---

### Correction 4: Bandwidth Terminology Alignment
- **Files Modified:**
  - [`docs/PROJECT_PROGRESS_AND_TECHNICAL_SUMMARY.md`](file:///d:/ml%20final%20project/docs/PROJECT_PROGRESS_AND_TECHNICAL_SUMMARY.md)
  - [`docs/project_overview/PROJECT_PROGRESS_AND_TECHNICAL_SUMMARY.md`](file:///d:/ml%20final%20project/docs/project_overview/PROJECT_PROGRESS_AND_TECHNICAL_SUMMARY.md)
  - [`docs/DAY1_COMPLETE_TECHNICAL_AUDIT.md`](file:///d:/ml%20final%20project/docs/DAY1_COMPLETE_TECHNICAL_AUDIT.md)
  - [`results/tiling/tile_prediction_summary.txt`](file:///d:/ml%20final%20project/results/tiling/tile_prediction_summary.txt)
- **Defect Description:**
  Earlier reports characterized the $8.87 / 24 = 36.97\%$ tile ratio as "bandwidth savings of ~63%–70%". Furthermore, an unsupported claim of "10–25% encoding overhead" was cited without benchmark data.
- **Correction Applied:**
  - Replaced all claims of measured bandwidth reduction with the standardized scientific phrasing:
    > *"The GRU-based strategy fetches an average of 8.87 out of 24 tiles, corresponding to approximately 36.97% of the full-frame uncompressed pixel area under equal-sized tile assumptions. This is a theoretical spatial-area ratio, not measured network bandwidth savings."*
  - Removed the unsupported 10–25% encoding overhead claim.
  - Bandwidth consumption measurements are reserved exclusively for Day 3 simulation traces under realistic mobile network profiles.

---

## 3. Final Validation Results

### 3.1 Unit Test Suite Verification
Executed command:
```powershell
py -3.14 -m unittest discover tests -v
```
Output:
```text
test_boundary_conditions_and_shapes (test_causal_features.TestCausalFeatures.test_boundary_conditions_and_shapes) ... ok
test_circular_yaw_wraparound (test_causal_features.TestCausalFeatures.test_circular_yaw_wraparound) ... ok
test_feature_order_and_encoding_integrity (test_causal_features.TestCausalFeatures.test_feature_order_and_encoding_integrity) ... ok
test_irregular_timestamps_and_zero_interval (test_causal_features.TestCausalFeatures.test_irregular_timestamps_and_zero_interval) ... ok
test_prefix_invariance (test_causal_features.TestCausalFeatures.test_prefix_invariance) ... ok
test_streaming_inference_equivalence (test_causal_features.TestCausalFeatures.test_streaming_inference_equivalence) ... ok
test_strict_causality_under_future_mutation (test_causal_features.TestCausalFeatures.test_strict_causality_under_future_mutation) ... ok
test_circular_neighbors (test_tiling.TestTileGridPartitioner.test_circular_neighbors) ... ok
test_grid_initialization (test_tiling.TestTileGridPartitioner.test_grid_initialization) ... ok
test_pixel_to_spherical_roundtrip (test_tiling.TestTileGridPartitioner.test_pixel_to_spherical_roundtrip) ... ok
test_solid_angle_sum (test_tiling.TestTileGridPartitioner.test_solid_angle_sum) ... ok
test_intersection_weights_sum_to_one (test_tiling.TestViewportMapper.test_intersection_weights_sum_to_one) ... ok
test_perfect_prediction_recall (test_tiling.TestViewportMapper.test_perfect_prediction_recall) ... ok

----------------------------------------------------------------------
Ran 13 tests in 0.103s

OK
```
**Status:** **13 passed, 0 failed, 0 skipped, 0 warnings.**

### 3.2 CSV and Model Verification
Executed automated verification script:
- `results/metrics/final_model_evaluation_metrics.csv`: Loaded cleanly (26 rows $\times$ 4 columns). Verified `Test Spherical MAE = 9.4828`, `Validation Spherical MAE = 9.6698`.
- `results/tiling/tile_prediction_metrics.csv`: Loaded cleanly. Verified `Mean Viewport Coverage = 98.0023%`, `10th Percentile Coverage (P10) = 94.3311%`, `90th Percentile Coverage (P90) = 100.0000%`, `Tile Area Ratio = 36.97%`.
- Model inference (`src.models.final_model_inference`): Loaded `final_viewport_model.pth` and executed prediction cleanly without warnings.

---

## 4. Remaining Warnings and Scientific Limitations

The following scientific limitations remain documented and understood:
1. **Ray-Sampling Discretization:** Viewport-to-tile overlap is computed via numerical ray sampling ($21 \times 21$ viewing frustum grid), which serves as a discrete approximation rather than analytical spherical polygon clipping.
2. **Behavioral vs. Millisecond Video Synchronization:** The downloaded 30-second stimulus segment (`data/videos/11_erp_30s.mp4`) matches the AVTrack360 visual content and user gaze bias (empirical $-19.45^\circ$ downward pitch) at a semantic level (~within 0.5–1.0s). Exact frame-for-frame synchronization cannot be guaranteed without the authors' 2018 raw master AVI file.
3. **Bandwidth Savings Scope:** Day 1 metrics establish spatial rendering efficiency (fetching 8.87 / 24 tiles covers 36.97% of uncompressed canvas pixels). True network bitrate savings will be simulated and measured in Day 3.

---

## 5. Final Day 1 Readiness Verdict

### Verdict: **A. READY**

**Justification:**  
All four audit findings have been resolved and verified against the actual codebase and generated data. The dataset splits are disjoint, feature engineering is causally clean, model weights and inference APIs are verified ($9.4828^\circ$ Test MAE), the 30-second stimulus video matches the 901-frame specification, the 24-tile partitioner is mathematically exact ($\sum \Omega_i = 4\pi$), coverage percentile calculations and column labels are verified, and all 13 unit tests pass. 

The project repository is in a sound state and is cleared to proceed to **Day 2 of Phase 5 (Adaptive Quality Allocation Optimization)**.

