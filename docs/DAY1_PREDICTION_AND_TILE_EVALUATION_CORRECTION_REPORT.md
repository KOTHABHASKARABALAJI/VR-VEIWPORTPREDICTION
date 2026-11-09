# Day 1 Prediction CSV and Tile Evaluation Correction Report

**Project Title:** Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming  
**Audit Phase:** Day 1 Critical Corrections (Technical Audit Follow-Up)  
**Date:** November 8, 2026  
**Status:** **PASSED & FULLY VERIFIED**  

---

## 1. Executive Summary

Following the comprehensive Day 1 technical audit, rigorous corrections were implemented to resolve identified defects in the test prediction generation and spatial tile evaluation pipelines:
1. **Elimination of Heuristic Persistence Approximation:** Replaced the heuristic angle offset calculation (`actual_yaw + persistence_sph_err_deg * 0.7`, `actual_pitch + persistence_sph_err_deg * 0.3`) with true, exact persistence coordinates (`persistence_predicted_yaw_deg`, `persistence_predicted_pitch_deg`) derived from the last observed orientation in each trajectory window.
2. **Decoupling of Persistence Tile Counts:** Eliminated the reuse of GRU tile count lists for the Persistence Baseline. Implemented and populated an independent list (`persist_active_counts`), calculating actual average fetched tiles and tile area ratio.
3. **Scientifically Accurate Strategy Naming:** Renamed `Final GRU Model (Exact)` to `Final GRU Model (No Guard Band)` to accurately reflect the numerical nature of ray-casting approximations (21×21 = 441 rays).
4. **Comprehensive Input Validation:** Implemented `validate_predictions_df` enforcing file existence, required schema, non-empty data, finite values, positive integer stride, valid FoV $(0^\circ, 180^\circ]$, and valid spherical angle domains ($[-90^\circ, 90^\circ]$ pitch, $[-360^\circ, 360^\circ]$ yaw).
5. **Granular Subject-Wise Reporting:** Generated `results/tiling/tile_prediction_metrics_by_subject.csv` across all 8 disjoint unseen test subjects ($N=8$: subjects 9, 16, 20, 22, 30, 40, 44, 49), proving subject-independent generalization.
6. **Dynamic Reporting & Automated Testing:** Rewrote `results/tiling/tile_prediction_summary.txt` to dynamically derive all numbers directly from the calculated summary DataFrame. Added 10 automated unit test cases in `tests/test_tile_evaluation.py`; all 23 tests in the full test suite pass cleanly in ~3.0s.

---

## 2. Files Modified and Bugs Fixed

| File Path | Nature of Changes / Bug Fix |
| :--- | :--- |
| `src/models/train_final_model.py` | Added temporal identifiers (`sample_index`, `trajectory_id`, `target_timestamp`), and exact persistence predictions (`persistence_predicted_yaw_deg`, `persistence_predicted_pitch_deg`) to the prediction generation routine (lines 315–365). |
| `results/predictions/final_model_test_predictions.csv` | Regenerated with all 14 required columns using canonical checkpoint `saved_models/final_viewport_model.pth`. Zero missing or fabricated values. |
| `src/tiling/evaluate_tile_prediction.py` | 1) Implemented `validate_predictions_df()`.<br>2) Replaced heuristic persistence approximation with exact coordinates.<br>3) Created independent `persist_active_counts`.<br>4) Renamed strategy to `Final GRU Model (No Guard Band)`.<br>5) Added subject-wise metrics compilation exporting `tile_prediction_metrics_by_subject.csv`.<br>6) Generated fully dynamic text report. |
| `tests/test_tile_evaluation.py` | Implemented 10 unit test cases validating coordinate preservation, mathematical independence of tile counts, identical sample selection, input rejection (stride, columns, NaNs), percentile validity, subject presence, and guard band separation. |
| `docs/PROJECT_PROGRESS_AND_TECHNICAL_SUMMARY.md` | Synchronized verified metrics (97.98% mean coverage, 94.33% P10, 36.95% spatial area ratio). |
| `docs/project_overview/PROJECT_PROGRESS_AND_TECHNICAL_SUMMARY.md` | Synchronized verified metrics with current canonical evaluations. |

---

## 3. Changes to the Prediction CSV

The test prediction file (`results/predictions/final_model_test_predictions.csv`) was updated from 10 columns to 14 columns across all 45,125 test samples:

| Column Name | Data Type | Verification / Description |
| :--- | :--- | :--- |
| `sample_index` | `int64` | Monotonically increasing index $[0 \dots 45,124]$. |
| `subject_id` | `int64` | Disjoint test subject identifier (8 subjects: 9, 16, 20, 22, 30, 40, 44, 49). |
| `video` | `str` | Stimulus video file name (`1.mp4` to `20.mp4`). |
| `trajectory_id` | `str` | Trajectory identifier format `sub_{id}_vid_{name}` (160 unique trajectories). |
| `target_timestamp` | `float64` | Ground-truth stimulus timestamp at prediction horizon ($t+k$, frame index $i+14$). |
| `actual_yaw_deg` | `float64` | Ground-truth yaw orientation in $[-180^\circ, 180^\circ]$. |
| `actual_pitch_deg` | `float64` | Ground-truth pitch orientation in $[-90^\circ, 90^\circ]$. |
| `predicted_yaw_deg` | `float64` | Final GRU model predicted yaw orientation. |
| `predicted_pitch_deg` | `float64` | Final GRU model predicted pitch orientation. |
| `spherical_error_deg` | `float64` | Great-circle distance between ground-truth and GRU prediction ($9.4828^\circ$ MAE). |
| `persistence_predicted_yaw_deg` | `float64` | Last observed orientation yaw at input sequence window end ($t=9$). |
| `persistence_predicted_pitch_deg` | `float64` | Last observed orientation pitch at input sequence window end ($t=9$). |
| `persistence_sph_err_deg` | `float64` | Great-circle distance between ground-truth and persistence ($13.9562^\circ$ MAE). |
| `const_vel_sph_err_deg` | `float64` | Constant velocity baseline error ($11.5508^\circ$ MAE). |

---

## 4. Evaluation Methodology

1. **Prediction Horizon & Alignment:**
   - Input Sequence: $L = 10$ frames ($1.0\,\text{s}$ context at $\sim 10\,\text{Hz}$).
   - Prediction Horizon: $k = 5$ frames ($\sim 0.505\,\text{s}$ future lead time).
   - Sample Stride: Stride = 5 yields $9,025$ evaluated samples across all 8 test subjects.
   - For every sample index $i$, all three strategies are evaluated on the exact same ground-truth viewport $(y_t^{\text{yaw}}, y_t^{\text{pitch}})$.
2. **Tile & Frustum Geometry:**
   - ERP Canvas: $3840 \times 1920$ pixels partitioned into $4 \text{ rows} \times 6 \text{ columns} = 24$ equal spatial tiles ($640 \times 480$ pixels each, $60^\circ$ yaw $\times 45^\circ$ pitch).
   - Viewport FoV: $100^\circ \text{ Horizontal} \times 100^\circ \text{ Vertical}$.
   - Frustum Ray Casting: $21 \times 21 = 441$ uniform rays projected into 3D Cartesian coordinates, rotated by the camera orientation $(y, p)$, and back-projected to ERP spherical coordinates to compute fractional ray intersections with tiles.
3. **Strategy Definitions:**
   - **Persistence Baseline:** Viewport centered at $(y_{t-k}^{\text{yaw}}, y_{t-k}^{\text{pitch}})$, no guard band.
   - **Final GRU Model (No Guard Band):** Viewport centered at $(\hat{y}_{t}^{\text{yaw}}, \hat{y}_{t}^{\text{pitch}})$, fetching only intersecting tiles.
   - **Final GRU Model (+1 Guard Band Margin):** Expands the predicted tile set with an immediate 1-ring 8-neighborhood of adjacent tiles (with horizontal wraparound).

---

## 5. Before vs. After Persistence Evaluation Results

| Metric | Previous Audit (Heuristic Approx. & Reused Counts) | Corrected Implementation (Exact Coordinates & Independent Counts) | Relative Delta / Note |
| :--- | :---: | :---: | :--- |
| **Persistence Coordinates Used** | `actual + sph_err * 0.7` | `persistence_predicted_yaw_deg` | Defect resolved |
| **Mean Tile Hit Ratio (Recall)** | 89.26% | **90.00%** | $+0.74\%$ |
| **Mean Tile Precision** | 89.26% | **90.01%** | $+0.75\%$ |
| **Mean Viewport Coverage** | 95.81% | **96.02%** | $+0.21\%$ |
| **10th Percentile Coverage (P10)** | 85.71% | **85.71%** | Identical |
| **90th Percentile Coverage (P90)** | 100.00% | **100.00%** | Identical |
| **Avg Tiles Fetched (out of 24)** | 8.87 tiles | **8.92 tiles** | Decoupled from GRU ($8.87 \to 8.92$) |
| **Tile Area Ratio (%)** | 36.97% | **37.16%** | Independent spatial calculation |

---

## 6. New Overall Evaluation Metrics

Evaluated on $9,025$ samples ($\text{stride}=5$ from $45,125$ test samples, 8 test subjects):

| Strategy | Tile Recall (%) | Tile Precision (%) | Mean Coverage (%) | P10 Coverage (%) | P90 Coverage (%) | Avg Tiles Fetched | Tile Area Ratio (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline** | 90.00% | 90.01% | 96.02% | 85.71% | 100.00% | 8.92 | 37.16% |
| **Final GRU Model (No Guard Band)** | **92.49%** | **93.05%** | **97.98%** | **94.33%** | **100.00%** | **8.87** | **36.95%** |
| **Final GRU Model (+1 Guard Band Margin)** | **99.94%** | 46.00% | **99.98%** | **100.00%** | **100.00%** | 19.44 | 81.01% |

### Key Observations:
1. **Prediction Gain over Baseline:** The Final GRU Model increases mean viewport visual coverage from $96.02\%$ to $97.98\%$ and elevates worst-decile performance (P10) from $85.71\%$ to $94.33\%$ ($+8.62\%$ absolute gain).
2. **Spatial Efficiency:** The Final GRU Model fetches fewer tiles on average ($8.87$ tiles, $36.95\%$ canvas area) than the Persistence Baseline ($8.92$ tiles, $37.16\%$ area), while delivering higher precision ($93.05\%$ vs. $90.01\%$).
3. **Guard Band Margin Protection:** Adding a 1-ring margin increases visual coverage to $99.98\%$ ($100.00\%$ at P10), providing protection against abrupt saccadic motions at the cost of fetching $19.44$ tiles ($81.01\%$ canvas area).

---

## 7. Granular Subject-Wise Results

Detailed breakdown across all 8 disjoint unseen test subjects (saved in `results/tiling/tile_prediction_metrics_by_subject.csv`):

| Subject | Strategy | Recall (%) | Precision (%) | Mean Cov. (%) | P10 Cov. (%) | P90 Cov. (%) | Avg Tiles | Area (%) | Samples |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **9** | Persistence Baseline | 92.57% | 92.63% | 97.18% | 89.80% | 100.00% | 8.81 | 36.72% | 1,128 |
| **9** | Final GRU (No Guard Band) | 94.35% | 95.00% | 98.76% | 96.76% | 100.00% | 8.77 | 36.52% | 1,128 |
| **9** | Final GRU (+1 Margin) | 99.99% | 46.39% | 100.00% | 100.00% | 100.00% | 19.07 | 79.44% | 1,128 |
| **16** | Persistence Baseline | 91.39% | 91.41% | 96.84% | 89.80% | 100.00% | 9.07 | 37.80% | 1,130 |
| **16** | Final GRU (No Guard Band) | 93.08% | 94.04% | 98.12% | 94.56% | 100.00% | 8.97 | 37.39% | 1,130 |
| **16** | Final GRU (+1 Margin) | 99.91% | 45.98% | 99.98% | 100.00% | 100.00% | 19.76 | 82.32% | 1,130 |
| **20** | Persistence Baseline | 89.23% | 89.25% | 95.74% | 85.71% | 100.00% | 8.98 | 37.43% | 1,129 |
| **20** | Final GRU (No Guard Band) | 91.76% | 92.26% | 97.55% | 92.93% | 100.00% | 8.93 | 37.23% | 1,129 |
| **20** | Final GRU (+1 Margin) | 99.88% | 45.90% | 99.91% | 100.00% | 100.00% | 19.60 | 81.66% | 1,129 |
| **22** | Persistence Baseline | 94.06% | 94.11% | 98.68% | 95.92% | 100.00% | 8.81 | 36.72% | 1,129 |
| **22** | Final GRU (No Guard Band) | 95.26% | 95.73% | 99.35% | 98.19% | 100.00% | 8.78 | 36.56% | 1,129 |
| **22** | Final GRU (+1 Margin) | 99.99% | 44.84% | 100.00% | 100.00% | 100.00% | 19.69 | 82.03% | 1,129 |
| **30** | Persistence Baseline | 86.51% | 86.45% | 93.38% | 76.19% | 100.00% | 8.85 | 36.89% | 1,128 |
| **30** | Final GRU (No Guard Band) | 90.17% | 90.03% | 96.63% | 88.66% | 100.00% | 8.87 | 36.95% | 1,128 |
| **30** | Final GRU (+1 Margin) | 99.90% | 45.89% | 99.97% | 100.00% | 100.00% | 19.36 | 80.66% | 1,128 |
| **40** | Persistence Baseline | 89.42% | 89.49% | 95.60% | 84.51% | 100.00% | 8.84 | 36.82% | 1,128 |
| **40** | Final GRU (No Guard Band) | 91.37% | 91.86% | 97.37% | 91.77% | 100.00% | 8.79 | 36.63% | 1,128 |
| **40** | Final GRU (+1 Margin) | 99.93% | 45.95% | 99.99% | 100.00% | 100.00% | 19.30 | 80.42% | 1,128 |
| **44** | Persistence Baseline | 87.24% | 87.23% | 94.63% | 81.63% | 100.00% | 8.99 | 37.44% | 1,127 |
| **44** | Final GRU (No Guard Band) | 91.04% | 91.77% | 97.65% | 92.97% | 100.00% | 8.91 | 37.12% | 1,127 |
| **44** | Final GRU (+1 Margin) | 99.98% | 46.38% | 100.00% | 100.00% | 100.00% | 19.44 | 80.99% | 1,127 |
| **49** | Persistence Baseline | 89.56% | 89.51% | 96.11% | 84.69% | 100.00% | 8.99 | 37.47% | 1,126 |
| **49** | Final GRU (No Guard Band) | 92.90% | 93.70% | 98.45% | 95.01% | 100.00% | 8.92 | 37.16% | 1,126 |
| **49** | Final GRU (+1 Margin) | 99.97% | 46.66% | 100.00% | 100.00% | 100.00% | 19.33 | 80.52% | 1,126 |

**Generalization Finding:** Across all 8 subjects without exception:
- The Final GRU Model achieves higher mean coverage and higher P10 coverage than Persistence.
- Even on the most erratic user trajectory (Subject 30, where Persistence drops to $76.19\%$ P10 coverage), the GRU maintains $88.66\%$ P10 coverage (an $+12.47\%$ absolute improvement).

---

## 8. Automated Test Suite Results

Full automated testing was executed via Python `unittest`:

```powershell
py -3.14 -m unittest discover tests -v
```

### Test Case Execution Log:
```text
test_boundary_conditions_and_shapes (test_causal_features.TestCausalFeatures) ... ok
test_circular_yaw_wraparound (test_causal_features.TestCausalFeatures) ... ok
test_feature_order_and_encoding_integrity (test_causal_features.TestCausalFeatures) ... ok
test_irregular_timestamps_and_zero_interval (test_causal_features.TestCausalFeatures) ... ok
test_prefix_invariance (test_causal_features.TestCausalFeatures) ... ok
test_streaming_inference_equivalence (test_causal_features.TestCausalFeatures) ... ok
test_strict_causality_under_future_mutation (test_causal_features.TestCausalFeatures) ... ok
test_1_persistence_coordinates_saved_correctly (test_tile_evaluation.TestTileEvaluation) ... ok
test_2_persistence_uses_saved_coordinates_not_spherical_approximation (test_tile_evaluation.TestTileEvaluation) ... ok
test_3_independent_tile_counts (test_tile_evaluation.TestTileEvaluation) ... ok
test_4_identical_evaluation_samples (test_tile_evaluation.TestTileEvaluation) ... ok
test_5_invalid_sample_stride_rejected (test_tile_evaluation.TestTileEvaluation) ... ok
test_6_missing_csv_columns_rejected (test_tile_evaluation.TestTileEvaluation) ... ok
test_7_nan_and_infinite_prediction_values_rejected (test_tile_evaluation.TestTileEvaluation) ... ok
test_8_p10_and_p90_percentile_calculations (test_tile_evaluation.TestTileEvaluation) ... ok
test_9_subject_wise_presence_all_test_subjects (test_tile_evaluation.TestTileEvaluation) ... ok
test_10_guard_band_and_no_guard_band_separated (test_tile_evaluation.TestTileEvaluation) ... ok
test_circular_neighbors (test_tiling.TestTileGridPartitioner) ... ok
test_grid_initialization (test_tiling.TestTileGridPartitioner) ... ok
test_pixel_to_spherical_roundtrip (test_tiling.TestTileGridPartitioner) ... ok
test_solid_angle_sum (test_tiling.TestTileGridPartitioner) ... ok
test_intersection_weights_sum_to_one (test_tiling.TestViewportMapper) ... ok
test_perfect_prediction_recall (test_tiling.TestViewportMapper) ... ok

----------------------------------------------------------------------
Ran 23 tests in 3.015s

OK
```

**Test Verdict:** 23 out of 23 unit tests PASSED (100% pass rate).

---

## 9. Checkpoint Preservation Confirmation

- Checkpoint Path: `saved_models/final_viewport_model.pth`
- File Size: `68,547 bytes` ($66.9\,\text{KB}$)
- File Modification Timestamp: Preserved unaltered (`1793811501.9774694`)
- Test Performance: $9.4828^\circ$ Spherical MAE, $6.8568^\circ$ Median, $19.9585^\circ$ P90, $13.2186^\circ$ RMSE.
- Parameter Count: 16,195 parameters.
- **Confirmation:** The model weights were not retrained, fine-tuned, or altered in any way.

---

## 10. Remaining Technical Limitations

1. **Spatial Area vs. Network Bandwidth:**
   - Tile Area Ratio represents the fraction of 24 ERP spatial tiles fetched ($36.95\%$ for GRU No Guard Band, $81.01\%$ with Guard Band Margin).
   - This ratio reflects uncompressed pixel area; actual network bitrate depends on video tile encoding efficiency (e.g., H.264 / HEVC intra/inter coding, motion vector variance across tiles). True bitrate savings will be measured in Phase 5 Day 2 and Day 3.
2. **Ray-Casting Discretization:**
   - Viewport intersections use $21 \times 21 = 441$ uniform rays. While providing high angular precision ($\sim 4.8^\circ$ ray spacing for a $100^\circ$ FoV), boundary rays near tile edges may exhibit slight discrete discretization effects.
3. **Guard Band Margin Overhead:**
   - While the 1-ring margin increases visual coverage to $99.98\%$ and eliminates black borders during sudden saccades, it fetches an average of $19.44$ tiles ($81.01\%$ canvas area), reducing precision from $93.05\%$ to $46.00\%$. Day 2's adaptive quality allocator will address this trade-off by allocating high bitrates to core viewport tiles and lower bitrates to margin tiles.

---

## 11. Final Day 1 Verification Sign-Off

- [x] Exact persistence coordinates (`persistence_predicted_yaw_deg`, `persistence_predicted_pitch_deg`) generated and verified.
- [x] Heuristic persistence approximation completely eliminated.
- [x] Separate `persist_active_counts` implemented and verified independent of GRU counts.
- [x] Renamed strategy to `Final GRU Model (No Guard Band)`.
- [x] Input validation (`validate_predictions_df`) implemented and covered by automated tests.
- [x] Subject-wise evaluation exported to `results/tiling/tile_prediction_metrics_by_subject.csv` across all 8 test subjects.
- [x] Dynamic report generation verified in `results/tiling/tile_prediction_summary.txt`.
- [x] 10 new test cases added; full 23-test suite passes cleanly.
- [x] Model checkpoint `final_viewport_model.pth` strictly preserved.

**Day 1 Status: FULLY COMPLETE & VERIFIED. Ready for Day 2 (Adaptive Quality Allocation).**

