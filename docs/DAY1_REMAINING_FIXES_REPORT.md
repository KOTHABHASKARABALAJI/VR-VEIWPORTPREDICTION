# Day 1 Remaining Fixes Report

**Project:** Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming  
**Author:** ML Engineering & Audit Team  
**Date:** November 2026  
**Status:** RESOLVED & VERIFIED  

---

## 1. Discrepancy Record & Technical Resolutions

### Discrepancy 1: Test Count Contradiction (23 vs 33 vs 40 Discovered Tests)
- **Observed Issue:**  
  Previous documentation stated 23 unit tests passed, while its embedded summary table listed 33 test names. Actual test discovery finds 40 tests.
- **Root Cause:**  
  An unhandled import error (`create_sliding_windows`) in `test_dataset_generation.py` caused discovery to terminate prematurely during an earlier run, leaving only 22 passing tests. The author subsequently inserted a static 33-item markdown table without executing a full discovery run.
- **Files Modified:**  
  - [`src/dataset_generation/sequence_generator.py`](file:///d:/ml%20final%20project/src/dataset_generation/sequence_generator.py)
  - [`src/dataset_generation/__init__.py`](file:///d:/ml%20final%20project/src/dataset_generation/__init__.py)
  - [`tests/test_dataset_generation.py`](file:///d:/ml%20final%20project/tests/test_dataset_generation.py)
- **Actual Correction:**  
  1. Restored the backward-compatible `create_sliding_windows()` wrapper in `sequence_generator.py` and exported it alongside `create_sliding_windows_with_metadata()` in `__init__.py`.
  2. Adjusted float comparison precision (`places=4` for `np.float32` targets vs `np.float64` trajectory coordinates) in `test_dataset_generation.py`.
- **Verification Performed:**  
  Executed `py -3.14 -m unittest discover -s tests -v`. Exactly **40 tests** are discovered, and all **40 tests pass (100%)**.

---

### Discrepancy 2: Model Test MAE Metric Contradiction (9.4828° vs 9.4454°)
- **Observed Issue:**  
  The audit documentation previously established a test MAE of 9.4828°, while newly retrained runs reported 9.4454° (rounded to 9.45°).
- **Root Cause:**  
  The 9.4828° metric was computed on an older test split of 45,125 samples that contained 4 uncleaned sensor calibration duplicate frames at $t=0.0\text{s}$ (from subjects 9, 20, and 22). When the raw trajectories were cleaned and deduplicated, the test set sample count became 45,121. Retraining on the clean dataset produced an improved, verified test MAE of **9.4454°**.
- **Files Modified:**  
  - [`docs/PROJECT_PROGRESS_AND_TECHNICAL_SUMMARY.md`](file:///d:/ml%20final%20project/docs/PROJECT_PROGRESS_AND_TECHNICAL_SUMMARY.md)
  - [`docs/DAY1_FINAL_VALIDATION_REPORT.md`](file:///d:/ml%20final%20project/docs/DAY1_FINAL_VALIDATION_REPORT.md)
  - [`docs/DAY1_FINAL_VERIFICATION_REPORT.md`](file:///d:/ml%20final%20project/docs/DAY1_FINAL_VERIFICATION_REPORT.md)
- **Actual Correction:**  
  Documented the exact mathematical origin of the difference. All canonical documentation now references the retrained model metric on the clean 45,121-sample test set: **9.4454° Spherical MAE** (saving checkpoint SHA-256 `45a7dbec9063411cb80074130d6391fccc7036f4fe34164b93954326c1356c38`).
- **Verification Performed:**  
  Recalculated spherical MAE directly from [`results/predictions/final_model_test_predictions.csv`](file:///d:/ml%20final%20project/results/predictions/final_model_test_predictions.csv): `df['spherical_error_deg'].mean() == 9.445351°`.

---

### Discrepancy 3: Online Streaming vs Offline Batch Feature Drift
- **Observed Issue:**  
  `StreamingViewportPredictor` previously re-extracted features from a truncated 15-frame buffer, causing boundary conditions ($v_0=0, a_0=0$) to be applied to sliding window prefixes rather than accumulating continuous kinematic state.
- **Root Cause:**  
  Truncating raw coordinate buffers to 15 frames forced `extract_7_features()` to treat the 15-frame window as starting anew at index 0, introducing a ~0.02° to 0.03° feature drift.
- **Files Modified:**  
  - [`src/models/final_model_inference.py`](file:///d:/ml%20final%20project/src/models/final_model_inference.py)
- **Actual Correction:**  
  Re-engineered `push_and_predict_streaming()` to compute exact causal kinematic state recursively step-by-step ($v_t, a_t$) and push the resulting 7-feature vectors into a 10-step `feature_history` buffer.
- **Verification Performed:**  
  Ran a comparative trace over a 25-frame sequence comparing single-step streaming outputs against full offline batch extraction. Maximum yaw and pitch difference is **0.000000° (exact match)**.

---

### Discrepancy 4: Overstated Determinism and Bitrate Savings Claims
- **Observed Issue:**  
  Documentation previously asserted "100% deterministic" training and suggested tile area reduction directly represented compressed video bitrate savings.
- **Root Cause:**  
  1. `os.environ["PYTHONHASHSEED"]` was set inside `set_seed()`, which does not affect the already-initialized Python hash seed of the current process.
  2. Tile Area Ratio ($8.87 / 24 \approx 36.97\%$) represents uncompressed ERP spatial canvas area fetched, not compressed H.264/HEVC bitrate savings.
- **Files Modified:**  
  - [`docs/DAY1_FINAL_VALIDATION_REPORT.md`](file:///d:/ml%20final%20project/docs/DAY1_FINAL_VALIDATION_REPORT.md)
  - [`docs/DAY1_BUG_FIX_REPORT.md`](file:///d:/ml%20final%20project/docs/DAY1_BUG_FIX_REPORT.md)
- **Actual Correction:**  
  1. Clarified that process-level `PYTHONHASHSEED` must be set in the shell before Python launches.
  2. Added explicit disclaimers: Tile Area Ratio measures uncompressed canvas pixel area fetched and does not guarantee compressed network bitrate savings or zero streaming stalls.
- **Verification Performed:**  
  Verified that all documentation uses qualified, technically precise language.

---

## 2. Issues Intentionally Left Unresolved & Technical Justification

1. **Polar Convergence under ERP Projection:**  
   At $\text{pitch} \to \pm 90^\circ$, all 6 longitudinal tiles in rows 0 and 3 converge at the poles. This is an inherent geometric property of Equirectangular Projection, not a software bug. It is documented as a known physical constraint.
2. **Fixed 0.5-Second Horizon ($k=5$):**  
   The model is intentionally trained and optimized for a single 5-step horizon. Multi-horizon recursive rollout is out of scope for Day 1 and reserved for subsequent adaptive allocation stages.

---

## 3. Summary of Files Changed

1. [`src/dataset_generation/sequence_generator.py`](file:///d:/ml%20final%20project/src/dataset_generation/sequence_generator.py): Added `create_sliding_windows()` backward-compatibility wrapper; enforced fixed-width string dtypes (`<U32`, `<U64`) for metadata arrays.
2. [`src/dataset_generation/__init__.py`](file:///d:/ml%20final%20project/src/dataset_generation/__init__.py): Exported `create_sliding_windows` and `create_sliding_windows_with_metadata`.
3. [`src/models/final_model_inference.py`](file:///d:/ml%20final%20project/src/models/final_model_inference.py): Added step-by-step causal kinematic state tracking for zero-drift streaming parity.
4. [`tests/test_dataset_generation.py`](file:///d:/ml%20final%20project/tests/test_dataset_generation.py): Fixed float comparison precision for target indexing validation.
5. [`docs/DAY1_FINAL_VERIFICATION_REPORT.md`](file:///d:/ml%20final%20project/docs/DAY1_FINAL_VERIFICATION_REPORT.md): Authoritative verification report resolving all contradictions.
6. [`docs/DAY1_REMAINING_FIXES_REPORT.md`](file:///d:/ml%20final%20project/docs/DAY1_REMAINING_FIXES_REPORT.md): Detailed root cause and resolution report for all discrepancies.
7. [`docs/FINAL_EXPERIMENT_CONFIGURATION.md`](file:///d:/ml%20final%20project/docs/FINAL_EXPERIMENT_CONFIGURATION.md): Complete system specification and parameter archive.

