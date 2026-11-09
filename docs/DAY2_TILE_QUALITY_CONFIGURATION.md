# Day 2 — Module 1: Tile Quality and Bitrate Configuration

**Project:** Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming  
**Phase:** Phase 5, Day 2 — Adaptive Quality Allocation  
**Module:** Module 1 — Tile Quality, Bitrates, and Utility Configuration  
**Date:** November 2026  
**Status:** IMPLEMENTED & TESTED (11 / 11 Module Tests Passed, 51 / 51 Repository Tests Passed)  

---

## 1. System Overview & Purpose

Module 1 establishes the foundational rate-utility and configuration abstractions for Day 2 of the adaptive 360° video streaming system. It models discrete quality tiers, simulated bitrate costs, numerical perceptual utility, and per-tile quality assignments across the canonical **4 rows $\times$ 6 columns (24 tiles)** Equirectangular Projection (ERP) video canvas.

This module provides the core data structures and validation constraints required by subsequent adaptive allocation algorithms:
1. **Uniform Quality Allocation** (Benchmark baseline).
2. **Viewport-Aware Quality Allocation** (Assigning higher quality to tiles in the predicted viewport).
3. **Neighbourhood-Aware Quality Allocation** (Applying multi-tier guard bands around predicted focus regions).

---

## 2. Simulated Quality Tiers & Bitrate Assumptions

### 2.1 Discrete Quality Levels
Four discrete quality levels are defined in `QualityLevel`:

| Quality Tier | Code | Tier Name | Simulated Bitrate per Tile ($R_k$) | Bitrate Ratio vs Q1 | Primary Simulated Role |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Q1** | `QualityLevel.Q1` | Low | **100 kbps** | $1.0\times$ | Low-overhead background fallback for non-visible tiles. |
| **Q2** | `QualityLevel.Q2` | Medium | **300 kbps** | $3.0\times$ | Standard delivery for peripheral tiles and outer margins. |
| **Q3** | `QualityLevel.Q3` | High | **600 kbps** | $6.0\times$ | High-fidelity delivery for active viewport boundaries. |
| **Q4** | `QualityLevel.Q4` | Ultra | **1000 kbps** | $10.0\times$ | Maximum quality allocation for primary line-of-sight focal tiles. |

### 2.2 Simulation Assumption Disclaimer
> [!IMPORTANT]
> The bitrates above ($100$, $300$, $600$, and $1000\text{ kbps}$) are **simulated transmission assumptions** representing standard tile rate envelopes in adaptive streaming literature. They are **not** empirically measured video encoding bitrates, and the system does not transcode or store 24 separate video files.

---

## 3. Numerical Quality Utility Formulation

### 3.1 Illustrative Simulation Assumptions (Diminishing Marginal Returns)
In video streaming Quality of Experience (QoE) simulation, visual satisfaction is modeled with diminishing marginal utility increments as bitrate increases. The default numerical quality utility $U(q)$ is configured with the illustrative values:

| Quality Tier | Bitrate ($R_k$) | Utility ($U_k$) | Incremental Bitrate ($\Delta R$) | Incremental Utility ($\Delta U$) | Marginal Utility Efficiency ($\Delta U / \Delta R$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Q1 (Low)** | 100 kbps | **1.00** | — | — | — |
| **Q2 (Medium)** | 300 kbps | **2.00** | +200 kbps | +1.00 | $0.00500\text{ util/kbps}$ |
| **Q3 (High)** | 600 kbps | **2.80** | +300 kbps | +0.80 | $0.00267\text{ util/kbps}$ |
| **Q4 (Ultra)** | 1000 kbps | **3.40** | +400 kbps | +0.60 | $0.00150\text{ util/kbps}$ |

### 3.2 Scientific Scope & Disclaimers
> [!NOTE]
> 1. **Illustrative Modeling Assumptions:** The default values (1.00, 2.00, 2.80, 3.40) are **illustrative, configurable simulation assumptions** designed to model diminishing returns. They are **not** empirically calibrated constants derived from the Weber-Fechner law or ITU-T Rec. P.1203.
> 2. **Unitless Abstractions:** These numerical quality utilities are unitless simulation scores for multi-tile optimization. They do **not** represent objective video quality metrics such as Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), or Video Multi-Method Assessment Fusion (VMAF). All values remain fully configurable in `TileQualityConfig`.

---

## 4. Mathematical Formulations

### 4.1 Total Canvas Bitrate Consumption
For an allocation assigning quality tier $q_i \in \{Q1, Q2, Q3, Q4\}$ to each tile $i \in \{0, 1, \dots, 23\}$:
$$R_{\text{total}} = \sum_{i=0}^{23} R(q_i) \quad [\text{kbps}]$$
$$R_{\text{total, Mbps}} = \frac{R_{\text{total}}}{1000.0} \quad [\text{Mbps}]$$

#### Benchmark Extremes:
- **Uniform Q1 (Lowest):** $24 \times 100\text{ kbps} = 2,400\text{ kbps}$ ($2.4\text{ Mbps}$)
- **Uniform Q4 (Highest):** $24 \times 1000\text{ kbps} = 24,000\text{ kbps}$ ($24.0\text{ Mbps}$)

### 4.2 Spatially Weighted Viewport Utility
When evaluating viewport-adaptive allocations, tiles overlapping the user's line of sight carry greater visual importance than occluded background tiles. Given tile importance weights $w_i \ge 0$ (such as ray-intersection fractions derived from `ViewportMapper.get_active_tiles()`):
$$U_{\text{weighted}} = \sum_{i=0}^{23} w_i \cdot U(q_i)$$
When $\sum_{i} w_i = 1.0$, $U_{\text{weighted}} \in [1.00, 3.40]$ reflects the user-perceived visual quality inside their active field of view.

### 4.3 Bandwidth Budget Satisfaction
Given a client's available downlink bandwidth budget $B_{\text{budget}}$ in kbps:
$$\text{Satisfied} = \begin{cases} \text{True}, & \text{if } R_{\text{total}} \le B_{\text{budget}} \\ \text{False}, & \text{if } R_{\text{total}} > B_{\text{budget}} \end{cases}$$
$$\text{Headroom} = B_{\text{budget}} - R_{\text{total}} \quad [\text{kbps}]$$

---

## 5. Software Architecture & Implementation Details

Located in [`src/quality/quality_config.py`](file:///d:/ml%20final%20project/src/quality/quality_config.py):

```
src/quality/
├── __init__.py           # Exports QualityLevel, QualityTierSpec, TileQualityConfig, TileQualityAllocation
└── quality_config.py     # Core data classes, validation engine, and calculation routines
```

### Key Classes & Methods:
1. `QualityLevel(str, Enum)`: Standardized enum (`Q1`, `Q2`, `Q3`, `Q4`) with case-insensitive parsing (`from_string()`) supporting aliases (`"low"`, `"medium"`, `"high"`, `"ultra"`).
2. `QualityTierSpec`: Immutable dataclass storing tier attributes (`level`, `name`, `bitrate_kbps`, `utility`, `description`).
3. `TileQualityConfig`: Configuration manager validating grid dimensions ($4\times 6 = 24$ tiles), strictly increasing tier bitrates ($R_1 < R_2 < R_3 < R_4$), strictly increasing utilities ($U_1 < U_2 < U_3 < U_4$), and non-negative bandwidth budgets.
4. `TileQualityAllocation`: Manages concrete per-tile assignments, calculating `total_bitrate_kbps`, `mean_utility`, `weighted_utility()`, `satisfies_budget()`, `remaining_budget()`, `tier_distribution`, and missing tile diagnostics.

---

## 6. Verification & Test Suite Results

A dedicated unit test suite was implemented in [`tests/test_quality_config.py`](file:///d:/ml%20final%20project/tests/test_quality_config.py) covering 11 comprehensive test methods:

| Test Case Name | Target Capability | Result |
| :--- | :--- | :---: |
| `test_all_four_quality_levels_and_aliases` | Enum parsing, aliases (`"low"`, `"ultra"`), rejection of invalid strings/types | **PASS** |
| `test_default_bitrates_and_utilities` | Canonical rates (100, 300, 600, 1000 kbps), utilities, strict monotonicity | **PASS** |
| `test_uniform_allocation_bitrate_for_24_tiles` | Full 24-tile uniform calculations (2.4 Mbps to 24.0 Mbps), mean utility | **PASS** |
| `test_mixed_quality_assignment` | Complex mixed tier distribution (7.4 Mbps), tier counting, mean utility | **PASS** |
| `test_budget_satisfied_and_exceeded` | Headroom and deficit checks, strict boundary tests, type validation | **PASS** |
| `test_invalid_tile_ids_rejected` | Rejection of negative IDs (-1), out-of-range IDs (24), floats, booleans | **PASS** |
| `test_invalid_quality_levels_in_allocation_rejected` | Rejection of unknown tier names in allocation dictionaries | **PASS** |
| `test_invalid_bitrate_or_utility_configuration_rejected` | Rejection of negative rates, non-monotonic tiers, non-numeric values | **PASS** |
| `test_empty_and_incomplete_allocations` | Detection of missing tiles, `is_complete` flag, validation toggles | **PASS** |
| `test_weighted_utility_calculation` | Spatially weighted utility summation, rejection of negative/boolean weights | **PASS** |
| `test_tile_getter_setter_and_serialization` | Mutation of single-tile quality, dictionary serialization (`to_dict()`) | **PASS** |

### Test Suite Execution Summary:
- **Module Test Result:** 11 passed in 0.002s.
- **Full Repository Test Result:** **51 / 51 tests passed (100% pass rate)** with zero regressions on Day 1 modules.

