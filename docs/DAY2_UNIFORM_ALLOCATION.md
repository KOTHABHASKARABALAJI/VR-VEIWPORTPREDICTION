# Day 2 — Module 2: Uniform Quality Allocation

**Project:** Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming  
**Phase:** Phase 5, Day 2 — Adaptive Quality Allocation  
**Module:** Module 2 — Uniform Tile Quality Allocation (Benchmark Baseline)  
**Date:** November 2026  
**Status:** IMPLEMENTED & FULLY TESTED (13 / 13 Module Tests Passed, 64 / 64 Repository Tests Passed)  

---

## 1. System Overview & Purpose

Module 2 implements the **Uniform Quality Allocation Strategy** for the 360° Equirectangular Projection (ERP) video canvas. Under uniform streaming, the delivery client or media server allocates the identical video quality tier $q \in \{Q1, Q2, Q3, Q4\}$ across all 24 spatial tiles ($4\text{ rows} \times 6\text{ columns}$).

This module serves as the primary **benchmark baseline** in this research project. Subsequent viewport-adaptive (Module 3) and neighbourhood-aware (Module 4) allocation algorithms will be evaluated against this uniform baseline to quantify how predictive ML-guided quality allocation improves visual utility and bandwidth efficiency under identical downlink bandwidth constraints.

---

## 2. Mathematical Formulation

### 2.1 Uniform Quality Selection Problem
Given:
- A canonical grid of $N = 24$ spatial tiles,
- Four discrete simulated quality levels $\mathcal{Q} = \{Q1, Q2, Q3, Q4\}$ with per-tile bitrates $R(q)$ such that $R(Q1) < R(Q2) < R(Q3) < R(Q4)$,
- A client available bandwidth budget $B_{\text{budget}} \ge 0$ in kbps.

The uniform allocator selects the unique highest quality level $q^*$ that satisfies the bandwidth constraint across the entire canvas:
$$q^* = \max \left\{ q \in \mathcal{Q} \;\middle|\; \sum_{i=0}^{23} R(q) \le B_{\text{budget}} \right\} = \max \left\{ q \in \mathcal{Q} \;\middle|\; 24 \cdot R(q) \le B_{\text{budget}} \right\}$$

Every tile $i \in \{0, 1, \dots, 23\}$ is then uniformly assigned quality $q^*$:
$$q_i = q^* \quad \forall i \in \{0, 1, \dots, 23\}$$

### 2.2 Discrete 24-Tile Canvas Cost Schedule
Under the canonical project configuration ($4 \times 6 = 24$ tiles, $3840 \times 1920$ canvas):

| Quality Level | Tier Name | Per-Tile Bitrate $R(q)$ | 24-Tile Total Bitrate ($24 \cdot R(q)$) | Total Bitrate (Mbps) | Canvas Utility $U(q)$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Q1** | Low | 100 kbps | **2,400 kbps** | 2.40 Mbps | 1.00 |
| **Q2** | Medium | 300 kbps | **7,200 kbps** | 7.20 Mbps | 2.00 |
| **Q3** | High | 600 kbps | **14,400 kbps** | 14.40 Mbps | 2.80 |
| **Q4** | Ultra | 1000 kbps | **24,000 kbps** | 24.00 Mbps | 3.40 |

### 2.3 Step-Function Allocation Policy
Because quality is uniform across all tiles, the total transmitted bitrate follows a discrete step function with wide horizontal plateaus:

$$\text{Selected Tier}(B_{\text{budget}}) = \begin{cases}
\text{INFEASIBLE} & \text{if } B_{\text{budget}} < 2,400\text{ kbps} \\
Q1 \text{ (Low, } 2,400\text{ kbps}) & \text{if } 2,400\text{ kbps} \le B_{\text{budget}} < 7,200\text{ kbps} \\
Q2 \text{ (Medium, } 7,200\text{ kbps}) & \text{if } 7,200\text{ kbps} \le B_{\text{budget}} < 14,400\text{ kbps} \\
Q3 \text{ (High, } 14,400\text{ kbps}) & \text{if } 14,400\text{ kbps} \le B_{\text{budget}} < 24,000\text{ kbps} \\
Q4 \text{ (Ultra, } 24,000\text{ kbps}) & \text{if } B_{\text{budget}} \ge 24,000\text{ kbps}
\end{cases}$$

### 2.4 Remaining Bandwidth Headroom
The unused bandwidth headroom $H$ is computed as:
$$H = B_{\text{budget}} - R_{\text{total}} = B_{\text{budget}} - 24 \cdot R(q^*)$$

For example:
- At $B_{\text{budget}} = 2,400\text{ kbps} \implies \text{Tier } Q1$, $R_{\text{total}} = 2,400\text{ kbps}$, Headroom = $0\text{ kbps}$.
- At $B_{\text{budget}} = 5,000\text{ kbps} \implies \text{Tier } Q1$, $R_{\text{total}} = 2,400\text{ kbps}$, Headroom = $2,600\text{ kbps}$ (wasted unallocated capacity because 7,200 kbps is needed for Q2).
- At $B_{\text{budget}} = 7,200\text{ kbps} \implies \text{Tier } Q2$, $R_{\text{total}} = 7,200\text{ kbps}$, Headroom = $0\text{ kbps}$.
- At $B_{\text{budget}} = 14,400\text{ kbps} \implies \text{Tier } Q3$, $R_{\text{total}} = 14,400\text{ kbps}$, Headroom = $0\text{ kbps}$.
- At $B_{\text{budget}} = 24,000\text{ kbps} \implies \text{Tier } Q4$, $R_{\text{total}} = 24,000\text{ kbps}$, Headroom = $0\text{ kbps}$.
- At $B_{\text{budget}} = 30,000\text{ kbps} \implies \text{Tier } Q4$, $R_{\text{total}} = 24,000\text{ kbps}$, Headroom = $6,000\text{ kbps}$ (saturation at maximum tier).

---

## 3. Infeasibility Boundary & Safety Policy

### 3.1 Strict Budget Adherence
> [!IMPORTANT]
> The uniform allocator strictly guarantees that **no allocation will ever exceed the available bandwidth budget**:
> $$R_{\text{total}} \le B_{\text{budget}}$$

### 3.2 Handling Sub-2,400 kbps Budgets
The minimum possible uniform transmission cost across all 24 tiles is $24 \times 100\text{ kbps} = 2,400\text{ kbps}$. When $B_{\text{budget}} < 2,400\text{ kbps}$ (e.g., $1,500\text{ kbps}$ or $2,000\text{ kbps}$):
1. **No silent over-allocation:** The allocator does **not** silently assign Q1 and exceed the client's network capacity.
2. **Deterministic Exception:** The method `allocate()` raises `InsufficientBandwidthError(ValueError)` with a descriptive error message indicating the required minimum ($2,400\text{ kbps}$) and the deficit.
3. **Structured Result Inspection:** Alternatively, `allocate_with_result()` returns a `UniformAllocationResult` with `is_feasible = False`, `selected_quality = None`, `allocation = None`, and an explicit diagnostic message.

This highlights the fundamental structural limitation of non-adaptive uniform streaming: clients on sub-2.4 Mbps connections cannot receive full-canvas video without either stalling or dropping tiles. Viewport-adaptive allocation (Modules 3 and 4) solves this problem by allocating low quality or dropping non-visible tiles to sustain high quality within the user's active field of view.

---

## 4. Architecture & Implementation

Module 2 resides in [`src/quality/uniform_allocator.py`](file:///d:/ml%20final%20project/src/quality/uniform_allocator.py) and seamlessly integrates with Module 1:

```
src/quality/
├── __init__.py           # Unified exports for Module 1 & Module 2
├── quality_config.py     # Module 1: QualityLevel, TileQualityConfig, TileQualityAllocation
└── uniform_allocator.py  # Module 2: UniformQualityAllocator, UniformAllocationResult, InsufficientBandwidthError
```

### 4.1 Key Classes & Functions

#### 1. `InsufficientBandwidthError(ValueError)`
Custom exception raised when available budget is strictly less than the cost of Q1 across the canvas ($B_{\text{budget}} < N \cdot R(Q1)$).

#### 2. `UniformAllocationResult`
Immutable dataclass providing a comprehensive audit of the allocation decision:
- `selected_quality: Optional[QualityLevel]` (e.g. `QualityLevel.Q2`)
- `total_bitrate_kbps: float` (e.g. `7200.0`)
- `total_bitrate_mbps: float` (e.g. `7.2`)
- `bandwidth_budget_kbps: float` (e.g. `10000.0`)
- `remaining_budget_kbps: float` (e.g. `2800.0`)
- `is_budget_satisfied: bool` (`True`)
- `is_feasible: bool` (`True` if budget $\ge 2,400$ kbps)
- `allocation: Optional[TileQualityAllocation]` (Concrete 24-tile allocation object)
- `message: str` (Human-readable diagnostic summary)

#### 3. `UniformQualityAllocator`
Allocation engine initialized with an optional `TileQualityConfig` (defaults to canonical $4\times 6$ = 24 tile configuration):
- `get_tier_cost_kbps(quality)`: Computes $N \cdot R(q)$.
- `get_all_tier_costs()`: Returns dictionary mapping each `QualityLevel` to its 24-tile total cost.
- `select_highest_feasible_tier(bandwidth_budget_kbps)`: Implements the step-function selection algorithm, returning the highest feasible tier or `None`.
- `allocate(bandwidth_budget_kbps)`: Returns a validated `TileQualityAllocation` with complete tile IDs $0 \dots 23$.
- `allocate_with_result(bandwidth_budget_kbps)`: High-level wrapper returning `UniformAllocationResult`.

#### 4. `allocate_uniform_quality(bandwidth_budget_kbps, config=None)`
Convenience module-level function for rapid evaluation and benchmarking.

---

## 5. Verification & Test Suite Results

The module is verified by [`tests/test_uniform_allocator.py`](file:///d:/ml%20final%20project/tests/test_uniform_allocator.py) containing **13 dedicated unit test methods**:

```
Ran 13 tests in 0.002s
OK
```

### Test Case Coverage:
1. **`test_budget_2400_selects_q1`:** Confirms budget of 2,400 kbps selects Q1 (Low) with 0 kbps headroom.
2. **`test_budget_5000_selects_q1`:** Confirms budget of 5,000 kbps selects Q1 (2,400 kbps) with 2,600 kbps headroom (cannot reach Q2 at 7,200 kbps).
3. **`test_budget_7200_selects_q2`:** Confirms budget of 7,200 kbps selects Q2 (Medium) with 0 kbps headroom.
4. **`test_budget_14400_selects_q3`:** Confirms budget of 14,400 kbps selects Q3 (High) with 0 kbps headroom.
5. **`test_budget_24000_selects_q4`:** Confirms budget of 24,000 kbps selects Q4 (Ultra) with 0 kbps headroom.
6. **`test_budget_above_24000_selects_q4`:** Confirms budgets above 24,000 kbps (e.g. 30,000 kbps) cap at Q4 with surplus headroom.
7. **`test_budget_below_2400_raises_insufficient_bandwidth`:** Confirms budgets below 2,400 kbps raise `InsufficientBandwidthError` and report `is_feasible = False`.
8. **`test_budget_exactly_equal_to_each_tier_cost`:** Verifies boundary conditions at exactly 2,400, 7,200, 14,400, and 24,000 kbps.
9. **`test_complete_assignment_all_24_tiles`:** Validates that every tile ID from 0 to 23 has an assigned quality level and no tiles are omitted.
10. **`test_no_allocation_ever_exceeds_supplied_budget`:** Sweeps 20 arbitrary budget values confirming $R_{\text{total}} \le B_{\text{budget}}$ holds universally.
11. **`test_invalid_negative_nan_infinite_budgets_rejected`:** Confirms rejection of negative numbers, NaN, $\pm\infty$, booleans, and non-numeric types.
12. **`test_tier_costs_calculation`:** Validates accurate reporting of individual tier total bitrates ($2.4, 7.2, 14.4, 24.0\text{ Mbps}$).
13. **`test_convenience_function_and_serialization`:** Validates `allocate_uniform_quality()` helper and `UniformAllocationResult.to_dict()` dictionary serialization.

### Full Repository Regression Status:
- **Total Test Suite:** **64 tests passed, 0 failures, 0 errors** across all project modules.
- **Day 1 Integrity:** All Day 1 features, models, sequences, inference pipelines, and tiling evaluators remain untouched and fully functioning.

---

## 6. Scientific Scope & Disclaimers

> [!NOTE]
> 1. **Baseline Role:** Uniform allocation represents non-adaptive legacy streaming (e.g., transmitting the complete 360° sphere at uniform resolution). It serves as the formal control group against which our predictive ML-based adaptive allocation algorithms will be benchmarked.
> 2. **Simulated Bitrates:** All bandwidth figures ($100, 300, 600, 1000\text{ kbps}$ per tile; $2.4, 7.2, 14.4, 24.0\text{ Mbps}$ total) are **simulated transmission assumptions** derived from standard tile-based streaming models. They do not represent measured video codec outputs or guarantee specific network packet behavior.
> 3. **Non-Adaptive Inefficiency:** Notice that between 2,400 kbps and 7,199 kbps, uniform streaming wastes significant headroom (up to 4,799 kbps) because it cannot selectively upgrade tiles. Adaptive allocation in Modules 3 and 4 directly exploits this unused capacity.

