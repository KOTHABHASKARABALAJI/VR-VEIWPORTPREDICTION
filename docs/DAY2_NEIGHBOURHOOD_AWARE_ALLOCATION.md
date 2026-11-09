# Day 2 — Module 4: Neighbourhood-Aware Tile Quality Allocation

**Project:** Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming  
**Phase:** Phase 5, Day 2 — Adaptive Quality Allocation  
**Module:** Module 4 — Neighbourhood-Aware Tile Quality Allocation (Spatial Guard Band Heuristic)  
**Date:** November 2026  
**Status:** IMPLEMENTED & FULLY VERIFIED (21 / 21 Module Tests Passed, 99 / 99 Repository Tests Passed)  

---

## 1. System Overview & Motivation

Module 4 implements the **Neighbourhood-Aware Tile Quality Allocation Strategy** for the canonical 4 rows $\times$ 6 columns (24 tiles) Equirectangular Projection (ERP) video canvas ($3840 \times 1920$ resolution).

### 1.1 The Guard-Band Problem in 360° Adaptive Streaming
In tile-based 360° video streaming:
- **Pure Viewport-Aware Allocation (Module 3)** concentrates almost 100% of bandwidth into the predicted line-of-sight frustum ($\mathcal{V}$), leaving peripheral out-of-viewport tiles at baseline quality $Q1$ (100 kbps).
- However, human head motion includes continuous microsaccades, rapid gaze shifts, and unexpected rotations ($\sim 5^\circ$ to $25^\circ$).
- Because network adaptation operates on discrete temporal cycles (e.g. 500 ms chunks or lead time $k=5$ frames), a user rotating their head between adaptation intervals will abruptly rotate their field of view into tiles that were outside the predicted viewport.
- If those tiles were streamed at baseline tier $Q1$, the user suffers immediate visual quality collapse (the "black tile" or "low-res boundary" artifact).

### 1.2 The Spatial Guard-Band Solution
Neighbourhood-Aware Allocation introduces a **spatial guard band (protective buffer)** by dividing the 24 canvas tiles into three priority regions:
1. **Region 1 — Predicted Viewport ($\mathcal{V}$):** Primary active tiles intersecting the predicted FoV.
2. **Region 2 — First-Ring Neighbourhood Margin ($\mathcal{M}$):** Tiles immediately adjacent to the viewport that are at high risk of entering the user's line of sight during head rotation.
3. **Region 3 — Background Canvas ($\mathcal{B}$):** Remaining tiles far from the viewport, maintained at baseline $Q1$.

---

## 2. Topological Neighbourhood Identification

The 24 spatial tiles are arranged in a 4-row $\times$ 6-column grid on the ERP canvas:
$$\text{tile\_id} = r \times 6 + c \quad \text{where } r \in \{0, 1, 2, 3\}, \; c \in \{0, 1, 2, 3, 4, 5\}$$

### 2.1 Geometric Boundary Rules
1. **Horizontal Circular Wraparound (Yaw / Azimuth):**
   The ERP canvas represents a continuous $360^\circ$ cylinder in longitude. Column movement wraps around circularly:
   $$c_{\text{adj}} = (c + \Delta c) \pmod 6 \quad \forall \Delta c \in \{-1, 0, 1\}$$
   - Tile in column 0 (e.g. Tile 6: row 1, col 0) wraps left to column 5 (Tile 11).
   - Tile in column 5 (e.g. Tile 11: row 1, col 5) wraps right to column 0 (Tile 6).
2. **Vertical Polar Clamping (Pitch / Latitude):**
   The top canvas row ($r=0$) maps to the North Pole (Zenith, $+90^\circ$ pitch) and the bottom row ($r=3$) maps to the South Pole (Nadir, $-90^\circ$ pitch). Longitudes terminate at the poles; movement does **not** wrap over the poles:
   $$0 \le r + \Delta r \le 3 \quad \forall \Delta r \in \{-1, 0, 1\}$$
   - Rows $r < 0$ and $r > 3$ are strictly invalid and rejected.
3. **Connectivity (8-Neighbourhood):**
   Includes top, bottom, left, right, and all 4 diagonals (top-left, top-right, bottom-left, bottom-right).
   - **Zenith tiles (row 0, IDs 0..5):** Exactly 5 neighbours (left, right, bottom, bottom-left, bottom-right).
   - **Nadir tiles (row 3, IDs 18..23):** Exactly 5 neighbours (left, right, top, top-left, top-right).
   - **Interior tiles (rows 1..2, IDs 6..17):** Exactly 8 neighbours.
4. **Self-Exclusion & Uniqueness:**
   A tile is never included in its own neighbourhood ($\Delta r = \Delta c = 0$ excluded). All returned neighbour lists are strictly unique, sorted integer sets in $[0, 23]$.

Implemented in `TileNeighbourhoodFinder` and verified against Day 1 `TileGridPartitioner.get_adjacent_tiles()`.

---

## 3. Mathematical Importance Weight Formulation

Let:
- $\mathbf{w}^{\text{vp}} = [w_0^{\text{vp}}, \dots, w_{23}^{\text{vp}}]$ be the normalized input viewport weights from `ViewportMapper` ($\sum w_i^{\text{vp}} = 1.0$).
- $\mathcal{V} = \{ i \in [0, 23] \mid w_i^{\text{vp}} > 0 \}$ be the active viewport tiles (Region 1).
- For each $i \in \mathcal{V}$, let $\mathcal{N}_i$ denote its first-ring topological neighbours.
- The first-ring neighbourhood margin (Region 2) is:
  $$\mathcal{M} = \left( \bigcup_{i \in \mathcal{V}} \mathcal{N}_i \right) \setminus \mathcal{V}$$
- The background set (Region 3) is:
  $$\mathcal{B} = \{0, \dots, 23\} \setminus (\mathcal{V} \cup \mathcal{M})$$
- $\alpha \in [0.0, 1.0)$ be the configurable `neighbour_weight_factor` (default $\alpha = 0.25$).

### 3.1 Margin Tile Overlap Scoring
Neighbouring tiles differ in spatial risk: a margin tile bordering multiple active viewport tiles is more likely to enter the FoV than a corner tile bordering only one. For each $j \in \mathcal{M}$, its raw adjacency overlap score is:
$$s_j = \sum_{i \in \mathcal{V} \cap \mathcal{N}_j} w_i^{\text{vp}}$$
Normalizing across the margin:
$$\bar{s}_j = \frac{s_j}{\sum_{k \in \mathcal{M}} s_k}$$

### 3.2 Blended Weight Vector
The final 24-tile neighbourhood-aware weight vector $\mathbf{w}^{\text{neigh}}$ is:
$$w_i^{\text{neigh}} = \begin{cases}
(1 - \alpha) \cdot w_i^{\text{vp}}, & \text{if } i \in \mathcal{V} \text{ (Region 1 — Viewport)} \\
\alpha \cdot \bar{s}_j, & \text{if } i \in \mathcal{M} \text{ (Region 2 — Neighbourhood Margin)} \\
0.0, & \text{if } i \in \mathcal{B} \text{ (Region 3 — Background)}
\end{cases}$$

### 3.3 Mathematical Properties & Proofs
1. **Exact Unit Sum:**
   $$\sum_{i=0}^{23} w_i^{\text{neigh}} = (1 - \alpha)\sum_{i \in \mathcal{V}} w_i^{\text{vp}} + \alpha \sum_{j \in \mathcal{M}} \bar{s}_j + \sum_{k \in \mathcal{B}} 0 = (1 - \alpha)(1.0) + \alpha(1.0) + 0 = 1.0$$
2. **Preservation of Relative Viewport Priorities:**
   For any two active viewport tiles $a, b \in \mathcal{V}$:
   $$\frac{w_a^{\text{neigh}}}{w_b^{\text{neigh}}} = \frac{(1 - \alpha) w_a^{\text{vp}}}{(1 - \alpha) w_b^{\text{vp}}} = \frac{w_a^{\text{vp}}}{w_b^{\text{vp}}}$$
   The relative importance ranking within the viewport is strictly unaltered.
3. **Continuity & Backward Compatibility:**
   When $\alpha = 0.0$, $w_i^{\text{neigh}} = w_i^{\text{vp}}$ for all $i \in \mathcal{V}$ and $w_j^{\text{neigh}} = 0$ for all $j \in \mathcal{M}$, exactly reproducing Module 3 pure Viewport-Aware Allocation.

---

## 4. Greedy Marginal Quality Allocation Algorithm

Using the blended weights $\mathbf{w}^{\text{neigh}}$, the allocator applies the deterministic greedy marginal rate-utility optimization:
1. Initialize all 24 tiles to baseline tier $Q1$ ($R_{\text{total}} = 24 \times 100 = 2,400\text{ kbps}$).
2. For each tile $i$ currently at tier $q_i < Q4$, evaluate:
   $$\text{score}_i = \frac{w_i^{\text{neigh}} \cdot (U(q_i^+) - U(q_i))}{R(q_i^+) - R(q_i)}$$
3. Filter candidates where $R_{\text{total}} + \Delta R_i \le B_{\text{budget}}$.
4. Select feasible candidate with maximum score (deterministic tie-breaking: lower tile ID).
5. Apply upgrade, record upgrade by region (Viewport, Neighbourhood, or Background), and repeat until no feasible upgrade fits within the budget.

---

## 5. Comprehensive Multi-Strategy Comparison

Using identical predicted viewport orientations ($\text{yaw}=0^\circ, \text{pitch}=0^\circ$, 8 active viewport tiles, 12 first-ring margin tiles, 4 background tiles) across all 5 canonical budgets:

### 5.1 Full Performance Comparison Matrix

| Available Budget ($B_{\text{budget}}$) | Allocation Strategy | Total Bitrate ($R_{\text{total}}$) | Remaining Headroom | Viewport Utility (Original $U_{\text{weighted}}^{\text{orig}}$) | Guard-Band Utility ($U_{\text{weighted}}^{\text{neigh}}$) | Mean Canvas Utility | Quality Tier Distribution | Upgrades by Region (V / M / B) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2,400 kbps** | **Uniform (Mod 2)**<br>**Viewport (Mod 3)**<br>**Neighbourhood ($\alpha=0.25$)**<br>**Neighbourhood ($\alpha=0.50$)** | 2,400 kbps<br>2,400 kbps<br>2,400 kbps<br>2,400 kbps | 0 kbps<br>0 kbps<br>0 kbps<br>0 kbps | 1.0000<br>1.0000<br>1.0000<br>1.0000 | 1.0000<br>1.0000<br>1.0000<br>1.0000 | 1.0000<br>1.0000<br>1.0000<br>1.0000 | 24 $\times$ Q1<br>24 $\times$ Q1<br>24 $\times$ Q1<br>24 $\times$ Q1 | 0 / 0 / 0<br>0 / 0 / 0<br>0 / 0 / 0<br>0 / 0 / 0 |
| **5,000 kbps** | **Uniform (Mod 2)**<br>**Viewport (Mod 3)**<br>**Neighbourhood ($\alpha=0.25$)**<br>**Neighbourhood ($\alpha=0.50$)** | 2,400 kbps *(stuck)*<br>5,000 kbps<br>5,000 kbps<br>4,900 kbps | 2,600 kbps<br>0 kbps<br>0 kbps<br>100 kbps | 1.0000<br>**2.8268**<br>2.8041<br>2.4862 | 1.0000<br>2.8268<br>2.3960<br>2.0718 | 1.0000<br>1.3667<br>1.3667<br>1.4333 | 24 $\times$ Q1<br>19 Q1, 1 Q2, 3 Q3, 1 Q4<br>19 Q1, 1 Q2, 3 Q3, 1 Q4<br>16 Q1, 5 Q2, 3 Q3, 0 Q4 | 0 / 0 / 0<br>10 / 0 / 0<br>9 / 1 / 0<br>7 / 4 / 0 |
| **7,200 kbps** | **Uniform (Mod 2)**<br>**Viewport (Mod 3)**<br>**Neighbourhood ($\alpha=0.25$)**<br>**Neighbourhood ($\alpha=0.50$)** | 7,200 kbps<br>7,100 kbps<br>7,200 kbps<br>7,200 kbps | 0 kbps<br>100 kbps<br>0 kbps<br>0 kbps | 2.0000<br>**3.3039**<br>3.2041<br>2.6531 | 2.0000<br>3.3039<br>2.8623<br>2.5895 | 2.0000<br>1.6000<br>1.6500<br>1.7667 | 24 $\times$ Q2<br>16 Q1, 3 Q2, 1 Q3, 4 Q4<br>14 Q1, 6 Q2, 0 Q3, 4 Q4<br>12 Q1, 4 Q2, 8 Q3, 0 Q4 | 0 / 0 / 0<br>17 / 0 / 0<br>12 / **6** / 0<br>8 / **12** / 0 |
| **10,000 kbps** | **Uniform (Mod 2)**<br>**Viewport (Mod 3)**<br>**Neighbourhood ($\alpha=0.25$)**<br>**Neighbourhood ($\alpha=0.50$)** | 7,200 kbps *(stuck)*<br>10,000 kbps<br>10,000 kbps<br>9,900 kbps | 2,800 kbps<br>0 kbps<br>0 kbps<br>100 kbps | 2.0000<br>**3.4000**<br>3.2857<br>3.2041 | 2.0000<br>3.4000<br>3.1215<br>3.0056 | 2.0000<br>1.8833<br>2.0583<br>1.9500 | 24 $\times$ Q2<br>14 Q1, 2 Q2, 0 Q3, 8 Q4<br>8 Q1, 8 Q2, 3 Q3, 5 Q4<br>12 Q1, 3 Q2, 3 Q3, 6 Q4 | 0 / 0 / 0<br>24 / 0 / 2<br>16 / **13** / 0<br>12 / **15** / 0 |
| **14,400 kbps** | **Uniform (Mod 2)**<br>**Viewport (Mod 3)**<br>**Neighbourhood ($\alpha=0.25$)**<br>**Neighbourhood ($\alpha=0.50$)** | 14,400 kbps<br>14,300 kbps<br>14,400 kbps<br>14,300 kbps | 0 kbps<br>100 kbps<br>0 kbps<br>100 kbps | 2.8000<br>**3.4000**<br>3.3510<br>3.3039 | 2.8000<br>3.4000<br>3.3388<br>3.3519 | 2.7999<br>2.3417<br>2.4500<br>2.4000 | 24 $\times$ Q3<br>10 Q1, 1 Q2, 0 Q3, 13 Q4<br>8 Q1, 0 Q2, 6 Q3, 10 Q4<br>8 Q1, 3 Q2, 1 Q3, 12 Q4 | 0 / 0 / 0<br>24 / 0 / 16<br>20 / **22** / 0<br>17 / **24** / 0 |

---

## 6. Critical Scientific Analysis: The Guard-Band Trade-Off

When evaluated strictly on the **original viewport weights** ($U_{\text{weighted}}^{\text{orig}}$), Neighbourhood-Aware Allocation achieves slightly lower in-FoV utility than Viewport-Aware Allocation:
- At 7,200 kbps: Viewport-Aware achieves $3.3039$, whereas Neighbourhood ($\alpha=0.25$) achieves $3.2041$ ($-0.0998$, or $-3.0\%$).
- At 10,000 kbps: Viewport-Aware achieves $3.4000$ (all viewport tiles at Q4), whereas Neighbourhood ($\alpha=0.25$) achieves $3.2857$ ($-0.1143$, or $-3.4\%$).

### Why This Trade-Off is Structurally Essential:
1. **The Cost of Defense:**
   In Neighbourhood-Aware Allocation ($\alpha=0.25$ at 7,200 kbps), 6 upgrades are deliberately diverted into first-ring margin tiles. As a result, 6 margin tiles are elevated from $Q1$ (100 kbps) to $Q2$ (300 kbps).
2. **Motion Resilience:**
   - Under static viewing (user head perfectly still), these 6 margin upgrades contribute $0$ to $U_{\text{weighted}}^{\text{orig}}$ because the user is not yet looking at them.
   - However, if the user rotates their head by $15^\circ$, in Viewport-Aware streaming they immediately encounter $Q1$ tiles ($100\text{ kbps}$, causing severe visual quality drop).
   - In Neighbourhood-Aware streaming, the tiles entering their field of view are already buffered at $Q2$ ($300\text{ kbps}$), providing seamless visual continuity.
3. **Parameter Sensitivity ($\alpha$):**
   - $\alpha = 0.0$: Pure Viewport-Aware streaming (maximum in-FoV utility, zero motion resilience).
   - $\alpha = 0.25$ (Recommended Default): Optimal balance—retains $>96\%$ of maximum in-FoV utility while assigning 6–13 guard-band upgrades to surrounding margin tiles.
   - $\alpha = 0.50$: Aggressive guard band—diverts excessive bandwidth to margins, causing noticeable in-FoV degradation ($2.65$ vs $3.30$ at 7,200 kbps).

---

## 7. Software Architecture & Verification

Located in [`src/quality/neighbourhood_allocator.py`](file:///d:/ml%20final%20project/src/quality/neighbourhood_allocator.py):

```
src/quality/
├── __init__.py                # Exports all Module 1, 2, 3, and 4 classes
├── quality_config.py          # Module 1: QualityLevel, TileQualityConfig, TileQualityAllocation
├── uniform_allocator.py       # Module 2: UniformQualityAllocator, InsufficientBandwidthError
├── viewport_allocator.py      # Module 3: ViewportQualityAllocator, ViewportAllocationResult
└── neighbourhood_allocator.py # Module 4: TileNeighbourhoodFinder, NeighbourhoodQualityAllocator
```

### 7.1 Test Suite Results
Verified by [`tests/test_neighbourhood_allocator.py`](file:///d:/ml%20final%20project/tests/test_neighbourhood_allocator.py) containing **21 dedicated unit tests**:

```text
Ran 21 tests in 0.020s
OK
```

### 7.2 Full Repository Regression Status
Executing `py -3.14 -m unittest discover -s tests -v`:

```text
Ran 99 tests in 3.268s
OK
```
- **Total Tests Discovered:** **99**
- **Passed:** **99 (100%)**
- **Failed:** **0**
- **Errors:** **0**
- **Breakdown:**
  - Day 1 Core: 40 tests
  - Day 2 Module 1: 11 tests
  - Day 2 Module 2: 13 tests
  - Day 2 Module 3: 14 tests
  - Day 2 Module 4: 21 tests

---

## 8. Scientific Scope & Disclaimers

> [!NOTE]
> 1. **Heuristic Nature:** Neighbourhood-aware quality allocation is a heuristic optimization inspired by spatial guard-band buffering in adaptive streaming. It does not constitute a formal proof of global Pareto-optimality.
> 2. **Simulated Metrics:** Bitrate figures ($100, 300, 600, 1000\text{ kbps}$) and utilities ($1.00, 2.00, 2.80, 3.40$) are simulated modeling assumptions. They do not represent measured video codec streams (H.264/H.265/AV1) or objective quality metrics such as PSNR, SSIM, or VMAF.
> 3. **Head Motion Modeling:** The guard band reduces risk under small rotational perturbations; extreme rapid saccades ($>90^\circ$) may still exceed the first-ring neighbourhood margin.

