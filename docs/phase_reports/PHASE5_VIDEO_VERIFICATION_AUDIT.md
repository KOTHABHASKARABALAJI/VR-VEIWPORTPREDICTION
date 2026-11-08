# Phase 5.1.1 — Independent Video Verification Audit

**Project Title:** Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming  
**Phase:** Phase 5.1.1 (Independent Video Acquisition Verification Audit)  
**Date:** November 4, 2026  
**Document Location:** `docs/PHASE5_VIDEO_VERIFICATION_AUDIT.md`  

---

## 1. Executive Summary & Audit Verdict

This audit independently verified the acquisition report for [`data/videos/11.mp4`](file:///d:/ml%20final%20project/data/videos/11.mp4) (*Journey To The Edge Of Space*, Seeker VR), acquired as the benchmark video for Phase 5.

### Summary Audit Verdict:
* **EAC Projection Confirmed:** The downloaded MP4 container does not carry MP4-level spherical box atoms (`sv3d`, `st3d`), but the underlying video bitstream is visually packaged in **Equi-Angular Cubemap (EAC)** format at $3840 \times 2048$ resolution.
* **ERP Transformation Verified:** The FFmpeg `v360=eac:equirect:w=3840:h=1920:interp=cube` filter accurately converts the EAC stream into a **canonical 2:1 Equirectangular Projection ($3840 \times 1920$)**. Seam and wrap continuity tests show negligible interpolation variation (mean wrap difference: $1.79 / 255$).
* **Dataset Alignment Scope Corrected:** While visual scene content and user gaze orientation (empirical $-19.45^\circ$ pitch bias) confirm high-level behavioral alignment with the AVTrack360 viewing test, **the claim of "$\pm 1$-frame alignment" cannot be verified** without access to the authors' 2018 raw AVI master (`tmp\edgeofspace.avi`). This limitation is explicitly documented.
* **Tile Geometry Mathematically Valid:** A $4 \times 6$ spatial grid on the $3840 \times 1920$ ERP frame yields exactly 24 tiles of $640 \times 480$ pixels ($60^\circ \times 45^\circ$ spherical coverage per tile), perfectly covering the canvas with zero missing pixels and zero overlap.
* **Readiness:** **Conditioned Approval for Phase 5.2.** The asset is fully suitable for tile partitioning, coordinate mapping, and streaming simulations once the 30-second stimulus window (`00:00:10` to `00:00:40`) is converted to canonical ERP.

---

## 2. EAC Projection & Spherical Metadata Audit (Item 1)

### 2.1 Container-Level Spherical Metadata Inspection
A binary atom walk of `data/videos/11.mp4` was executed across the `moov` container box (offset 28, length 326,554 bytes):
* **Spherical Atoms Tested:** `st3d` (stereoscopic 3D), `sv3d` (spherical video v1), `proj` (projection box), `equi` (equirectangular projection box), `cbmp` (cubemap box), and Google spherical `uuid` (`ffcc8263-f855-4a93-8814-58a4441444bc`).
* **Finding:** **None of these atoms are present in the MP4 container.**
* **Explanation:** When `yt-dlp` downloaded the VP9 DASH video stream (format 313, flagged as `mesh` in YouTube manifests) and multiplexed it with AAC audio via FFmpeg (`-c:v copy -c:a copy`), FFmpeg generated a generic ISO/IEC 14496-14 MP4 container (`major_brand: isom`). Container-level spherical tags were not synthesized.

### 2.2 Bitstream-Level EAC Packing Verification
Visual inspection of the raw frames (`frame_05s.jpg`, `frame_10s.jpg`, `frame_25s.jpg`) confirmed that the video frames are packed in YouTube's proprietary **Equi-Angular Cubemap (EAC)** 3×2 layout:
* **Dimensions:** $3840 \times 2048$ pixels ($15:8$ aspect ratio).
* **Face Geometry:** Six cube faces, each measuring $1280 \times 1024$ pixels.
* **Top Row ($Y \in [0, 1024]$):** Left face ($X \in [0, 1280]$), Front face ($X \in [1280, 2560]$), Right face ($X \in [2560, 3840]$).
* **Bottom Row ($Y \in [1024, 2048]$):** Bottom/Nadir face (rotated $90^\circ$ counter-clockwise), Back face (rotated), Top/Zenith face (rotated).

```text
+---------------------+---------------------+---------------------+
|    Left (-90° Y)    |     Front (0° Y)    |    Right (+90° Y)   |  (Y: 0 to 1024)
+---------------------+---------------------+---------------------+
|    Bottom (Nadir)   |     Back (180° Y)   |     Top (Zenith)    |  (Y: 1024 to 2048)
+---------------------+---------------------+---------------------+
```

### 2.3 FFmpeg v360 Conversion Command & Seam Analysis
The conversion filter utilized is:
```bash
ffmpeg -i data/videos/previews/frame_10s.jpg -vf "v360=eac:equirect:w=3840:h=1920:interp=cube" data/videos/previews/frame_10s_bicubic.jpg -y
```

#### Objective Seam & Distortion Evaluation:
To verify whether the transformation produces artifacts or discontinuities, we performed programmatic pixel-level analyses on [`frame_10s_erp.jpg`](file:///d:/ml%20final%20project/data/videos/previews/frame_10s_erp.jpg):

1. **Circular Boundary Wrap Continuity (Column 0 vs. Column 3839):**
   * Longitude $-180^\circ$ and $+180^\circ$ meet at the horizontal edges.
   * **Mean Absolute Pixel Difference:** **$1.79 / 255$** (less than 1% variance, consistent with bicubic subpixel interpolation and JPEG quantization).
   * **Result:** Seamless horizontal wrapping.
2. **Internal Cubemap Face Boundaries:**
   * Vertical cube seams occur at azimuths $-135^\circ$ ($X=480$), $-45^\circ$ ($X=1440$), $+45^\circ$ ($X=2400$), and $+135^\circ$ ($X=3360$).
   * Mean horizontal gradients at face seams:
     * $X = 480$: $1.99$
     * $X = 1440$: $2.81$
     * $X = 2400$: $1.81$
     * $X = 3360$: $2.39$
     * Baseline global image gradient: $0.94$.
   * **Result:** No step discontinuities, tearing, or black border lines. Slight sub-pixel gradient elevation is an expected mathematical byproduct of cubemap edge re-sampling.
3. **Polar Convergence:**
   * Zenith row ($Y=0$, elevation $+90^\circ$): Standard deviation across all 3,840 columns is $3.09$ (uniform sky).
   * Nadir row ($Y=1919$, elevation $-90^\circ$): Standard deviation is $10.41$ (weather balloon payload ground bracket).
   * **Result:** Correct spherical convergence without polar pinching singularities.

---

## 3. Canonical ERP Output Verification (Item 2)

* **Full Video ERP Status:** A full converted 30-second ERP video does **not** exist yet, in strict compliance with Phase 5 preservation guidelines ("Do not crop, resize, re-encode, or partition it yet").
* **Preview Frames Generated & Verified:**
  * [`data/videos/previews/frame_05s_erp.jpg`](file:///d:/ml%20final%20project/data/videos/previews/frame_05s_erp.jpg) ($3840 \times 1920$, 2:1 DAR)
  * [`data/videos/previews/frame_10s_erp.jpg`](file:///d:/ml%20final%20project/data/videos/previews/frame_10s_erp.jpg) ($3840 \times 1920$, 2:1 DAR)
  * [`data/videos/previews/frame_25s_erp.jpg`](file:///d:/ml%20final%20project/data/videos/previews/frame_25s_erp.jpg) ($3840 \times 1920$, 2:1 DAR)
  * [`data/videos/previews/frame_40s_erp.jpg`](file:///d:/ml%20final%20project/data/videos/previews/frame_40s_erp.jpg) ($3840 \times 1920$, 2:1 DAR)
  * [`data/videos/previews/frame_60s_erp.jpg`](file:///d:/ml%20final%20project/data/videos/previews/frame_60s_erp.jpg) ($3840 \times 1920$, 2:1 DAR)
* **Preservation Status:** The downloaded master file [`data/videos/11.mp4`](file:///d:/ml%20final%20project/data/videos/11.mp4) remains 100% untouched.

---

## 4. Dataset Alignment Audit & Critical Findings (Item 3)

### 4.1 AVTrack360 Stimulus Metadata for Video 11
From [`data/raw/SRC_info/11.json`](file:///d:/ml%20final%20project/data/raw/SRC_info/11.json) and Table 1 of the Fremerey et al. (2018) paper:
* **Stimulus Name:** `Edge of Space` (internal testbed filename: `tmp\edgeofspace.avi`)
* **Documented Extraction Timestamp:** `00:00:10` (10 seconds into the original YouTube video)
* **Stimulus Duration:** $30.033333\text{ seconds}$
* **Number of Frames:** $901\text{ frames}$ at $30/1\text{ fps}$ (30.0 fps)
* **Stimulus Resolution:** $3840 \times 1920$ (2:1 ERP)

### 4.2 Timeline Discrepancy & Resolution
* In [`data/processed/viewport_trajectories.csv`](file:///d:/ml%20final%20project/data/processed/viewport_trajectories.csv), all 48 participant trajectory logs for `11.mp4` have timestamps starting at $t \approx 0.0\text{s}$ ($0.244\text{s}$) and ending at $t \approx 30.0\text{s}$ ($30.025\text{s}$).
* **Finding:** The dataset timestamps refer to the **trimmed 30-second stimulus timeline**, NOT the full 5-minute YouTube video timeline.
* Therefore, the trajectory timeline $[0.0\text{s}, 30.0\text{s}]$ corresponds to $[00:00:10, 00:00:40]$ in the downloaded YouTube file.

### 4.3 Visual Content & Gaze Correlation
* **Visual Scene:** During $[00:00:10, 00:00:40]$, the video portrays the Seeker VR field crew on the asphalt parking lot securing the telemetry payload beneath the deflated weather balloon. At $t = 60\text{s}$, the balloon is airborne at 1000 ft altitude.
* **Gaze Correlation:** The empirical mean pitch angle recorded across all 48 human participants watching Video 11 is **$-19.45^\circ$** (with minimum gaze down to $-81^\circ$). In `frame_10s_erp.jpg` through `frame_40s_erp.jpg`, the equipment and crew are situated directly beneath the camera at $-20^\circ$ to $-45^\circ$ elevation.

### 4.4 Audit Correction: Limitation of Temporal Alignment
> [!IMPORTANT]
> **Audit Finding on Temporal Alignment:**  
> The Phase 5.1 acquisition report claimed "alignment within $\pm 1$ frame". **This claim is unverified and scientifically unsustainable.**  
> 
> **Why:**
> 1. The original 2018 researchers did not publish the exact millisecond cut command used to extract `tmp\edgeofspace.avi`.
> 2. YouTube has re-encoded its media streams since 2018, potentially shifting initial GOP keyframe timestamps by several hundred milliseconds.
> 3. The downloaded video has an average frame rate of $29.97\text{ fps}$ ($30000/1001$) whereas the 2018 stimulus was stored at $30.00\text{ fps}$, accumulating approximately $1$ frame of drift over 30 seconds.
> 
> **Corrected Alignment Statement:**  
> **Alignment is verified at the semantic and behavioral level (~within 0.5 to 1.0 second). Exact millisecond-level or frame-for-frame synchronization cannot be guaranteed without the original 2018 uncompressed raw AVI file.**

---

## 5. Tile Geometry Verification (Item 4)

We mathematically verified the $4 \times 6$ spatial tiling configuration on a $3840 \times 1920$ ERP frame:

### 5.1 Spatial Grid Mathematics
* **Canvas Width:** $W = 3840\text{ pixels}$
* **Canvas Height:** $H = 1920\text{ pixels}$
* **Grid Layout:** $M = 4\text{ rows} \times N = 6\text{ columns} = 24\text{ tiles}$
* **Tile Width:** $W_{\text{tile}} = \frac{3840}{6} = \mathbf{640\text{ pixels}}$
* **Tile Height:** $H_{\text{tile}} = \frac{1920}{4} = \mathbf{480\text{ pixels}}$
* **Exact Divisibility:**
  $$3840 \pmod 6 = 0, \quad 1920 \pmod 4 = 0$$
* **Pixel Completeness:**
  $$24 \times (640 \times 480) = 24 \times 307,200 = 7,372,800\text{ pixels} \equiv 3840 \times 1920$$
  * Overlapping pixels: **0**
  * Missing pixels: **0**

### 5.2 Spherical Coordinate Bounding Box Formulation
For Equirectangular Projection, spherical coordinates map linearly:
$$\phi(x) = -180^\circ + \frac{x}{W} \times 360^\circ, \quad \theta(y) = +90^\circ - \frac{y}{H} \times 180^\circ$$

* **Azimuth Span per Tile ($\Delta\phi$):** $\frac{360^\circ}{6} = \mathbf{60.0^\circ}$
* **Elevation Span per Tile ($\Delta\theta$):** $\frac{180^\circ}{4} = \mathbf{45.0^\circ}$

#### Full 24-Tile Coordinate Reference Matrix:
| Row Index ($i$) | Elevation Range ($\theta$) | Col 0 ($-180^\circ \dots -120^\circ$) | Col 1 ($-120^\circ \dots -60^\circ$) | Col 2 ($-60^\circ \dots 0^\circ$) | Col 3 ($0^\circ \dots +60^\circ$) | Col 4 ($+60^\circ \dots +120^\circ$) | Col 5 ($+120^\circ \dots +180^\circ$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Row 0** ($y: 0\dots 479$) | **$+45^\circ \dots +90^\circ$ (Zenith)** | Tile 0 | Tile 1 | Tile 2 | Tile 3 | Tile 4 | Tile 5 |
| **Row 1** ($y: 480\dots 959$) | **$0^\circ \dots +45^\circ$ (Upper Eq.)** | Tile 6 | Tile 7 | Tile 8 | Tile 9 | Tile 10 | Tile 11 |
| **Row 2** ($y: 960\dots 1439$) | **$-45^\circ \dots 0^\circ$ (Lower Eq.)** | Tile 12 | Tile 13 | Tile 14 | Tile 15 | Tile 16 | Tile 17 |
| **Row 3** ($y: 1440\dots 1919$) | **$-90^\circ \dots -45^\circ$ (Nadir)** | Tile 18 | Tile 19 | Tile 20 | Tile 21 | Tile 22 | Tile 23 |

Every point on the sphere $(\phi, \theta)$ maps unambiguously to exactly one tile (or a boundary between two tiles).

---

## 6. Audit Summary: Confirmed Facts vs. Unverified Claims

| Audit Criterion | Verification Status | Evidence / Finding |
|---|---|---|
| **EAC Projection** | **CONFIRMED** | Stream format note is `mesh`; 3×2 cube face packing visually verified on raw frames. |
| **Spherical MP4 Box Metadata** | **ABSENT (CONFIRMED)** | MP4 box walk shows no `sv3d`/`st3d`/`proj`/`uuid` boxes; projection is bitstream-level. |
| **ERP Transformation Accuracy** | **CONFIRMED** | FFmpeg `v360=eac:equirect` produces exact $3840 \times 1920$ (2:1) output. Mean circular wrap delta is $1.79/255$. |
| **Zero Conversion Artifacts** | **CORRECTED (UNVERIFIED)** | Minor subpixel re-sampling smoothing and slight gradient variance at seams ($1.8\dots 2.8 / 255$) occur as mathematical byproducts. Visually seamless, but not mathematically zero. |
| **Visual Scene Match** | **CONFIRMED** | High-altitude balloon preparation matches Seeker VR content and 2018 paper descriptions. |
| **Temporal Alignment ($\pm 1$ Frame)** | **RETRACTED (UNVERIFIED)** | Cannot be guaranteed without the 2018 raw master AVI file. High-level alignment confirmed within ~0.5–1.0s. |
| **Dataset Timeline Meaning** | **CONFIRMED** | Dataset trajectory timestamps $[0.0\text{s}, 30.0\text{s}]$ refer to the trimmed stimulus, not the YouTube timeline. |
| **Tile Grid Math ($4 \times 6$)** | **CONFIRMED** | 24 tiles of $640 \times 480$ px covering $60^\circ \times 45^\circ$ cleanly divide the $3840 \times 1920$ canvas with 0 residual pixels. |
| **Asset Preservation** | **CONFIRMED** | `data/videos/11.mp4` remains completely untouched in its original state. |

---

## 7. Recommendation for Phase 5.2

The video asset [`data/videos/11.mp4`](file:///d:/ml%20final%20project/data/videos/11.mp4) is **approved for Phase 5.2 (Spatial Tile Grid Partitioning and Coordinate Mapping)** under the following procedural workflow:

1. **Extract Canonical 30-Second Stimulus:**
   Convert the specific $[00:00:10, 00:00:40.033]$ segment from EAC to canonical ERP:
   ```bash
   ffmpeg -ss 00:00:10 -i "data/videos/11.mp4" -t 30.033333 -vf "v360=eac:equirect:w=3840:h=1920:interp=cube" -c:v libx264 -crf 18 -preset slow -r 30 -an "data/videos/11_erp_30s.mp4"
   ```
2. **Implement Tile Grid Partitioner:**
   Implement spatial grid cropping ($4 \times 6$, 24 tiles) based on the verified coordinate matrix in Section 5.2.
3. **Preserve Master Files:**
   Retain `data/videos/11.mp4` as the untouched reference archive.

*Audit completed by Antigravity Agent. No source files, datasets, or models were modified.*
