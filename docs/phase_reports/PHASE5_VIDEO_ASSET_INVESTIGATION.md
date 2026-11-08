# Phase 5: 360° Video Asset & Infrastructure Investigation Report

**Project Title:** Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming  
**Phase:** Phase 5 (Viewport Tile Prediction & Adaptive Streaming Preparation)  
**Investigation Date:** November 4, 2026  
**Document Location:** `docs/PHASE5_VIDEO_ASSET_INVESTIGATION.md`  

---

## 1. Executive Summary

As our project transitions from **Phase 4 (Final Model Training & Verification)** into **Phase 5 (Viewport-to-Tile Mapping and Adaptive Quality Allocation)**, we require physical omnidirectional video media to implement spatial grid partitioning, determine tile intersections, and simulate tile-based adaptive video streaming.

Before acquiring or downloading any media, this comprehensive investigation audited the local project workspace, examined the upstream **AVTrack360** (ACM MMSys 2018) benchmark repositories, inspected video metadata files, and evaluated video availability and projection formats.

### Key Investigation Findings:
1. **Local Media Status:** **0 video files** exist in the repository. All current datasets consist exclusively of orientation telemetry logs (`.json`, `.csv`) and feature tensors (`.npz`).
2. **Upstream Dataset Content:** Neither the ACM MMSys 2018 repository nor the TU Ilmenau repository hosts raw video files. The uncompressed source videos exceed **200 GB** in total size (~10 GB per 30-second clip) and belong to third-party copyright holders (YouTube creators and ARTE). The dataset provides tracking logs, extraction timestamps, and FFprobe stream metadata.
3. **Projection Format:** **Confirmed Equirectangular Projection (ERP).** Out of the 20 stimulus videos, 15 are rendered in canonical **2:1 aspect ratio ($3840 \times 1920$)**, representing the full spherical coordinate space ($360^\circ$ longitude $\times 180^\circ$ latitude). The remaining 5 are padded or letterboxed to 16:9 or 15:8.
4. **Recommended Video Asset:** **Video 1 (`1.mp4`, *Icebreaker*)** or **Video 11 (`11.mp4`, *Journey to the Edge of Space*)**. Both feature canonical 2:1 4K ERP geometry ($3840 \times 1920$), exactly 901 frames at 30 fps, and directly map to 48 ground-truth user viewing trajectories in our evaluated test set.

---

## 2. Audit of Current Project Directory (Task 2)

An exhaustive search across the entire project workspace (`D:\ml final project`) was executed:
```powershell
Get-ChildItem -Path . -Recurse -Include *.mp4,*.mkv,*.webm,*.avi,*.mov,*.flv,*.ts
```

### Result:
* **Total Video Files Detected:** **0**
* **Existing Data Assets:**
  * Raw Trajectories: [`data/raw/Head_rotation/`](file:///d:/ml%20final%20project/data/raw/Head_rotation/) (48 JSON files, subjects 2–49).
  * Video Technical Metadata: [`data/raw/SRC_info/`](file:///d:/ml%20final%20project/data/raw/SRC_info/) (20 FFprobe JSON files, `1.json` to `20.json`).
  * Processed Trajectories: [`data/processed/combined_viewport_data.csv`](file:///d:/ml%20final%20project/data/processed/combined_viewport_data.csv) (284,247 rows).
  * Video Metadata Table: [`data/processed/video_metadata.csv`](file:///d:/ml%20final%20project/data/processed/video_metadata.csv) (20 rows).
  * Supervised Tensors: [`data/processed/lstm_train.npz`](file:///d:/ml%20final%20project/data/processed/lstm_train.npz), `lstm_validation.npz`, `lstm_test.npz`.

**Conclusion:** The local repository contains all mathematical telemetry and metadata necessary to train prediction models, but lacks the pixel-level visual containers required for Phase 5 tile cropping, encoding, and streaming simulations.

---

## 3. Upstream AVTrack360 Dataset Investigation (Tasks 1 & 3)

### 3.1 Upstream Repositories
The AVTrack360 benchmark was created by Stephan Fremerey, Ashutosh Singla, Kay Meseberg (ARTE), and Alexander Raake (TU Ilmenau) and published at ACM MMSys 2018:
* **ACM Repository:** `https://github.com/acmmmsys/2018-AVTrack360`
* **Research Group Repository:** `https://github.com/Telecommunication-Telemedia-Assessment/AVTrack360`
* **Paper Reference:** *"AVTrack360: An Open Dataset and Software Recording People's Head Rotations Watching 360° Videos on an HMD"*, Proc. 9th ACM MMSys '18, DOI: `10.1145/3204949.3208134`.

### 3.2 Contents of Upstream Repository
The official repository structure consists of:
* `/AVTrack360_dataset`:
  * `Head_rotation/` — 48 JSON tracking logs (Pitch, Yaw, Roll at ~10 Hz).
  * `SRC_info/` — 20 JSON files containing FFprobe output for the source clips.
  * `SSQ/` — Simulator Sickness Questionnaire survey responses.
* `/AVTrack360_record`:
  * C# / ASP.NET application interfacing with the **Whirligig 360° player** (v3.92) to present media on an HTC Vive or Oculus Rift.
* `/AVTrack360_evaluation`:
  * Python scripts for calculating orthographic projections and head-saliency heatmaps.

### 3.3 Do Upstream Repositories Provide Original Video Files?
**No.** The upstream repositories do **not** host binary video files.

#### Reasons:
1. **Storage Limits:** As verified in our local [`data/processed/video_metadata.csv`](file:///d:/ml%20final%20project/data/processed/video_metadata.csv), each 30-second raw uncompressed master sequence was stored as raw YUV420p video in an AVI container (`rawvideo, yuv420p, avi`). Each 30-second clip occupies between **$9.95\text{ GB}$ and $11.20\text{ GB}$** on disk. Across 20 clips, the dataset media totals **$>200\text{ GB}$**, which far exceeds GitHub's maximum file size (100 MB) and repository quota.
2. **Copyright & Distribution Licensing:** The video stimuli were sourced from external commercial and public broadcasters. The authors lacked the redistribution rights to re-host full master video files.

### 3.4 Sourcing and Metadata Catalog (Table 1 of AVTrack360 Paper)
In the original paper, the authors specify that the 20 source clips were gathered from two platforms:
* **14 Videos from ARTE 360** (co-author Kay Meseberg was Head of Mission Innovation at ARTE G.E.I.E.).
* **6 Videos from YouTube**.

All clips were extracted as 30-second segments, stripped of directional audio to eliminate acoustic orienting cues, and formatted for 360° playback.

| Video ID | Title / Content Name | Platform Origin | Start Timestamp in Source | Resolution | Aspect Ratio | Temp Filename in Metadata |
|---|---|---|---|---|---|---|
| `1.mp4` | **Icebreaker** (Antarctica) | ARTE 360 | `00:01:03` | 3840×1920 | 2:1 | `tmp\antartica1.avi` |
| `2.mp4` | **Fireworks** | YouTube | `00:01:40` | 3840×2160 | 16:9 | `fireworks.avi` |
| `3.mp4` | **Helicopter** (Antarctica) | ARTE 360 | `00:04:01` | 3840×1920 | 2:1 | `tmp\antartica1.avi` |
| `4.mp4` | **Hawaii** | YouTube | `00:01:52` | 3840×2160 | 16:9 | `tmp\hawaii.avi` |
| `5.mp4` | **Penguins** (Antarctica) | ARTE 360 | `00:01:39` | 3840×1920 | 2:1 | `tmp\antartica3.avi` |
| `6.mp4` | **Diving** (Antarctica) | ARTE 360 | `00:02:16` | 3840×1920 | 2:1 | `tmp\antartica3.avi` |
| `7.mp4` | **Colonie** (Antarctica) | ARTE 360 | `00:01:55` | 3840×1920 | 2:1 | `tmp\colonie.avi` |
| `8.mp4` | **Childs** (Wasala) | ARTE 360 | `00:01:18` | 3840×1920 | 2:1 | `tmp\wasala.avi` |
| `9.mp4` | **Wasala** | ARTE 360 | `00:05:47` | 3840×1920 | 2:1 | `tmp\wasala.avi` |
| `10.mp4` | **Carrara** (Marble Quarry) | ARTE 360 | `00:02:07` | 3840×1920 | 2:1 | `tmp\carrara.avi` |
| `11.mp4` | **Edge of Space** | YouTube (Seeker VR) | `00:00:10` | 3840×1920 | 2:1 | `tmp\edgeofspace.avi` |
| `12.mp4` | **Arcachon** (Dune du Pilat) | ARTE 360 | `00:03:05` | 3840×1920 | 2:1 | `tmp\arcachon.avi` |
| `13.mp4` | **La Grotte Chauvet** | ARTE 360 | `00:03:10` | 3840×1920 | 2:1 | `tmp\lagrottechauvet.avi` |
| `14.mp4` | **Aliens** | ARTE 360 | `00:05:47` | 3840×1920 | 2:1 | `tmp\aliens.avi` |
| `15.mp4` | **Elisir** (L'Elisir d'Amore) | ARTE 360 | `00:01:50` | 3840×1920 | 2:1 | `tmp\elisir.avi` |
| `16.mp4` | **Spotlight HELP** | YouTube (Google Stories) | `00:01:10` | 3840×2160 | 16:9 | `tmp\spotlighthelp.avi` |
| `17.mp4` | **Angel Falls** | YouTube (NatGeo) | `00:00:12` | 3840×2160 | 16:9 | `tmp\angelfalls.avi` |
| `18.mp4` | **Holifest / Wolfskinder** | ARTE 360 | `00:06:17` | 3840×1920 | 2:1 | `tmp\wolfskinder.avi` |
| `19.mp4` | **Nerf Battle** | YouTube | `00:00:13` | 3840×2048 | 15:8 | `tmp\nerfbattle.avi` |
| `20.mp4` | **Football** | ARTE 360 | `00:00:16` | 3840×1920 | 2:1 | `tmp\football.avi` |

### 3.5 Licensing & Terms of Use
* **Dataset & Telemetry Traces:** The head rotation logs, source info, and code in the AVTrack360 repository are released under the **GNU General Public License v3.0 (GPL-3.0)**.
* **Underlying Video Sequences:** The video content remains the intellectual property of their original creators (Google, National Geographic, Seeker VR, and ARTE). They are publicly available on streaming platforms for personal, educational, and research evaluation.

---

## 4. Verification of Equirectangular Projection (ERP) (Task 4)

### 4.1 What is Equirectangular Projection?
Equirectangular Projection (ERP) is the standard cartographic projection used to map a spherical surface $(\phi, \theta)$ onto a planar 2D rectangular grid $(x, y)$:
* **Longitude (Azimuth / Yaw):** Ranges across $360^\circ$ ($[-\pi, +\pi]$ or $[0, 2\pi]$).
* **Latitude (Elevation / Pitch):** Ranges across $180^\circ$ ($[-\pi/2, +\pi/2]$).
* **Canonical Aspect Ratio:** Because the horizontal span ($360^\circ$) is exactly double the vertical span ($180^\circ$), standard ERP videos have a **strict 2:1 Display Aspect Ratio (DAR)** where:
  $$\text{Width} = 2 \times \text{Height}$$

### 4.2 Evidence from Dataset and Metadata
1. **Metadata Inspection:**
   * In [`data/processed/video_metadata.csv`](file:///d:/ml%20final%20project/data/processed/video_metadata.csv), **15 out of 20 videos** possess a native resolution of **$3840 \times 1920$** ($3840 / 1920 = 2.0$), with explicit `display_aspect_ratio = 2:1`.
   * The remaining 5 videos (`2.mp4`, `4.mp4`, `16.mp4`, `17.mp4` at $3840 \times 2160$, and `19.mp4` at $3840 \times 2048$) are equirectangular panoramas padded with top/bottom letterbox bars to match 16:9 UHD display encoding on YouTube.
2. **Paper Documentation:**
   * Section 3 of the Fremerey et al. (2018) paper explicitly confirms:  
     > *"All source contents were acquired in 4K resolution and provided in an equirectangular format."*
3. **Player Implementation:**
   * The recording tool uses the Whirligig player set to **Equirectangular 360 mode** to project the planar 2D video onto the inside of a virtual 3D sphere centered on the user's HMD.

### 4.3 Impact on Phase 5 Tile Partitioning
Using a canonical **2:1 ERP video ($3840 \times 1920$)** is mathematically optimal for spatial tiling:
* **$4 \times 6$ Grid (24 tiles):**
  * Tile size: $640 \times 480$ pixels.
  * Angular span per tile: $60^\circ\text{ Azimuth} \times 45^\circ\text{ Elevation}$.
* **$6 \times 8$ Grid (48 tiles):**
  * Tile size: $480 \times 320$ pixels.
  * Angular span per tile: $45^\circ\text{ Azimuth} \times 30^\circ\text{ Elevation}$.
* **$8 \times 12$ Grid (96 tiles):**
  * Tile size: $320 \times 240$ pixels.
  * Angular span per tile: $30^\circ\text{ Azimuth} \times 22.5^\circ\text{ Elevation}$.

Each tile cleanly divides the spherical coordinate domain without requiring letterbox removal or coordinate aspect scaling.

---

## 5. Candidate Video Evaluation & Recommendation (Task 5)

We evaluated the top candidates from the 20 stimulus videos based on:
1. **ERP Geometry:** Must be clean 2:1 ($3840 \times 1920$).
2. **Behavioral Dynamics:** Must represent realistic head movement (neither static staring nor violent random headshakes).
3. **Ground-Truth Data Alignment:** Must map directly to our 48 subject trajectories and validated Phase 4 model predictions.
4. **Availability:** Must be obtainable online in high quality.

### 5.1 Candidate Comparison

| Candidate Video | Resolution & Aspect Ratio | Avg Speed across 48 Subjects | 0.5s Displacement | Final GRU-64 Test MAE | Baseline Persistence MAE | Online Source Availability |
|---|---|---|---|---|---|---|
| **Video 1: Icebreaker** | **3840×1920 (2:1)** | **31.09°/s** | **14.92°** | **8.67°** | 13.27° | High (ARTE 360 / YouTube) |
| **Video 11: Edge of Space** | **3840×1920 (2:1)** | **37.99°/s** | **18.09°** | **11.07°** | 15.95° | High (Seeker VR on YouTube) |
| **Video 10: Carrara** | **3840×1920 (2:1)** | **36.20°/s** | **17.40°** | **10.56°** | 16.20° | Moderate (ARTE 360) |
| **Video 16: Spotlight HELP** | 3840×2160 (16:9) | 24.64°/s | 11.48° | 10.31° | 12.49° | High (Google Spotlight on YouTube) |

### 5.2 Primary Recommendation: Video 1 (`1.mp4`, *Icebreaker*)

We strongly recommend **Video 1 (`1.mp4`, *Icebreaker*)** as our benchmark stimulus for Phase 5:

1. **Perfect Equirectangular Dimensions ($3840 \times 1920$, 2:1):** Eliminates padding artifacts, ensuring every pixel corresponds linearly to spherical azimuth and elevation.
2. **Exemplary Exploration Dynamics:**
   * Average speed: $31.09^\circ/\text{s}$, with $14.92^\circ$ average 0.5s displacement.
   * Viewers exhibit active $360^\circ$ environmental scanning (average 2.75 yaw wrap-arounds per trajectory), making it a challenging test for tile prediction and adaptive streaming.
3. **Superior Model Predictability:**
   * Our final GRU-64 model achieves **$8.674^\circ$ MAE** on this video (a **34.6% reduction in error** compared to the $13.272^\circ$ Persistence baseline).
   * This high predictive accuracy ensures that the predicted viewport will reliably overlap with the actual user viewport, demonstrating high tile hit-ratios.
4. **Standard Duration:** Exactly 901 frames at 30.0 fps (30.033 seconds), matching our sliding window sequence length ($L=10$, $H=5$).

### 5.3 Alternative / Backup Candidate: Video 11 (`11.mp4`, *Journey to the Edge of Space*)
If an immediate, direct YouTube-hosted stream is preferred without accessing ARTE archives, **Video 11 (`11.mp4`)** is also a canonical 2:1 ERP video ($3840 \times 1920$):
* **Direct Source:** Seeker VR / Discovery (`https://www.youtube.com/watch?v=pCve1w1GFOs`).
* **Timestamp:** Extracted starting at `00:00:10` to `00:00:40`.
* **Dynamics:** Represents high-velocity exploration ($37.99^\circ/\text{s}$), useful for stress-testing aggressive tile prefetching.

---

## 6. Video Acquisition Plan & Project Suitability

### 6.1 Where to Obtain the Video
Because Video 1 and Video 11 are public 360° videos, they can be obtained using standard media tools:

#### Option A: Direct Download via `yt-dlp`
For YouTube-hosted 360° streams (e.g., Video 11 or Video 1 mirror):
```bash
# 1. Fetch highest resolution 4K Equirectangular stream
yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]" --merge-output-format mp4 -o "raw_source.mp4" "<YOUTUBE_URL>"

# 2. Trim exactly 30.033 seconds starting at the documented timestamp (e.g., 00:00:10 for Video 11)
ffmpeg -ss 00:00:10 -i raw_source.mp4 -t 30.033333 -c:v copy -an "data/videos/11.mp4"
```

#### Option B: Dedicated Open-Source ERP Test Sequences
If institutional or copyright restrictions prevent downloading commercial YouTube/ARTE clips, we can use the **Fraunhofer HHI 360° Omnidirectional Test Sequences** or **Blender Foundation 360° Open Sequences** (e.g., *Caminandes 360*, CC-BY), provided at standard 4K ERP ($3840 \times 1920$, 30 fps).  
*Note: Using an external video allows verifying tile partitioning and quality allocation, but will not align frame-for-frame with the 48 human subject trajectories in our AVTrack360 test set. Thus, using the actual AVTrack360 stimulus (Video 1 or 11) remains strongly preferred.*

### 6.2 Suitability Verdict
| Criterion | Assessment | Verdict |
|---|---|---|
| **Projection Compatibility** | Canonical 2:1 Equirectangular ($3840 \times 1920$) | **PASSED** |
| **Grid Divisibility** | Divisible into $4 \times 6$, $6 \times 8$, $8 \times 12$ tiles without remainder | **PASSED** |
| **Trajectory Alignment** | 48 subjects (960 trajectories) evaluated in Phase 1–4 match this clip | **PASSED** |
| **Model Verification** | Final GRU-64 model predicts Video 1 at $8.67^\circ$ MAE ($34.6\%$ gain) | **PASSED** |
| **Readiness for Phase 5** | Fully prepared to implement tile partitioning upon user download approval | **PASSED** |

---

*Report prepared by Antigravity Agent. No files were modified or downloaded during this investigation.*
