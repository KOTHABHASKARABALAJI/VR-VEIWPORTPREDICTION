# AVTrack360 Dataset Specification & Preprocessing Workflow

**Project:** Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming  
**Dataset Reference:** Fremerey et al., *"AVTrack360: An Open Dataset and Software Recording People's Head Rotations Watching 360° Videos on an HMD"*, ACM MMSys 2018.

---

## 1. Dataset Dimensions & Summary

| Parameter | Specification | Verification Source |
|---|---|---|
| **Human Subjects** | Exactly 48 subjects (IDs 2 through 49) | `data/raw/Head_rotation/*.json` |
| **Omnidirectional Videos** | Exactly 20 distinct videos (`1.mp4` to `20.mp4`) | `data/processed/unique_videos.csv` |
| **Total Trajectories** | Exactly **960 trajectories** ($48 \times 20$) | `data/processed/trajectory_summary.csv` |
| **Raw Orientation Records** | Exactly **284,247 timestamped records** ($t$, pitch, yaw, roll) | `data/processed/combined_viewport_data.csv` |
| **Missing / Null Values** | Exactly **0 missing values** across all columns | Verified via `verify_csv.py` |
| **Sampling Frequency** | Quasi-uniform at **~10 Hz** ($\Delta t = 100.91\text{ ms}$) | `data/processed/viewport_movement_analysis.csv` |
| **HMD Hardware** | HTC Vive Head-Mounted Display | Recorded via Whirligig player integration |

---

## 2. Subject-Disjoint Dataset Partitioning

To completely eliminate subject-level data leakage, dataset splitting was performed strictly by subject ID:

```text
48 Total Subjects
├── Training Set (68.75%): 33 Subjects (186,176 samples)
│   └── IDs: 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34
├── Validation Set (14.59%): 7 Subjects (39,506 samples)
│   └── IDs: 35, 36, 37, 38, 39, 40, 41
└── Test Set (16.66%): 8 Subjects (45,125 samples)
    └── IDs: 42, 43, 44, 45, 46, 47, 48, 49
```

* **Total Supervised Samples ($L=10, H=5$):**  
  $$186,176 + 39,506 + 45,125 = \mathbf{270,807\text{ samples}}$$
* **Mathematical Sample Conservation:**
  $$284,247 - (960 \times 14) = 284,247 - 13,440 = \mathbf{270,807\text{ samples}}$$
  Each trajectory loses exactly $L + H - 1 = 10 + 5 - 1 = 14$ boundary samples during sliding window sequence construction.

---

## 3. Seven Engineered Features & Target Vector

For every time step $t$, a 7-dimensional causal feature vector is extracted:

| # | Feature Name | Mathematical Definition | Physical Interpretation |
|---|---|---|---|
| 1 | `sin_yaw` | $\sin(\text{radians}(\text{yaw}_t))$ | Azimuth unit circle coordinate $Y$ |
| 2 | `cos_yaw` | $\cos(\text{radians}(\text{yaw}_t))$ | Azimuth unit circle coordinate $X$ |
| 3 | `pitch` | Spherical elevation angle | Pitch angle in degrees $[-90^\circ, +90^\circ]$ |
| 4 | `yaw_velocity` | $\text{atan2}(\sin(\Delta\text{yaw}), \cos(\Delta\text{yaw})) / \Delta t$ | Causal angular velocity along azimuth ($\text{deg/s}$) |
| 5 | `pitch_velocity` | $(\text{pitch}_t - \text{pitch}_{t-1}) / \Delta t$ | Causal angular velocity along elevation ($\text{deg/s}$) |
| 6 | `yaw_acceleration` | $(\text{yaw\_vel}_t - \text{yaw\_vel}_{t-1}) / \Delta t$ | Causal angular acceleration along azimuth ($\text{deg/s}^2$) |
| 7 | `pitch_acceleration` | $(\text{pitch\_vel}_t - \text{pitch\_vel}_{t-1}) / \Delta t$ | Causal angular acceleration along elevation ($\text{deg/s}^2$) |

* **Target Vector ($y_t$):**  
  $$y_t = \left[ \sin(\text{yaw}_{t+H}),\ \cos(\text{yaw}_{t+H}),\ \frac{\text{pitch}_{t+H} - \mu_{\text{pitch\_train}}}{\sigma_{\text{pitch\_train}}} \right]$$
  where $H = 5\text{ frames}$ ($0.5\text{ seconds}$ ahead).

