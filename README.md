# Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## 1. Project Title
**Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming**

---

## 2. Problem Statement
In conventional 2D video streaming, the entire video frame is displayed on a flat screen and viewed by the user. In contrast, 360-degree Virtual Reality (VR) video is captured as an omnidirectional spherical panorama, stored in Equirectangular Projection (ERP) format at ultra-high resolutions (such as 4K: 3840x1920 / 3840x2160, or 8K).

Streaming complete 360° spheres at ultra-high resolutions requires immense network throughput (often 50 to 100+ Mbps), which places heavy demands on residential and mobile networks. However, human visual physiology and Head-Mounted Displays (HMDs, e.g., HTC Vive, Oculus Quest) restrict the user's instantaneous Field of View (FoV) to approximately 90° to 110° diagonally. At any given moment, the viewer observes only **15% to 20%** of the full spherical canvas (the **viewport**), while the remaining **80% to 85%** lies outside their field of vision.

Transmitting the non-visible panorama at full ultra-high quality consumes substantial network bandwidth. To address this challenge, modern 360° video delivery architectures propose **tile-based adaptive streaming**:
1. The 360° equirectangular video canvas is segmented into an $M \times N$ spatial grid of independent rectangular tiles.
2. A machine learning model predicts the user's future head orientation short intervals in advance (e.g., 0.5 seconds ahead).
3. The streaming client can then request tiles falling inside the predicted viewport at higher bitrates, while background or peripheral tiles can be requested at lower quality or omitted.
4. Accurate head motion prediction provides the foundational mechanism for dynamic tile quality allocation, enabling adaptive streaming systems to prioritize network resources where the user is looking.

> **Scope Clarification:** This repository implements the **machine learning viewport prediction pipeline** (data harvesting, kinematic feature engineering, recurrent sequence modeling, and spherical error benchmarking). The subsequent adaptive tile-quality allocation and dynamic bitrate scheduling mechanisms are planned extensions for future stages of this research.

---

## 3. Project Objectives
- **Accurate Viewport Trajectory Forecasting:** Predict human head orientation (azimuth/yaw and elevation/pitch) 0.5 seconds ahead (5 frames at 10 Hz) using a 1.0-second historical trajectory (10 frames).
- **Trigonometric Continuous Formulation:** Eliminate angular wrap-around boundary discontinuities ($\pm 180^\circ$) by mapping yaw into continuous 2D Cartesian coordinates $[\sin(\text{yaw}), \cos(\text{yaw})]$.
- **Kinematic Feature Engineering:** Extract first-order (angular velocity) and second-order (angular acceleration) motion dynamics using phase-unwrapped trajectories.
- **Subject-Independent Benchmark Evaluation:** Partition subjects disjointly (33 training, 7 validation, 8 test) to guarantee zero data leakage between user movement patterns.
- **Multi-Model Architecture Comparison:** Rigorously compare LSTM and GRU recurrent neural networks under identical conditions with early stopping and great-circle spherical distance evaluation.
- **Predictive Foundation for Adaptive Streaming:** Establish benchmarked head motion models to drive planned downstream spatial tile selection and tile bitrate allocation algorithms.

---

## 4. Dataset Information (AVTrack360)
The project utilizes the public **AVTrack360** (Audio-Visual Tracking in 360° Video) benchmark dataset (ACM MMSys 2018):
- **Participants:** Exactly 48 distinct human participants (identified as subjects 2 through 49).
- **Videos:** Exactly 20 diverse omnidirectional panoramic videos (1.mp4 through 20.mp4), each ~30 seconds in duration at 30 fps with resolutions of 3840x1920 or 3840x2160.
- **Viewing Trajectories:** 48 subjects x 20 videos = Exactly **960 unique trajectories**.
- **Data Points:** **284,247 timestamped orientation records** ($t$, pitch, yaw, roll) captured using HTC Vive HMDs.
- **Sampling Frequency:** ~10 Hz (mean sampling interval $\Delta t = 100.91\text{ ms}$).

---

## 5. Data Preprocessing Workflow
The data processing workflow implemented across the repository consists of:
1. **Raw Trajectory Harvesting:** Parses 48 individual participant JSON files (`data/raw/Head_rotation/`) extracting timestamped pitch, yaw, and roll logs into a consolidated tabular dataset (`data/processed/combined_viewport_data.csv`).
2. **Metadata Extraction:** Gathers technical video parameters from 20 source JSON files (`data/raw/SRC_info/`) via FFprobe (`data/processed/video_metadata.csv`).
3. **Data Cleaning & Sanity Auditing:** Validates that all 284,247 records contain complete numeric values with 0 missing values across all columns.
4. **Trajectory Segmentation:** Reconstructs continuous user sessions across 960 trajectories using timestamp resets and video transitions (`data/processed/viewport_trajectories.csv`).
5. **Yaw Phase Unwrapping:** Unwraps circular yaw angles across the $\pm 180^\circ$ boundary before calculating derivatives, preventing artificial velocity spikes when yaw wraps between $+180^\circ$ and $-180^\circ$.
6. **Kinematic Differentiation:** Computes angular velocities ($\text{deg/s}$) and accelerations ($\text{deg/s}^2$) for both yaw and pitch:
   $$\text{vel}_t = \frac{\theta_t - \theta_{t-1}}{\Delta t}, \quad \text{acc}_t = \frac{\text{vel}_t - \text{vel}_{t-1}}{\Delta t}$$

---

## 6. Seven Input Features & Motion Dynamics
For every timestep $t$, a 7-dimensional feature vector is constructed:

| # | Feature Name | Description | Units / Range |
|---|---|---|---|
| 1 | `sin_yaw` | $\sin(\text{radians}(\text{yaw}_t))$ | $[-1.0, +1.0]$ |
| 2 | `cos_yaw` | $\cos(\text{radians}(\text{yaw}_t))$ | $[-1.0, +1.0]$ |
| 3 | `pitch` | Spherical elevation angle | Degrees ($[-90^\circ, +90^\circ]$) |
| 4 | `yaw_vel` | Angular velocity along unwrapped azimuth | $\text{deg/s}$ |
| 5 | `pitch_vel` | Angular velocity along elevation | $\text{deg/s}$ |
| 6 | `yaw_acc` | Angular acceleration along unwrapped azimuth | $\text{deg/s}^2$ |
| 7 | `pitch_acc` | Angular acceleration along elevation | $\text{deg/s}^2$ |

### Why Roll is Excluded
In VR headsets, roll represents lateral tilting of the head towards the shoulders. In 360° viewing, lateral tilting is minimal (mean roll in the dataset is $0.51^\circ$, with 99% of samples within $[-14^\circ, +15^\circ]$). Because the geometric center of the viewport on a sphere is determined entirely by azimuth (yaw) and elevation (pitch), roll does not change the viewport center coordinates. Excluding roll reduces input dimensionality without sacrificing predictive fidelity.

### Why Trigonometric Encoding is Essential for Yaw
Yaw is periodic on the unit circle: $-180^\circ$ and $+180^\circ$ represent the exact same physical head orientation (looking directly backwards). If raw angles are fed to neural networks:
- A user rotating slightly from $+179^\circ$ to $-179^\circ$ makes a tiny physical turn of only $2^\circ$.
- As raw numbers, the angle jumps by $358^\circ$, which under Mean Squared Error creates an artificial penalty: $(358)^2 = 128,164$.
- This causes gradient spikes and destabilizes backpropagation.

By encoding yaw into $[\sin(\text{yaw}), \cos(\text{yaw})]$, the Euclidean distance between $+179^\circ$ and $-179^\circ$ is only $\approx 0.035$, preserving physical continuity. At inference time, predicted yaw is reconstructed unambiguously via the four-quadrant inverse tangent:
$$\text{pred\_yaw} = \text{atan2}(\text{pred\_sin}, \text{pred\_cos}) \times \frac{180^\circ}{\pi}$$

---

## 7. Model Architectures & Sequential Formulation
- **Input Sequence Length ($L$):** 10 frames (1.0 second history at 10 Hz). Autocorrelation analysis proves position autocorrelation remains high ($r = 0.86\text{--}0.99$) and velocity autocorrelation remains positive ($r = 0.26$) over 1.0s, supplying necessary momentum context.
- **Prediction Horizon ($k$):** 5 frames ahead (0.5 seconds into the future), matching typical video segment buffering and network round-trip time (RTT).
- **Target Vector ($y$):** Shape $(N, 3)$ representing $[\sin(\text{future\_yaw}), \cos(\text{future\_yaw}), \text{norm\_future\_pitch}]$.
- **Data Splitting (Subject-Independent):**
  - **Training Set:** 33 subjects (186,176 samples, 68.75%)
  - **Validation Set:** 7 subjects (39,506 samples, 14.59%)
  - **Test Set:** 8 subjects (45,125 samples, 16.66%)
  - Strict disjoint partitioning prevents subject-level data leakage.
- **Normalization:** Z-score normalization computed strictly on the training set:
  $$\hat{X} = \frac{X - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

---

## 8. Model Comparison Experiments
Four deep recurrent neural network architectures were implemented, trained, and compared under strictly identical conditions:

1. **Model A (LSTM-64 Baseline):**
   `Input (10, 7) -> LSTM(64 units) -> Dense(32, ReLU) -> Dense(3, Linear)` (20,835 parameters)
2. **Model B (LSTM-128):**
   `Input (10, 7) -> LSTM(128 units) -> Dense(32, ReLU) -> Dense(3, Linear)` (73,923 parameters)
3. **Model C (GRU-64 — Selected Model):**
   `Input (10, 7) -> GRU(64 units) -> Dense(32, ReLU) -> Dense(3, Linear)` (16,035 parameters)
4. **Model D (LSTM with Dropout):**
   `Input (10, 7) -> LSTM(64 units) -> Dropout(p=0.2) -> Dense(32, ReLU) -> Dense(3, Linear)` (20,835 parameters)

### Training Hyperparameters
- **Framework:** PyTorch (`torch`, `torch.nn`)
- **Optimizer:** Adam (learning rate $\alpha = 0.001$)
- **Loss Function:** Mean Squared Error (MSE)
- **Batch Size:** 256
- **Maximum Epochs:** 100 with Early Stopping ($\text{patience} = 10$ epochs monitoring validation loss)
- **Weight Restoration:** Model weights restored from the epoch with lowest validation loss.
- **Random Seed:** Fixed at 42 across Python, NumPy, and PyTorch.

---

## 9. Evaluation Metrics & Experimental Results

### Evaluation Metrics
1. **Circular Yaw MAE & RMSE:** Absolute error accounting for circular wrap-around ($|\Delta\text{yaw}| \bmod 360^\circ$).
2. **Pitch MAE & RMSE:** Elevation absolute error in degrees.
3. **Great-Circle Spherical Angular Error (Orthodromic Distance):** True angular separation on the 3D unit sphere between predicted viewport center $(\text{yaw}_1, \text{pitch}_1)$ and actual viewport center $(\text{yaw}_2, \text{pitch}_2)$:
   $$\Delta\sigma = \arccos\Big(\sin(\text{pitch}_1)\sin(\text{pitch}_2) + \cos(\text{pitch}_1)\cos(\text{pitch}_2)\cos(\text{yaw}_1 - \text{yaw}_2)\Big)$$

### Validation Set Comparison (All 4 Architectures)
Evaluated on 39,506 validation sequences across 7 unseen subjects (recorded in `results/metrics/model_comparison.csv` and `results/metrics/model_comparison_report.txt`):

| Model | Best Epoch | Training Time (s) | Val Loss (MSE) | Yaw MAE (deg) | Yaw RMSE (deg) | Pitch MAE (deg) | Pitch RMSE (deg) | Spherical MAE (deg) | Spherical Median (deg) | Spherical 90th (deg) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model C (GRU-64) [Selected]** | **6** | 105.69 | **0.053500** | **8.2182°** | **12.3028°** | 4.2132° | **6.4094°** | **9.7289°** | **7.1616°** | **20.5276°** |
| Model A (LSTM-64 baseline) | 9 | **98.45** | 0.053511 | 8.3083° | 12.3209° | **4.1462°** | 6.4116° | 9.7504° | 7.1889° | 20.6641° |
| Model B (LSTM-128) | 8 | 159.13 | 0.054109 | 8.3992° | 12.3930° | 4.2508° | 6.4358° | 9.8698° | 7.3099° | 20.6774° |
| Model D (LSTM with Dropout) | 21 | 168.99 | 0.053813 | 8.3085° | 12.3504° | 4.1775° | 6.4316° | 9.7736° | 7.1778° | 20.6922° |

### Model Selection Rationale & Test Set Performance Nuance
- **Selection Decision:** Following standard machine learning protocol to prevent data leakage and test-set snooping, **Model C (GRU-64)** was selected based strictly on validation set performance, where it achieved the lowest validation Mean Spherical Error (**9.7289°**) and lowest validation MSE loss (**0.053500**). In addition, GRU-64 features 23% fewer parameters than LSTM-64 (16,035 vs. 20,835), providing computational efficiency for real-time inference.
- **Test Set Evaluation:** When evaluated on the independent test set (45,125 samples across 8 unseen subjects):
  - **Selected Model C (GRU-64):** Achieved Spherical MAE = **9.5525°** (recorded in `results/metrics/best_model_test_metrics.txt`).
  - **Baseline Model A (LSTM-64):** Achieved Spherical MAE = **9.5329°** (recorded in `results/metrics/lstm_training_report.txt`).
- **Performance Comparison:** The baseline LSTM-64 model achieved marginally lower test spherical error (**9.5329° vs. 9.5525°**, a slight difference of 0.0196° or ~0.21%). Both models exhibit virtually equivalent generalization on unseen subjects, confirming that single-layer recurrent architectures with 64 units effectively capture 1-second viewport trajectory dynamics without overfitting.

### Final Benchmark Test Results (Selected GRU-64 Model)
Evaluated on **45,125 completely unseen test samples** across 8 test subjects (from `results/metrics/best_model_test_metrics.txt`):

| Metric | Result |
|---|---|
| **Spherical Mean Error (MAE)** | **9.5525°** |
| **Spherical Median Error (50th %)** | **6.8841°** |
| **Spherical 90th Percentile Error** | **20.0725°** |
| **Spherical RMSE** | **13.3058°** |
| **Yaw Mean Absolute Error (MAE)** | **8.2482°** |
| **Yaw Root Mean Squared Error (RMSE)** | **12.6100°** |
| **Pitch Mean Absolute Error (MAE)** | **3.6847°** |
| **Pitch Root Mean Squared Error (RMSE)** | **5.6157°** |
| Reference Baseline (LSTM-64) Test Spherical MAE | 9.5329° (~9.53°) |
| Reference Baseline (LSTM-64) Test Yaw MAE | 8.2519° |
| Reference Baseline (LSTM-64) Test Pitch MAE | 3.6493° |
| Percentage Difference vs. 9.5329° Reference | -0.21% |
| Percentage Difference vs. 9.53° Reference | -0.24% |

---

## 10. Folder Structure
```text
ml-final-project/
│
├── data/
│   ├── raw/
│   │   ├── Head_rotation/          # 48 raw human subject head-tracking JSON files (2.json to 49.json)
│   │   └── SRC_info/               # 20 raw video technical metadata JSON files (1.json to 20.json)
│   └── processed/
│       ├── combined_viewport_data.csv       # Consolidated raw dataset (284,247 rows)
│       ├── viewport_trajectories.csv        # Segmented trajectory dataset with subject IDs
│       ├── trajectory_summary.csv           # Summary metrics for all 960 trajectories
│       ├── unique_videos.csv                # Video distribution and sample counts
│       ├── video_metadata.csv               # Extracted FFprobe video specifications
│       ├── viewport_movement_analysis.csv   # Motion dynamics per trajectory
│       ├── lstm_subject_split.csv           # Subject-to-split mapping (33 train, 7 val, 8 test)
│       ├── lstm_train.npz                   # Supervised training tensors (186,176 sequences)
│       ├── lstm_validation.npz              # Supervised validation tensors (39,506 sequences)
│       └── lstm_test.npz                    # Supervised test tensors (45,125 sequences)
│
├── src/
│   ├── preprocessing/
│   │   ├── download_src_info.py             # Automates fetching video metadata JSONs
│   │   ├── extract_video_metadata.py        # Compiles technical video specifications to CSV
│   │   ├── combine_viewport_data.py         # Parses and consolidates head tracking JSONs
│   │   ├── verify_csv.py                    # Fast dataset schema and integrity check
│   │   └── analyze_unique_videos.py         # Video sample distribution analysis
│   ├── feature_engineering/
│   │   ├── __init__.py                      # Package exports
│   │   └── features.py                      # 7-feature kinematics & trigonometric encoding
│   ├── dataset_generation/
│   │   ├── __init__.py                      # Package exports
│   │   └── sequence_generator.py            # Sliding window sequence creation (10-step, 5-horizon)
│   ├── models/
│   │   ├── __init__.py                      # Package exports
│   │   └── architectures.py                 # PyTorch recurrent architectures (LSTM & GRU)
│   └── evaluation/
│       ├── __init__.py                      # Package exports
│       └── compare_models.py                # Multi-model training, early stopping & evaluation
│
├── saved_models/
│   ├── viewport_lstm_model.pth              # Baseline LSTM-64 state dict (PyTorch native)
│   └── best_viewport_model.pth              # Winning GRU-64 state dict (PyTorch native)
│
├── results/
│   ├── metrics/
│   │   ├── best_model_test_metrics.txt      # Final evaluation report for selected GRU-64 model
│   │   ├── model_comparison_report.txt      # Comprehensive comparison report across all 4 models
│   │   ├── model_comparison.csv             # Validation metrics comparison table
│   │   ├── lstm_training_report.txt         # Baseline LSTM training report
│   │   ├── lstm_training_history.csv        # Baseline LSTM epoch-by-epoch loss history
│   │   ├── lstm_test_metrics_by_video.csv   # Baseline test error breakdown per video
│   │   ├── viewport_data_analysis.txt       # Exploratory analysis on sampling and jitter
│   │   ├── viewport_movement_analysis.txt   # Motion dynamics statistics (velocity, acceleration)
│   │   └── lstm_dataset_summary.txt         # Supervised sequence split documentation
│   ├── predictions/
│   │   ├── best_model_test_predictions.csv  # Frame-by-frame actual vs predicted viewports (GRU-64)
│   │   └── lstm_test_predictions.csv        # Frame-by-frame actual vs predicted viewports (LSTM-64)
│   └── plots/
│       ├── model_comparison_training_curves.png  # Training vs validation loss across all 4 models
│       └── lstm_training_curve.png               # Baseline LSTM loss convergence curve
│
├── docs/
│   ├── CODE_EXPLANATION_AND_LIBRARIES.txt   # Detailed codebase architecture & 30 viva Q&As
│   └── code_backup_before_commenting/       # Preserved scripts prior to inline documentation
│
├── requirements.txt                         # Third-party dependencies
├── .gitignore                               # Git ignore configuration
└── README.md                                # Comprehensive repository documentation
```

---

## 11. Libraries and Dependencies
The project uses standard Python with verified third-party scientific computing and deep learning packages:

- **Python:** 3.10+
- **PyTorch (`torch`):** Deep learning framework powering recurrent layers (`nn.LSTM`, `nn.GRU`), automatic differentiation, Adam optimization, mini-batch `DataLoader`, and weight persistence.
- **NumPy (`numpy`):** N-dimensional array processing, trigonometric calculations, spherical dot products, inverse tangents (`np.arctan2`), and binary `.npz` I/O.
- **Pandas (`pandas`):** Tabular data ingestion, missing value verification, aggregation, and CSV exports.
- **Matplotlib (`matplotlib.pyplot`):** Scientific visualization and loss trajectory plotting.

> [!NOTE]
> **Model Checkpoint Format (.pth):**
> The model checkpoints in `saved_models/` (`best_viewport_model.pth` and `viewport_lstm_model.pth`) are PyTorch state dictionaries serialized natively via `torch.save()`. Binary inspection confirms they are standard PyTorch state dictionaries containing TorchScript zip headers and `data.pkl`. They are loaded and evaluated using `torch.load()`. Neither TensorFlow nor Keras is required or used in this codebase.

Standard library modules (`pathlib`, `json`, `urllib.request`, `time`, `os`, `random`) are utilized without requiring external pip installations.

---

## 12. Installation Instructions
Clone the repository and set up a Python virtual environment:

```bash
# Clone the repository
git clone https://github.com/KOTHABHASKARABALAJI/VR-VEIWPORTPREDICTION.git
cd VR-VEIWPORTPREDICTION

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 13. Execution Instructions (Run Order)
The project workflow consists of data ingestion, exploratory validation, precomputed artifact loading, and multi-model evaluation. Execute scripts in the following order:

### Step 1: Preprocess Raw Head Rotation Logs
Ingests all 48 participant JSON files in `data/raw/Head_rotation/` and generates the consolidated master dataset in `data/processed/combined_viewport_data.csv`:
```bash
python src/preprocessing/combine_viewport_data.py
```

### Step 2: Validate Data Schema & Video Sample Distribution
Performs sanity checks on table dimensions, columns, and sample counts across all 20 videos, exporting `data/processed/unique_videos.csv`:
```bash
python src/preprocessing/verify_csv.py
python src/preprocessing/analyze_unique_videos.py
```

### Step 3 (Optional): Re-download and Extract Source Video Metadata
Fetches the 20 source video technical FFprobe metadata files from GitHub and extracts container/codec specifications into `data/processed/video_metadata.csv`:
```bash
python src/preprocessing/download_src_info.py
python src/preprocessing/extract_video_metadata.py
```

### Step 4: Intermediate Trajectory & Sequence Artifacts
The repository includes precomputed intermediate data files in `data/processed/`:
- `data/processed/viewport_trajectories.csv`: Trajectories segmented by subject and video.
- `data/processed/trajectory_summary.csv`: Duration and sample count summary for all 960 trajectories.
- `data/processed/viewport_movement_analysis.csv`: Motion dynamics and velocity/acceleration percentiles.
- `data/processed/lstm_subject_split.csv`: Subject assignment across train (33), validation (7), and test (8) splits.
- `data/processed/lstm_train.npz`, `lstm_validation.npz`, `lstm_test.npz`: Supervised sequences ($X$: shape `(N, 10, 7)`) and targets ($y$: shape `(N, 3)`).

The underlying logic for feature computation (unwrapping, velocity, acceleration) and sequence generation (10 history frames, 5 horizon frames) is organized in modular libraries:
- `src/feature_engineering/features.py`
- `src/dataset_generation/sequence_generator.py`

### Step 5: Run Multi-Model Training, Evaluation & Comparison
Loads the preprocessed sequence tensors from `data/processed/`, trains all 4 recurrent architectures (`ModelA_LSTM64`, `ModelB_LSTM128`, `ModelC_GRU64`, `ModelD_LSTMDropout` from `src/models/architectures.py`) with early stopping, compares validation metrics, saves the winning model state dictionary to `saved_models/`, and generates comparison tables, test predictions, metrics reports, and training curves:
```bash
python src/evaluation/compare_models.py
```

---

## 14. Completed Work
- [x] Full ingestion and validation pipeline for the AVTrack360 benchmark dataset (48 subjects, 20 videos, 284,247 samples).
- [x] Phase-unwrapped angular velocity and acceleration feature engineering (`src/feature_engineering/features.py`).
- [x] Continuous trigonometric $[\sin(\text{yaw}), \cos(\text{yaw})]$ encoding eliminating boundary discontinuities.
- [x] Subject-disjoint train/validation/test sequence generation (`src/dataset_generation/sequence_generator.py`) yielding 270,807 valid sliding windows.
- [x] Modular implementation of 4 deep recurrent architectures (LSTM-64, LSTM-128, GRU-64, LSTM-Dropout) in PyTorch (`src/models/architectures.py`).
- [x] Early stopping with automatic best-weight checkpoint restoration.
- [x] Spherical great-circle orthodromic distance evaluation metric.
- [x] Model comparison experiment demonstrating consistent convergence across all variants and benchmarking winning GRU-64 model on unseen test subjects.
- [x] Comprehensive documentation, project viva guide, and modular repository organization.

---

## 15. Planned Future Work
The predictive viewport model developed here forms the foundation for subsequent adaptive streaming stages:

1. **360ProbDASH-Based Adaptive Tile Quality Allocation:**
   - Map continuous predicted $(\text{yaw}, \text{pitch})$ coordinates to discrete equirectangular $(M \times N)$ tile grid indices.
   - Formulate probabilistic viewing likelihoods for spatial tiles based on Gaussian spherical uncertainty distributions.
   - Implement optimization algorithms (greedy knapsack / Lagrangian relaxation) to maximize viewport Quality of Experience (QoE) under fluctuating client bandwidth constraints.

2. **Two-Stage Fallback Tile Retrieval for Sudden Viewport Changes:**
   - Detect rapid head saccades exceeding prediction velocity thresholds.
   - Stage 1: Immediately fetch newly visible peripheral tiles at lower base quality to prevent visible rebuffering stalls.
   - Stage 2: Progressively upgrade visible tiles to higher quality layers as network buffer headroom stabilizes.

> **Important Notice:** The components listed in this section represent planned extensions for subsequent stages of this research and are **not yet implemented** in this repository. They are presented here to delineate the future research trajectory without claiming existing implementation.

---

## 16. License
No license is currently specified. All rights reserved. Please consult the repository owner for licensing and usage permissions.
