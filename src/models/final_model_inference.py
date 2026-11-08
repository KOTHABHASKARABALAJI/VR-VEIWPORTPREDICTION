"""
Module: final_model_inference.py
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming

Provides a clean, reproducible, production-ready inference API for the final viewport prediction model:
- Model: saved_models/final_viewport_model.pth (ConfigurableGRU, 64 hidden units)
- Configuration: saved_models/final_model_config.json
- Historical Input: L = 10 frames (~1.0 second context)
- Prediction Lead: k = 5 frames (~0.5 second ahead)
- Features: 7 Kinematic Features (strictly causal backward differences)

Usage:
    from src.models.final_model_inference import FinalViewportPredictor

    predictor = FinalViewportPredictor()
    pred_yaw, pred_pitch = predictor.predict(yaw_history_10_frames, pitch_history_10_frames)
"""

import sys
import json
from pathlib import Path
from collections import deque
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.architectures import ConfigurableGRU
from src.feature_engineering.features import extract_7_features


class FinalViewportPredictor:
    """
    Production-ready viewport predictor implementing strictly causal inference.

    Consumes 10 historical frames (~1.0s) of head orientation (yaw, pitch) and
    predicts the user's viewport center 5 frames (~0.5s) into the future.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        config_path: str | Path | None = None,
        device: str = "cpu"
    ):
        self.device = torch.device(device)

        if model_path is None:
            model_path = PROJECT_ROOT / "saved_models" / "final_viewport_model.pth"
        if config_path is None:
            config_path = PROJECT_ROOT / "saved_models" / "final_model_config.json"

        self.model_path = Path(model_path)
        self.config_path = Path(config_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {self.model_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Model config not found at: {self.config_path}")

        # Load Configuration
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.history_len = self.config["sequence_parameters"]["history_frames_L"]
        self.horizon = self.config["sequence_parameters"]["prediction_horizon_k"]
        self.lead_time_s = self.config["sequence_parameters"]["prediction_lead_time_s"]

        self.norm_mean = np.array(self.config["normalization"]["mean"], dtype=np.float32)
        self.norm_std = np.array(self.config["normalization"]["std"], dtype=np.float32)
        self.pitch_mean = float(self.config["normalization"]["pitch_mean"])
        self.pitch_std = float(self.config["normalization"]["pitch_std"])

        # Instantiate Architecture
        arch = self.config["architecture"]
        self.model = ConfigurableGRU(
            input_dim=arch["input_dim"],
            hidden_dim=arch["hidden_dim"],
            dropout=arch["dropout"]
        )

        # Load Weights
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

        # Streaming FIFO buffers for online inference
        self.buffer_yaw = deque(maxlen=self.history_len + 2)
        self.buffer_pitch = deque(maxlen=self.history_len + 2)
        self.buffer_ts = deque(maxlen=self.history_len + 2)

    def predict(
        self,
        yaw_history_deg: np.ndarray | list[float],
        pitch_history_deg: np.ndarray | list[float],
        timestamps: np.ndarray | list[float] | None = None
    ) -> tuple[float, float]:
        """
        Predicts future viewport (yaw, pitch) 0.5s ahead given at least 10 historical frames.

        Parameters:
            yaw_history_deg: 1D array-like of past yaw angles in degrees [-180, +180].
            pitch_history_deg: 1D array-like of past pitch angles in degrees [-90, +90].
            timestamps: Optional 1D array-like of corresponding timestamps in seconds.

        Returns:
            tuple[float, float]: (predicted_yaw_deg, predicted_pitch_deg).
        """
        yaw_arr = np.asarray(yaw_history_deg, dtype=np.float64)
        pitch_arr = np.asarray(pitch_history_deg, dtype=np.float64)

        if len(yaw_arr) < self.history_len:
            raise ValueError(
                f"Input sequence length ({len(yaw_arr)}) must be >= required history length ({self.history_len})."
            )

        ts_arr = np.asarray(timestamps, dtype=np.float64) if timestamps is not None else None

        # 1. Strictly causal 7-feature extraction
        features = extract_7_features(yaw_arr, pitch_arr, timestamps=ts_arr)

        # 2. Extract final historical window of length L (10 frames)
        input_window = features[-self.history_len:]

        # 3. Z-score normalization using training split statistics
        input_norm = (input_window - self.norm_mean) / self.norm_std

        # 4. Neural inference
        tensor_in = torch.tensor(input_norm.reshape(1, self.history_len, 7), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.model(tensor_in).cpu().numpy()[0]

        # 5. Trigonometric and spatial denormalization
        pred_sin, pred_cos, pred_norm_pitch = out[0], out[1], out[2]
        pred_yaw = float(np.degrees(np.arctan2(pred_sin, pred_cos)))
        pred_pitch = float(pred_norm_pitch * self.pitch_std + self.pitch_mean)
        pred_pitch = float(np.clip(pred_pitch, -90.0, 90.0))

        return pred_yaw, pred_pitch

    def push_and_predict_streaming(
        self,
        yaw_deg: float,
        pitch_deg: float,
        timestamp: float | None = None
    ) -> tuple[float, float] | None:
        """
        Online streaming interface: accepts 1 incoming viewport frame per call.
        Returns predicted viewport (yaw, pitch) once buffer reaches required history length.

        Parameters:
            yaw_deg: Current frame yaw in degrees [-180, +180].
            pitch_deg: Current frame pitch in degrees [-90, +90].
            timestamp: Optional current timestamp in seconds.

        Returns:
            tuple[float, float] | None: Predicted (yaw, pitch) or None if buffer not yet full.
        """
        self.buffer_yaw.append(yaw_deg)
        self.buffer_pitch.append(pitch_deg)
        if timestamp is not None:
            self.buffer_ts.append(timestamp)

        if len(self.buffer_yaw) < self.history_len:
            return None

        ts = list(self.buffer_ts) if len(self.buffer_ts) == len(self.buffer_yaw) else None
        return self.predict(list(self.buffer_yaw), list(self.buffer_pitch), timestamps=ts)


def main():
    print("Testing FinalViewportPredictor...")
    predictor = FinalViewportPredictor()
    print("Loaded model successfully. History L:", predictor.history_len, "Horizon k:", predictor.horizon)

    # Synthetic test input
    synthetic_yaw = [0.0, 1.0, 2.0, 4.0, 7.0, 10.0, 13.0, 16.0, 19.0, 22.0]
    synthetic_pitch = [-5.0, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0]

    pyaw, ppitch = predictor.predict(synthetic_yaw, synthetic_pitch)
    print(f"Input last frame: yaw={synthetic_yaw[-1]}°, pitch={synthetic_pitch[-1]}°")
    print(f"Predicted viewport (+0.5s): yaw={pyaw:.2f}°, pitch={ppitch:.2f}°")


if __name__ == "__main__":
    main()
