"""
Module: final_model_inference.py
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming

Purpose:
    Provides a clean, validated, production-ready inference API for the final viewport prediction model:
    - Model: saved_models/final_viewport_model.pth (ConfigurableGRU, 64 hidden units)
    - Configuration: saved_models/final_model_config.json
    - Historical Input: L = 10 frames (~1.0 second context at ~10 Hz)
    - Prediction Lead: k = 5 frames (~0.5 second ahead)
    - Features: Strictly causal 7-feature kinematic representation
    - Causal parity: Guarantees feature and derivative parity between training and inference

Usage:
    from src.models.final_model_inference import FinalViewportPredictor

    # Batch prediction
    predictor = FinalViewportPredictor()
    pred_yaw, pred_pitch = predictor.predict(yaw_history_10_frames, pitch_history_10_frames)

    # Online streaming prediction
    for frame in video_stream:
        prediction = predictor.push_and_predict_streaming(frame.yaw, frame.pitch, frame.timestamp)
        if prediction is not None:
            pred_yaw, pred_pitch = prediction
"""

import sys
import json
from pathlib import Path
from collections import deque
from typing import Tuple, List, Optional
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.architectures import ConfigurableGRU
from src.feature_engineering.features import extract_7_features, standardize_features


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

        # 1. Load and Validate Configuration
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self._validate_model_config(self.config)

        self.history_len = self.config["sequence_parameters"]["history_frames_L"]
        self.horizon = self.config["sequence_parameters"]["prediction_horizon_k"]
        self.lead_time_s = self.config["sequence_parameters"]["prediction_lead_time_s"]

        self.norm_mean = np.array(self.config["normalization"]["mean"], dtype=np.float32)
        self.norm_std = np.array(self.config["normalization"]["std"], dtype=np.float32)
        self.pitch_mean = float(self.config["normalization"]["pitch_mean"])
        self.pitch_std = float(self.config["normalization"]["pitch_std"])

        # 2. Instantiate Architecture and Load Weights
        arch = self.config["architecture"]
        self.model = ConfigurableGRU(
            input_dim=arch["input_dim"],
            hidden_dim=arch["hidden_dim"],
            dropout=arch.get("dropout", 0.0)
        )

        state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # 3. Streaming FIFO buffers and state for online inference
        self.buffer_yaw: deque[float] = deque(maxlen=self.history_len + 5)
        self.buffer_pitch: deque[float] = deque(maxlen=self.history_len + 5)
        self.buffer_ts: deque[float] = deque(maxlen=self.history_len + 5)
        self.streaming_mode: Optional[str] = None  # 'timestamped' or 'fixed_interval'
        self.feature_history: deque[np.ndarray] = deque(maxlen=self.history_len)
        self.prev_yaw: Optional[float] = None
        self.prev_pitch: Optional[float] = None
        self.prev_ts: Optional[float] = None
        self.prev_yaw_vel: Optional[float] = None
        self.prev_pitch_vel: Optional[float] = None
        self.streaming_step: int = 0

    @staticmethod
    def _validate_model_config(cfg: dict) -> None:
        """Validates configuration parameters, architecture shapes, and normalization vectors."""
        arch = cfg.get("architecture", {})
        if arch.get("input_dim") != 7:
            raise ValueError(f"Expected architecture input_dim=7, got {arch.get('input_dim')}")
        if arch.get("output_dim") != 3:
            raise ValueError(f"Expected architecture output_dim=3, got {arch.get('output_dim')}")

        seq_params = cfg.get("sequence_parameters", {})
        if seq_params.get("history_frames_L") != 10:
            raise ValueError(f"Expected history_frames_L=10, got {seq_params.get('history_frames_L')}")
        if seq_params.get("prediction_horizon_k") != 5:
            raise ValueError(f"Expected prediction_horizon_k=5, got {seq_params.get('prediction_horizon_k')}")

        norm = cfg.get("normalization", {})
        mean = np.array(norm.get("mean", []))
        std = np.array(norm.get("std", []))

        if mean.shape != (7,) or not np.all(np.isfinite(mean)):
            raise ValueError(f"Normalization mean must have 7 finite values, got {mean}")
        if std.shape != (7,) or not np.all(np.isfinite(std)) or np.any(std <= 0.0):
            raise ValueError(f"Normalization std must have 7 positive finite values, got {std}")

        p_mean = norm.get("pitch_mean")
        p_std = norm.get("pitch_std")
        if p_mean is None or not np.isfinite(p_mean):
            raise ValueError(f"Invalid pitch_mean: {p_mean}")
        if p_std is None or not np.isfinite(p_std) or p_std <= 0.0:
            raise ValueError(f"Invalid pitch_std: {p_std}")

    def predict(
        self,
        yaw_history_deg: np.ndarray | List[float],
        pitch_history_deg: np.ndarray | List[float],
        timestamps: np.ndarray | List[float] | None = None
    ) -> Tuple[float, float]:
        """
        Predicts future viewport center (yaw, pitch) 5 frames ahead given at least 10 historical frames.

        Parameters:
            yaw_history_deg: 1D array-like of past yaw angles in degrees [-180, +180].
            pitch_history_deg: 1D array-like of past pitch angles in degrees [-90, +90].
            timestamps: Optional 1D array-like of corresponding timestamps in seconds.
                Must be strictly monotonically increasing.

        Returns:
            Tuple[float, float]: (predicted_yaw_deg, predicted_pitch_deg).
        """
        yaw_arr = np.asarray(yaw_history_deg, dtype=np.float64)
        pitch_arr = np.asarray(pitch_history_deg, dtype=np.float64)

        if yaw_arr.ndim != 1 or pitch_arr.ndim != 1:
            raise ValueError(f"Input arrays must be 1D, got yaw.ndim={yaw_arr.ndim}, pitch.ndim={pitch_arr.ndim}")
        if len(yaw_arr) != len(pitch_arr):
            raise ValueError(f"Input arrays must have equal length, got yaw={len(yaw_arr)}, pitch={len(pitch_arr)}")
        if len(yaw_arr) < self.history_len:
            raise ValueError(
                f"Input sequence length ({len(yaw_arr)}) must be >= required history length ({self.history_len})."
            )

        # 1. Strictly causal 7-feature extraction (with full input validation inside extract_7_features)
        ts_arr = np.asarray(timestamps, dtype=np.float64) if timestamps is not None else None
        features = extract_7_features(yaw_arr, pitch_arr, timestamps=ts_arr)

        # 2. Extract final historical window of length L (10 frames)
        input_window = features[-self.history_len:]

        # 3. Z-score normalization using training split statistics
        input_norm = standardize_features(input_window, self.norm_mean, self.norm_std)

        # 4. Neural inference
        tensor_in = torch.tensor(
            input_norm.reshape(1, self.history_len, 7),
            dtype=torch.float32,
            device=self.device
        )
        with torch.inference_mode():
            out = self.model(tensor_in).cpu().numpy()[0]

        # 5. Output shape and finite validation
        if out.shape != (3,) or not np.all(np.isfinite(out)):
            raise ValueError(f"Model generated invalid output tensor: {out}")

        # 6. Trigonometric and spatial denormalization
        pred_sin, pred_cos, pred_norm_pitch = float(out[0]), float(out[1]), float(out[2])
        pred_yaw = float(np.degrees(np.arctan2(pred_sin, pred_cos)))
        pred_pitch = float(pred_norm_pitch * self.pitch_std + self.pitch_mean)
        pred_pitch = float(np.clip(pred_pitch, -90.0, 90.0))

        return pred_yaw, pred_pitch

    def push_and_predict_streaming(
        self,
        yaw_deg: float,
        pitch_deg: float,
        timestamp: float | None = None
    ) -> Optional[Tuple[float, float]]:
        """
        Online streaming interface: accepts 1 incoming viewport frame per call.
        Returns predicted viewport (yaw, pitch) once buffer reaches required history length.

        Streaming Policy:
            Enforces strict consistency: a stream is either timestamped for ALL frames,
            or fixed-interval for ALL frames. Mixing modes raises ValueError.

        Parameters:
            yaw_deg: Current frame yaw in degrees [-180, +180].
            pitch_deg: Current frame pitch in degrees [-90, +90].
            timestamp: Optional current timestamp in seconds.

        Returns:
            Optional[Tuple[float, float]]: Predicted (yaw, pitch) or None if buffer not yet full.
        """
        if isinstance(yaw_deg, bool) or not np.isfinite(float(yaw_deg)):
            raise ValueError(f"Invalid yaw_deg: {yaw_deg}")
        if isinstance(pitch_deg, bool) or not np.isfinite(float(pitch_deg)):
            raise ValueError(f"Invalid pitch_deg: {pitch_deg}")

        # Enforce streaming mode consistency
        if self.streaming_mode is None:
            self.streaming_mode = "timestamped" if timestamp is not None else "fixed_interval"
        else:
            if self.streaming_mode == "timestamped" and timestamp is None:
                raise ValueError("Streaming predictor initialized in timestamped mode, but received frame without timestamp.")
            elif self.streaming_mode == "fixed_interval" and timestamp is not None:
                raise ValueError("Streaming predictor initialized in fixed-interval mode, but received frame with timestamp.")

        yaw_val = float(yaw_deg)
        pitch_val = float(pitch_deg)
        self.buffer_yaw.append(yaw_val)
        self.buffer_pitch.append(pitch_val)
        ts_val = None
        if timestamp is not None:
            ts_val = float(timestamp)
            if not np.isfinite(ts_val):
                raise ValueError(f"Non-finite timestamp in streaming: {timestamp}")
            if len(self.buffer_ts) > 0 and ts_val <= self.buffer_ts[-1]:
                raise ValueError(
                    f"Non-monotonic timestamp in streaming: current={ts_val} <= previous={self.buffer_ts[-1]}"
                )
            self.buffer_ts.append(ts_val)

        # Causal kinematic feature calculation bit-for-bit identical to extract_7_features
        sin_y = float(np.sin(np.radians(yaw_val)))
        cos_y = float(np.cos(np.radians(yaw_val)))

        if self.streaming_step == 0:
            y_vel, p_vel = 0.0, 0.0
            y_acc, p_acc = 0.0, 0.0
        elif self.streaming_step == 1:
            dt = (ts_val - self.prev_ts) if self.streaming_mode == "timestamped" else 0.1
            y_diff = ((yaw_val - self.prev_yaw + 180.0) % 360.0) - 180.0
            y_vel = float(y_diff / dt)
            p_vel = float((pitch_val - self.prev_pitch) / dt)
            y_acc, p_acc = 0.0, 0.0
        else:
            dt = (ts_val - self.prev_ts) if self.streaming_mode == "timestamped" else 0.1
            y_diff = ((yaw_val - self.prev_yaw + 180.0) % 360.0) - 180.0
            y_vel = float(y_diff / dt)
            p_vel = float((pitch_val - self.prev_pitch) / dt)
            y_acc = float((y_vel - self.prev_yaw_vel) / dt)
            p_acc = float((p_vel - self.prev_pitch_vel) / dt)

        feat_vec = np.array([sin_y, cos_y, pitch_val, y_vel, p_vel, y_acc, p_acc], dtype=np.float32)
        self.feature_history.append(feat_vec)

        self.prev_yaw = yaw_val
        self.prev_pitch = pitch_val
        self.prev_ts = ts_val
        self.prev_yaw_vel = y_vel
        self.prev_pitch_vel = p_vel
        self.streaming_step += 1

        if len(self.feature_history) < self.history_len:
            return None

        feats_10 = np.array(self.feature_history, dtype=np.float32)
        norm_in = standardize_features(feats_10, self.norm_mean, self.norm_std)
        tensor_in = torch.tensor(
            norm_in.reshape(1, self.history_len, 7),
            dtype=torch.float32,
            device=self.device
        )
        with torch.inference_mode():
            pred = self.model(tensor_in).cpu().numpy()[0]

        pred_yaw = float(np.degrees(np.arctan2(pred[0], pred[1])))
        pred_pitch = float(pred[2] * self.pitch_std + self.pitch_mean)
        pred_pitch = float(np.clip(pred_pitch, -90.0, 90.0))
        return pred_yaw, pred_pitch

    def reset_streaming(self) -> None:
        """Clears streaming buffers and resets streaming mode."""
        self.buffer_yaw.clear()
        self.buffer_pitch.clear()
        self.buffer_ts.clear()
        self.streaming_mode = None
        self.feature_history.clear()
        self.prev_yaw = None
        self.prev_pitch = None
        self.prev_ts = None
        self.prev_yaw_vel = None
        self.prev_pitch_vel = None
        self.streaming_step = 0
