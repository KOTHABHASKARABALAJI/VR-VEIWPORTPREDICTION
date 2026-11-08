"""
Deep recurrent neural network models for 360° viewport prediction:
- ModelA_LSTM64: LSTM-64 baseline (1-layer LSTM with 64 units)
- ModelB_LSTM128: LSTM-128 (1-layer LSTM with 128 units)
- ModelC_GRU64: GRU-64 (1-layer GRU with 64 units - Selected Best Model)
- ModelD_LSTMDropout: LSTM-Dropout (1-layer LSTM with 64 units and p=0.2 dropout)
"""

from .architectures import (
    ModelA_LSTM64,
    ModelB_LSTM128,
    ModelC_GRU64,
    ModelD_LSTMDropout,
    ModelE_BiLSTM64,
    ConfigurableGRU,
    ConfigurableLSTM,
    ViewportGRU,
    ViewportLSTM,
)

__all__ = [
    "ModelA_LSTM64",
    "ModelB_LSTM128",
    "ModelC_GRU64",
    "ModelD_LSTMDropout",
    "ModelE_BiLSTM64",
    "ConfigurableGRU",
    "ConfigurableLSTM",
    "ViewportGRU",
    "ViewportLSTM",
    "FinalViewportPredictor",
]


def __getattr__(name: str):
    if name == "FinalViewportPredictor":
        from .final_model_inference import FinalViewportPredictor
        return FinalViewportPredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

