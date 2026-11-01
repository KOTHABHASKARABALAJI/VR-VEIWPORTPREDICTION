"""
Model: architectures.py
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming

Defines the 4 deep recurrent neural network architectures evaluated in the research:
- Model A: LSTM-64 baseline (Single-layer LSTM with 64 units)
- Model B: LSTM-128 (High-capacity LSTM with 128 units)
- Model C: GRU-64 (Gated Recurrent Unit with 64 units - Winning Model)
- Model D: LSTM with Dropout (LSTM-64 with p=0.2 dropout regularization)
"""

import torch
import torch.nn as nn


class ModelA_LSTM64(nn.Module):
    """
    Model A — Baseline Recurrent Architecture: Single-layer LSTM with 64 hidden units.

    Layers:
        1. nn.LSTM(7, 64, batch_first=True): Recurrent layer with 64 memory cells.
        2. nn.Linear(64, 32): Fully connected projection layer reducing 64 features to 32.
        3. nn.ReLU(): Non-linear activation function f(x) = max(0, x).
        4. nn.Linear(32, 3): Final regression output layer predicting [sin, cos, norm_pitch].

    Trainable Parameters: 20,835
    """
    def __init__(self):
        super().__init__()
        # Input size = 7 features, Hidden size = 64 units, batch_first=True makes input (batch, seq, feature)
        self.lstm = nn.LSTM(7, 64, batch_first=True)
        # Dense projection layer from LSTM hidden state (64) to latent representation (32)
        self.fc1 = nn.Linear(64, 32)
        # Rectified Linear Unit activation
        self.relu = nn.ReLU()
        # Output layer producing 3 continuous regression coordinates
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        # Pass input sequence through LSTM: out shape is (batch, 10, 64)
        out, _ = self.lstm(x)
        # Extract the hidden state of the final timestep (index -1) representing sequence summary
        out = self.fc1(out[:, -1, :])
        # Apply non-linear ReLU activation
        out = self.relu(out)
        # Produce final 3-element prediction vector
        return self.fc2(out)


class ModelB_LSTM128(nn.Module):
    """
    Model B — High-Capacity Architecture: Single-layer LSTM with 128 hidden units.

    Tests whether doubling recurrent capacity to 128 units captures subtler motion trends.

    Layers:
        1. nn.LSTM(7, 128, batch_first=True): 128 hidden memory units.
        2. nn.Linear(128, 32): Dense layer projecting 128 recurrent features to 32.
        3. nn.ReLU(): Non-linear activation function.
        4. nn.Linear(32, 3): Linear regression output layer.

    Trainable Parameters: 73,923
    """
    def __init__(self):
        super().__init__()
        # 128 hidden units for increased recurrent representational capacity
        self.lstm = nn.LSTM(7, 128, batch_first=True)
        self.fc1 = nn.Linear(128, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Take the final timestep's hidden state (shape: batch, 128)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        return self.fc2(out)


class ModelC_GRU64(nn.Module):
    """
    Model C — Gated Recurrent Unit Architecture: Single-layer GRU with 64 hidden units.

    GRU simplifies the LSTM architecture by merging cell and hidden states and using only 2 gates
    (Reset Gate and Update Gate) instead of 3 gates, reducing parameters by ~23% for faster inference.

    Layers:
        1. nn.GRU(7, 64, batch_first=True): 64 GRU memory units.
        2. nn.Linear(64, 32): Fully connected layer.
        3. nn.ReLU(): Non-linear activation function.
        4. nn.Linear(32, 3): Linear regression output layer.

    Trainable Parameters: 16,035 (~23% fewer than LSTM-64)
    """
    def __init__(self):
        super().__init__()
        # GRU recurrent layer with 64 units
        self.gru = nn.GRU(7, 64, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        out, _ = self.gru(x)
        # Take the final timestep's hidden state (shape: batch, 64)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        return self.fc2(out)


class ModelD_LSTMDropout(nn.Module):
    """
    Model D — Regularized Architecture: LSTM-64 with Dropout (p = 0.2).

    Randomly zeroes out 20% of the LSTM output activations during training to prevent
    co-adaptation of recurrent features and discourage overfitting.

    Layers:
        1. nn.LSTM(7, 64, batch_first=True): 64 LSTM memory units.
        2. nn.Dropout(0.2): Dropout regularization layer (rate = 0.2).
        3. nn.Linear(64, 32): Fully connected layer.
        4. nn.ReLU(): Non-linear activation function.
        5. nn.Linear(32, 3): Linear regression output layer.

    Trainable Parameters: 20,835
    """
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(7, 64, batch_first=True)
        # Dropout regularization layer with 20% deactivation probability
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Apply dropout to the final timestep's hidden state before linear projection
        out = self.dropout(out[:, -1, :])
        out = self.fc1(out)
        out = self.relu(out)
        return self.fc2(out)

