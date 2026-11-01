"""
Deepfake Detection Model Architectures for VoiceGuard AI.
Provides:
1. DeepfakeCNNBiLSTMClassifier: Temporal CNN-BiLSTM hybrid for audio deepfake detection:
   - 4-Block 2D CNN with exact channel progression: 1 -> 32 -> 64 -> 128 -> 256
   - Frequency adaptive average pooling to 8 bins while preserving full temporal axis
   - Bidirectional LSTM (2 layers, hidden size 128)
   - Binary classification head producing unconstrained raw logits for BCEWithLogitsLoss
2. DeepfakeCNNClassifier: 2D Convolutional neural network baseline (Model A Baseline)
3. Strict Checkpoint Loader: Validates checkpoint bundle integrity, architecture configuration,
   and parameters with informative error messages.
"""

import os
from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn


class DeepfakeCNNBiLSTMClassifier(nn.Module):
    """
    Four-Block 2D CNN + Bidirectional LSTM Hybrid Classifier for Voice Deepfake Detection.
    Preserves exact channel progression: 1 -> 32 -> 64 -> 128 -> 256.
    """
    def __init__(self, num_mels: int = 64, lstm_hidden: int = 128, dropout_rate: float = 0.35):
        super().__init__()
        self.num_mels = num_mels
        self.lstm_hidden = lstm_hidden
        self.dropout_rate = dropout_rate
        self.decision_threshold = 0.5  # Calibrated on validation data

        # Block 1: Conv2d(1, 32) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Block 2: Conv2d(32, 64) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Block 3: Conv2d(64, 128) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Block 4: Conv2d(128, 256) -> BatchNorm2d -> ReLU -> AdaptiveAvgPool2d((8, None))
        # Standardizes frequency axis to 8 bins while preserving temporal sequence
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, None))
        )
        # Sequence modeling: BiLSTM over temporal frames
        self.lstm = nn.LSTM(
            input_size=256 * 8,  # 2048
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if dropout_rate > 0 else 0.0
        )
        # Binary classification head: outputs raw unconstrained logits
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch, 1, 64, time_frames)
        Returns:
            logits: Unconstrained raw logits of shape (batch, 1)
        """
        # Strict input assertions
        assert isinstance(x, torch.Tensor), f"Expected torch.Tensor input, got {type(x)}"
        assert x.ndim == 4, f"Expected 4D input tensor (batch, 1, {self.num_mels}, time_frames), got shape {list(x.shape)}"
        assert x.shape[1] == 1, f"Expected channel dimension = 1, got {x.shape[1]}"
        assert x.shape[2] == self.num_mels, f"Expected frequency dimension = {self.num_mels}, got {x.shape[2]}"
        assert torch.all(torch.isfinite(x)), "Input tensor contains NaN or Inf values"

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        B, C, F_dim, T_dim = x.shape
        assert F_dim == 8, f"Expected pooled frequency dimension 8, got {F_dim}"
        assert T_dim >= 1, f"Temporal dimension collapsed to {T_dim}"

        # Permute to (batch, time_frames, channels * freq_bins) = (B, T_dim, 2048)
        x = x.permute(0, 3, 1, 2).reshape(B, T_dim, C * F_dim)

        lstm_out, _ = self.lstm(x)
        pooled = torch.mean(lstm_out, dim=1)  # Mean pooling over temporal dimension
        logits = self.fc(pooled)

        assert logits.shape == (B, 1), f"Expected output shape ({B}, 1), got {list(logits.shape)}"
        assert torch.all(torch.isfinite(logits)), "Output logits contain NaN or Inf values"
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute sigmoid probabilities from raw logits without gradient tracking."""
        with torch.no_grad():
            logits = self.forward(x)
            prob = torch.sigmoid(logits)
        return prob

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_config(self) -> Dict[str, Any]:
        return {
            "architecture": "DeepfakeCNNBiLSTMClassifier",
            "num_mels": self.num_mels,
            "lstm_hidden": self.lstm_hidden,
            "dropout_rate": self.dropout_rate,
            "decision_threshold": self.decision_threshold,
        }

    def save_checkpoint_bundle(
        self,
        path: str,
        feature_config: Optional[Dict[str, Any]] = None,
        class_mapping: Optional[Dict[str, int]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """Save complete v2.0 checkpoint bundle with architecture, preprocessing, and training metadata."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        bundle = {
            "version": "2.0",
            "architecture": "DeepfakeCNNBiLSTMClassifier",
            "model_config": self.get_config(),
            "feature_config": feature_config or {
                "sr": 16000, "n_mels": 64, "n_fft": 512, "hop_length": 256,
                "fmin": 20.0, "fmax": 8000.0, "power": 2.0, "target_duration": 3.0
            },
            "class_mapping": class_mapping or {"REAL": 0, "FAKE": 1},
            "decision_threshold": float(self.decision_threshold),
            "metrics": metrics or {},
            "state_dict": self.state_dict(),
        }
        torch.save(bundle, path)

    def load_weights(self, path: str, map_location: str = "cpu"):
        """Load weights from a file strictly. Fails with clear error if missing or invalid."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Weights file not found: {path}")

        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
            if "decision_threshold" in checkpoint:
                self.decision_threshold = float(checkpoint["decision_threshold"])
        elif isinstance(checkpoint, dict):
            state = checkpoint
        else:
            raise ValueError(f"Unrecognized checkpoint format in {path}: {type(checkpoint)}")

        self.load_state_dict(state, strict=True)
        self.eval()


class DeepfakeCNNClassifier(nn.Module):
    """
    Standard 4-block 2D CNN feature extractor and classification head (Model A Baseline).
    """
    def __init__(self, dropout_rate: float = 0.3):
        super().__init__()
        self.decision_threshold = 0.5

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor), f"Expected torch.Tensor, got {type(x)}"
        assert x.ndim == 4, f"Expected 4D input tensor, got shape {list(x.shape)}"
        assert torch.all(torch.isfinite(x)), "Input tensor contains NaN or Inf"
        feats = self.features(x)
        logit = self.classifier(feats)
        assert torch.all(torch.isfinite(logit)), "Output logit contains NaN or Inf"
        return logit

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logit = self.forward(x)
            prob = torch.sigmoid(logit)
        return prob

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def load_weights(self, path: str, map_location: str = "cpu"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Weights file not found: {path}")
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
            if "decision_threshold" in checkpoint:
                self.decision_threshold = float(checkpoint["decision_threshold"])
        elif isinstance(checkpoint, dict):
            state = checkpoint
        else:
            raise ValueError(f"Unrecognized checkpoint format: {type(checkpoint)}")
        self.load_state_dict(state, strict=True)
        self.eval()


# Alias for backward compatibility
DeepfakeAudioClassifier = DeepfakeCNNBiLSTMClassifier


def load_classifier(weights_path: str, map_location: str = "cpu", strict: bool = True) -> nn.Module:
    """
    Strict classifier loader for VoiceGuard AI.
    Inspects the checkpoint state_dict and dynamically instantiates either
    DeepfakeCNNBiLSTMClassifier or DeepfakeCNNClassifier.
    Fails explicitly with FileNotFoundError or RuntimeError if missing or incompatible.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Required checkpoint file does not exist: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=map_location, weights_only=False)
    metadata = {}

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        metadata = {k: v for k, v in checkpoint.items() if k != "state_dict"}
    elif isinstance(checkpoint, dict):
        state = checkpoint
    else:
        raise ValueError(f"Unrecognized checkpoint data format in {weights_path}: {type(checkpoint)}")

    # Detect architecture based on key signatures
    if "lstm.weight_ih_l0" in state or "conv1.0.weight" in state:
        model = DeepfakeCNNBiLSTMClassifier()
        model.load_state_dict(state, strict=strict)
    elif "features.0.weight" in state or "classifier.1.weight" in state:
        model = DeepfakeCNNClassifier()
        model.load_state_dict(state, strict=strict)
    else:
        raise RuntimeError(f"Checkpoint keys in {weights_path} do not match either DeepfakeCNNBiLSTMClassifier or DeepfakeCNNClassifier.")

    if "decision_threshold" in metadata:
        model.decision_threshold = float(metadata["decision_threshold"])
    elif hasattr(model, "decision_threshold"):
        pass
    else:
        model.decision_threshold = 0.5

    model.metadata = metadata
    model.eval()
    return model
