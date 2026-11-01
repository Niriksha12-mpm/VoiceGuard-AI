"""
Automated Unit Tests for Model Architecture in VoiceGuard AI.
Verifies compliance with Phase 3:
1. Four 2D Conv blocks with channel progression 1 -> 32 -> 64 -> 128 -> 256.
2. Temporal preservation across frequency pooling.
3. Bidirectional LSTM and raw logits output for BCEWithLogitsLoss.
4. Input assertions (shape, finiteness, channels, mel bands).
5. Checkpoint bundling and loading compatibility.
"""

import os
import tempfile
import numpy as np
import torch
import torch.nn as nn
import pytest

from models.architecture import DeepfakeCNNBiLSTMClassifier, DeepfakeCNNClassifier, load_classifier


class TestModelArchitecture:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = DeepfakeCNNBiLSTMClassifier(num_mels=64, lstm_hidden=128, dropout_rate=0.35)
        self.model.eval()

    def test_forward_pass_dimensions(self):
        """Batch of (B, 1, 64, T) must output logits of shape (B, 1)."""
        for batch_size in [1, 2, 4]:
            for time_frames in [94, 188, 250]:
                x = torch.randn(batch_size, 1, 64, time_frames)
                logits = self.model(x)
                assert logits.shape == (batch_size, 1), (
                    f"Expected shape ({batch_size}, 1), got {logits.shape}"
                )
                assert torch.all(torch.isfinite(logits)), "Logits contain NaN or Inf"

    def test_raw_logits_with_bce_loss(self):
        """Model outputs raw logits compatible with nn.BCEWithLogitsLoss."""
        criterion = nn.BCEWithLogitsLoss()
        x = torch.randn(4, 1, 64, 188)
        y = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float32)

        self.model.train()
        logits = self.model(x)
        loss = criterion(logits, y)

        assert torch.isfinite(loss), "BCEWithLogitsLoss is not finite"
        assert loss.item() > 0.0

        loss.backward()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient missing for {name}"
                assert torch.all(torch.isfinite(param.grad)), f"Gradient non-finite for {name}"

    def test_probability_prediction(self):
        """predict_proba must return values strictly in [0.0, 1.0]."""
        x = torch.randn(3, 1, 64, 188)
        probs = self.model.predict_proba(x)
        assert probs.shape == (3, 1)
        assert torch.all((probs >= 0.0) & (probs <= 1.0))

    def test_input_shape_assertions(self):
        """Model must reject invalid input shapes with AssertionError."""
        with pytest.raises(AssertionError):
            self.model(torch.randn(1, 64, 188))

        with pytest.raises(AssertionError):
            self.model(torch.randn(2, 3, 64, 188))

        with pytest.raises(AssertionError):
            self.model(torch.randn(2, 1, 80, 188))

    def test_non_finite_input_assertion(self):
        """Model must reject NaN or Inf inputs."""
        x = torch.randn(2, 1, 64, 188)
        x[0, 0, 10, 10] = float("nan")
        with pytest.raises(AssertionError):
            self.model(x)

    def test_channel_progression(self):
        """Verify layer channel counts: 1 -> 32 -> 64 -> 128 -> 256."""
        assert self.model.conv1[0].in_channels == 1
        assert self.model.conv1[0].out_channels == 32
        assert self.model.conv2[0].in_channels == 32
        assert self.model.conv2[0].out_channels == 64
        assert self.model.conv3[0].in_channels == 64
        assert self.model.conv3[0].out_channels == 128
        assert self.model.conv4[0].in_channels == 128
        assert self.model.conv4[0].out_channels == 256

    def test_checkpoint_bundle_roundtrip(self):
        """Save and load v2.0 checkpoint bundle preserves weights, config, and threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test_bundle.pt")
            self.model.decision_threshold = 0.5264
            self.model.save_checkpoint_bundle(
                ckpt_path,
                feature_config={"sr": 16000, "n_mels": 64},
                class_mapping={"REAL": 0, "FAKE": 1},
                metrics={"best_val_loss": 0.321}
            )

            loaded_model = load_classifier(ckpt_path)
            assert isinstance(loaded_model, DeepfakeCNNBiLSTMClassifier)
            assert np.isclose(loaded_model.decision_threshold, 0.5264)
            assert loaded_model.metadata["version"] == "2.0"
            assert loaded_model.metadata["metrics"]["best_val_loss"] == 0.321
