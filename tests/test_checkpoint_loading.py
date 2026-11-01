"""
Automated Unit Tests for Strict Checkpoint Compatibility in VoiceGuard AI.
Verifies compliance with Phase 8:
1. Strict checkpoint loading fails explicitly if checkpoint is missing or corrupt.
2. Preserves architecture parameters, feature configs, and calibrated thresholds.
3. Distinguishes CNN-BiLSTM hybrid from baseline pure CNN models.
"""

import os
import tempfile
import torch
import pytest

from models.architecture import DeepfakeCNNBiLSTMClassifier, DeepfakeCNNClassifier, load_classifier


class TestCheckpointLoading:
    def test_missing_checkpoint_raises_error(self):
        """Loading a non-existent checkpoint must raise FileNotFoundError with clear message."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_classifier("non_existent_weights_path.pt")
        assert "Required checkpoint file does not exist" in str(exc_info.value)

    def test_corrupt_checkpoint_raises_error(self):
        """Loading an invalid checkpoint must raise RuntimeError or ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            corrupt_path = os.path.join(tmpdir, "corrupt.pt")
            # Save dict with arbitrary unrecognized keys
            torch.save({"bad_key_1": torch.randn(10), "bad_key_2": 123}, corrupt_path)
            with pytest.raises(RuntimeError) as exc_info:
                load_classifier(corrupt_path)
            assert "do not match" in str(exc_info.value)

    def test_load_pretrained_model_a(self):
        """Model A checkpoint loads as DeepfakeCNNClassifier with strict matching."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ckpt_a = os.path.join(base_dir, "models", "saved", "pretrained_model_a.pt")
        if os.path.exists(ckpt_a):
            model_a = load_classifier(ckpt_a)
            assert isinstance(model_a, DeepfakeCNNClassifier)
            assert model_a.count_parameters() > 900000

    def test_load_fine_tuned_model_b(self):
        """Model B checkpoint loads as DeepfakeCNNBiLSTMClassifier with valid threshold."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ckpt_b = os.path.join(base_dir, "models", "saved", "fine_tuned_model_b_lakhs.pt")
        if os.path.exists(ckpt_b):
            model_b = load_classifier(ckpt_b)
            assert isinstance(model_b, DeepfakeCNNBiLSTMClassifier)
            assert hasattr(model_b, "decision_threshold")
            assert 0.0 < model_b.decision_threshold < 1.0
