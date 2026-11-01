"""
Unit and Integration Tests for W2V2-AASIST Preprocessing and Classifier Wrapper.
"""

import os
import pytest
import numpy as np

from models.w2v2_aasist import pad_fixed, preprocess_waveform, W2V2AASISTClassifier


def test_pad_fixed_short_audio():
    """Short audio (<64600 samples) must be tile-repeated to exactly 64600 samples."""
    short_audio = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    padded = pad_fixed(short_audio, max_len=64600)
    assert padded.shape == (64600,)
    assert padded.dtype == np.float32
    # Verify repetition
    assert np.allclose(padded[:8], [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4])


def test_pad_fixed_long_audio():
    """Long audio (>64600 samples) must be truncated to the first 64600 samples."""
    long_audio = np.linspace(0.0, 10.0, 100000, dtype=np.float32)
    truncated = pad_fixed(long_audio, max_len=64600)
    assert truncated.shape == (64600,)
    assert np.allclose(truncated, long_audio[:64600])


def test_pad_fixed_exact_audio():
    """Audio of exact length 64600 must remain unchanged."""
    exact_audio = np.random.randn(64600).astype(np.float32)
    padded = pad_fixed(exact_audio, max_len=64600)
    assert padded.shape == (64600,)
    assert np.allclose(padded, exact_audio)


def test_pad_fixed_empty_audio():
    """Empty audio should return a zero array of length 64600."""
    empty = np.array([], dtype=np.float32)
    padded = pad_fixed(empty, max_len=64600)
    assert padded.shape == (64600,)
    assert np.all(padded == 0)


def test_preprocess_waveform_stereo():
    """Stereo audio must be converted to 1D mono before windowing."""
    stereo_audio = np.ones((2, 32000), dtype=np.float32)
    processed = preprocess_waveform(stereo_audio, sr=16000, target_len=64600)
    assert processed.ndim == 1
    assert processed.shape == (64600,)


def test_preprocess_waveform_resampling():
    """Non-16kHz audio must be resampled to 16kHz."""
    orig_sr = 8000
    # 2 seconds at 8kHz = 16000 samples
    audio_8k = np.sin(2 * np.pi * 440 * np.linspace(0, 2, 16000)).astype(np.float32)
    # After resampling to 16kHz, 2 seconds = 32000 samples -> tiled to 64600
    processed = preprocess_waveform(audio_8k, sr=orig_sr, target_sr=16000, target_len=64600)
    assert processed.shape == (64600,)
    assert processed.dtype == np.float32


def test_threshold_calibration():
    """Calibrate decision threshold using synthetic genuine and spoof scores."""
    np.random.seed(42)
    # Genuine scores higher (e.g. mean +2.0), Spoof scores lower (e.g. mean -2.0)
    genuine_scores = np.random.normal(loc=2.0, scale=1.0, size=100)
    spoof_scores = np.random.normal(loc=-2.0, scale=1.0, size=100)
    scores = np.concatenate([spoof_scores, genuine_scores])
    labels = np.concatenate([np.zeros(100), np.ones(100)])

    # Mock classifier object for threshold test
    class MockClassifier:
        def __init__(self):
            self.decision_threshold = 0.0
        calibrate_threshold = W2V2AASISTClassifier.calibrate_threshold

    mock = MockClassifier()
    calibrated = mock.calibrate_threshold(scores, labels, criterion="youden")
    # Optimal threshold should separate -2.0 and +2.0, near 0.0
    assert -1.0 <= calibrated <= 1.0
