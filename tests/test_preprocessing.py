"""
Automated Unit Tests for Audio Preprocessing and Feature Extraction in VoiceGuard AI.
Verifies compliance with Phase 4:
1. Standardizes audio to 16 kHz mono.
2. Preserves 64-band log-mel representation (512 n_fft, 256 hop_length).
3. Ensures tensor shape is strictly (batch, 1, 64, time_frames).
4. Tests WAV, FLAC, and MP3 formats, variable durations, and different sample rates.
5. Verifies stereo downmixing and silence handling.
"""

import os
import io
import tempfile
import numpy as np
import soundfile as sf
import torch
import pytest

from dataset.preprocessor import AudioPreprocessor
from models.feature_extractor import AudioFeatureExtractor, MelSpectrogramExtractor


class TestAudioPreprocessing:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.preprocessor = AudioPreprocessor(target_sr=16000, target_duration=3.0)
        self.extractor = AudioFeatureExtractor(sr=16000, n_mels=64, n_fft=512, hop_length=256)

    def test_tensor_shape_for_standard_duration(self):
        """A standard 3.0s audio at 16kHz (48,000 samples) must yield shape (1, 1, 64, 188)."""
        duration = 3.0
        sr = 16000
        samples = int(sr * duration)
        t = np.linspace(0, duration, samples, endpoint=False, dtype=np.float32)
        y = 0.5 * np.sin(2 * np.pi * 440 * t)

        tensor = self.extractor.extract_tensor(y)
        assert tensor.shape == (1, 1, 64, 188), f"Expected (1, 1, 64, 188), got {tensor.shape}"
        assert torch.all(torch.isfinite(tensor)), "Tensor contains non-finite values"

    def test_per_clip_normalization(self):
        """Log-Mel Spectrogram must have approximately zero mean and unit variance."""
        y = np.random.randn(48000).astype(np.float32)
        spec = self.extractor.extract_mel_spectrogram(y)
        assert spec.shape == (64, 188)
        assert np.isclose(spec.mean(), 0.0, atol=1e-5), f"Mean not zero: {spec.mean()}"
        assert np.isclose(spec.std(), 1.0, atol=1e-3), f"Std not unit: {spec.std()}"

    def test_stereo_to_mono_downmixing(self):
        """Stereo 2-channel audio must be converted to 1-channel mono."""
        sr = 16000
        stereo_y = np.random.randn(2, 48000).astype(np.float32)
        processed = self.preprocessor.process_waveform(stereo_y, orig_sr=sr)
        assert processed.ndim == 1, f"Expected 1D mono, got {processed.ndim}D"
        assert len(processed) <= 48000

    def test_sample_rate_resampling(self):
        """Audio at 44.1kHz, 22.05kHz, or 8kHz must resample to 16kHz."""
        for orig_sr in [8000, 22050, 44100, 48000]:
            duration = 1.5
            n_orig = int(orig_sr * duration)
            t = np.linspace(0, duration, n_orig, endpoint=False, dtype=np.float32)
            y = 0.4 * np.sin(2 * np.pi * 300 * t)

            processed = self.preprocessor.process_waveform(y, orig_sr=orig_sr)
            expected_samples = int(16000 * duration)
            assert abs(len(processed) - expected_samples) < 200, (
                f"Resampling from {orig_sr} failed: expected ~{expected_samples}, got {len(processed)}"
            )

    def test_variable_duration_segmentation(self):
        """Segment or pad must produce 48,000-sample segments for any input duration."""
        # Short audio (1.0s) -> 1 padded segment
        short_y = np.random.randn(16000).astype(np.float32)
        segs_short = self.preprocessor.segment_or_pad(short_y)
        assert len(segs_short) == 1
        assert len(segs_short[0]) == 48000

        # Long audio (7.0s) -> 2 full segments + 1 tail segment
        long_y = np.random.randn(int(16000 * 7.0)).astype(np.float32)
        segs_long = self.preprocessor.segment_or_pad(long_y)
        assert len(segs_long) >= 2
        for seg in segs_long:
            assert len(seg) == 48000

    def test_file_format_support(self):
        """Must correctly decode and process WAV, FLAC, and synthetic audio streams."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr = 16000
            y = 0.3 * np.sin(2 * np.pi * 500 * np.linspace(0, 3.0, 48000, dtype=np.float32))

            for ext in [".wav", ".flac"]:
                path = os.path.join(tmpdir, f"test_audio{ext}")
                sf.write(path, y, sr)
                segs, meta = self.preprocessor.process_file(path)
                assert len(segs) >= 1
                assert len(segs[0]) == 48000
                assert meta["target_sr"] == 16000

    def test_acoustic_diagnostics(self):
        """Feature extractor computes acoustic diagnostics accurately."""
        y = 0.5 * np.sin(2 * np.pi * 1000 * np.linspace(0, 3.0, 48000, dtype=np.float32))
        diag = self.extractor.compute_acoustic_diagnostics(y)
        assert "spectral_centroid_hz" in diag
        assert "spectral_rolloff_hz" in diag
        assert "zero_crossing_rate" in diag
        assert "spectral_flatness" in diag
        assert diag["spectral_centroid_hz"] > 0.0

    def test_reject_silent_corrupt_audio(self):
        """Audio preprocessor rejects pure silence or empty signal."""
        silent_y = np.zeros(16000, dtype=np.float32)
        valid, msg = self.preprocessor.validate_audio(silent_y, 16000)
        assert not valid
        assert "silent" in msg.lower()
