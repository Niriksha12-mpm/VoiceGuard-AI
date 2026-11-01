"""
Unified Audio Feature Extraction Module for VoiceGuard AI.
Shared across streaming training, local training, validation, testing, and application inference
to guarantee 100% preprocessing and numerical consistency.

Acoustic Specifications:
- Sample Rate: 16,000 Hz mono
- FFT Window (n_fft): 512 samples (~32 ms)
- Hop Length: 256 samples (~16 ms step, producing ~188 frames for a 3.0s audio clip)
- Mel Filter Banks: 64 bands
- Frequency Range: 20 Hz to 8,000 Hz (Nyquist limit at 16kHz)
- Power Spectrogram: power=2.0
- Log Transform: Natural log compressed with clamp at 1e-5: log(max(mel, 1e-5))
- Normalization: Per-clip zero-mean, unit-variance standardization: (spec - mean) / (std + 1e-6)
- Model Input Shape: (batch, 1, 64, time_frames)
"""

import os
from typing import Dict, Any, Optional, Union
import numpy as np
import librosa
import torch
import torch.nn as nn


class AudioFeatureExtractor(nn.Module):
    """
    Standardized Feature Extractor for Voice Deepfake Detection.
    Converts 16,000 Hz audio waveforms to 64-band Log-Mel Spectrograms.
    """
    def __init__(
        self,
        sr: int = 16000,
        n_mels: int = 64,
        n_fft: int = 512,
        hop_length: int = 256,
        fmin: float = 20.0,
        fmax: float = 8000.0,
        power: float = 2.0,
        target_duration: float = 3.0,
    ):
        super().__init__()
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.power = power
        self.target_duration = target_duration
        self.target_samples = int(sr * target_duration)  # 48,000 samples for 3.0s

    def extract_mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """
        Extract Log Mel-Spectrogram normalized to zero mean, unit variance.
        Output shape: (n_mels, time_steps) -> (64, 188) for 3.0s audio at 16kHz.
        """
        if not isinstance(y, np.ndarray):
            y = np.asarray(y, dtype=np.float32)

        # Ensure 1D mono: if stereo (channels, samples) or (samples, channels)
        if y.ndim > 1:
            y = np.mean(y, axis=0 if y.shape[0] < y.shape[1] else 1)

        y = np.ascontiguousarray(y, dtype=np.float32)

        # Mel power spectrogram
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            power=self.power,
        )

        # Natural log compression with numerical clamp to prevent -inf
        log_mel = np.log(np.maximum(mel, 1e-5))

        # Per-clip zero-mean unit-variance standardization
        mean = float(np.mean(log_mel))
        std = float(np.std(log_mel))
        norm_mel = (log_mel - mean) / (std + 1e-6)

        return norm_mel.astype(np.float32)

    def extract_tensor(self, y: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Convert audio waveform to 4D PyTorch tensor:
        Shape: (1, 1, n_mels, time_steps)
        """
        if isinstance(y, torch.Tensor):
            y_np = y.detach().cpu().numpy()
        else:
            y_np = y

        spec = self.extract_mel_spectrogram(y_np)
        tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)  # (1, 1, 64, T)
        return tensor

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        PyTorch forward pass for batch processing:
        Accepts (B, samples) or (samples,) and returns (B, 1, n_mels, time_steps).
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (1, samples)

        batch_specs = []
        for b in range(waveform.shape[0]):
            audio_np = waveform[b].detach().cpu().numpy()
            spec = self.extract_mel_spectrogram(audio_np)
            batch_specs.append(torch.from_numpy(spec).unsqueeze(0))  # (1, 64, T)

        stacked = torch.stack(batch_specs, dim=0).to(waveform.device)  # (B, 1, 64, T)
        return stacked

    def compute_acoustic_diagnostics(self, y: np.ndarray) -> Dict[str, Any]:
        """
        Compute acoustic diagnostic metrics differentiating natural human speech
        from vocoded / synthetic deepfakes:
        - Spectral Centroid: center of mass of spectrum
        - Spectral Rolloff: frequency below which 85% of energy resides
        - Zero-Crossing Rate: rate of sign-changes in signal
        - Spectral Flatness: tone vs noise ratio
        """
        if not isinstance(y, np.ndarray):
            y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = np.mean(y, axis=0 if y.shape[0] < y.shape[1] else 1)

        try:
            centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=self.sr, hop_length=self.hop_length)))
            rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=self.sr, hop_length=self.hop_length)))
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y, hop_length=self.hop_length)))
            flatness = float(np.mean(librosa.feature.spectral_flatness(y=y, hop_length=self.hop_length)))
        except Exception:
            centroid, rolloff, zcr, flatness = 0.0, 0.0, 0.0, 0.0

        return {
            "spectral_centroid_hz": round(centroid, 1),
            "spectral_rolloff_hz": round(rolloff, 1),
            "zero_crossing_rate": round(zcr, 4),
            "spectral_flatness": round(flatness, 6)
        }

    def get_config(self) -> Dict[str, Any]:
        """Return extractor configuration dictionary for checkpoint bundling."""
        return {
            "sr": self.sr,
            "n_mels": self.n_mels,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "fmin": self.fmin,
            "fmax": self.fmax,
            "power": self.power,
            "target_duration": self.target_duration,
            "target_samples": self.target_samples,
        }


# Backward compatibility alias
MelSpectrogramExtractor = AudioFeatureExtractor
