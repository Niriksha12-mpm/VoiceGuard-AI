"""
Channel Simulator for VoiceGuard AI.
Simulates real-world audio channel degradations:
1. Clean: Direct pass-through
2. WhatsApp Voice Note (Opus 16-24 kbps): 7.2kHz lowpass bandlimiting + MDCT quantization noise
3. Cellular Telephony (8kHz G.711): 300-3400 Hz telephony bandpass + mu-law 8-bit quantization + 8kHz downsampling
4. VoIP Meeting: Burst packet dropouts (15-30ms) + room acoustic reflections
"""

from typing import Dict, Any, Tuple
import numpy as np
from scipy import signal


def _mu_law_companding(x: np.ndarray, mu: int = 255) -> np.ndarray:
    """Simulate 8-bit mu-law companding used in G.711 North American telephony."""
    x = np.clip(x, -1.0, 1.0)
    # Compression: sign(x) * ln(1 + mu*|x|) / ln(1 + mu)
    compressed = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    # 8-bit quantization: [-1, 1] mapped to 256 discrete levels
    quantized = np.round((compressed + 1.0) * 127.5) / 127.5 - 1.0
    # Expansion: sign(y) * (1/mu) * ((1 + mu)^|y| - 1)
    expanded = np.sign(quantized) * (1.0 / mu) * (np.power(1.0 + mu, np.abs(quantized)) - 1.0)
    return expanded.astype(np.float32)


class ChannelSimulator:
    """Applies controlled acoustic and transmission channel degradation."""

    SUPPORTED_CHANNELS = {
        "clean": "Direct Studio Audio (No channel degradation)",
        "whatsapp_opus": "WhatsApp Voice Note (Opus 16-24 kbps, 7.2kHz cutoff & quantization)",
        "cellular_telephony": "Cellular Phone Call (8kHz Telephony, G.711 300-3400Hz bandpass)",
        "voip_meeting": "VoIP Conference (Zoom/Teams packet jitter & room reflection)"
    }

    @classmethod
    def get_supported_channels(cls) -> Dict[str, str]:
        return cls.SUPPORTED_CHANNELS

    @classmethod
    def apply(cls, y: np.ndarray, channel_type: str = "clean", sr: int = 16000) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply channel simulation to a 1D audio waveform.
        Args:
            y: Audio waveform array at sample rate `sr` (default 16kHz).
            channel_type: 'clean', 'whatsapp_opus', 'cellular_telephony', or 'voip_meeting'.
            sr: Sample rate in Hz.
        Returns:
            Tuple of (degraded_waveform, metadata_dict).
        """
        y = np.asarray(y, dtype=np.float32).copy()
        if len(y) == 0:
            return y, {"channel": "clean", "applied": False}

        c_type = (channel_type or "clean").lower().strip()
        if c_type not in cls.SUPPORTED_CHANNELS or c_type == "clean":
            return y, {
                "channel": "clean",
                "name": "Direct Studio Audio",
                "applied": False,
                "description": "Original audio with no channel modifications."
            }

        if c_type == "whatsapp_opus":
            # 1. 7.2 kHz lowpass filter simulating Opus wideband voice codec
            nyquist = 0.5 * sr
            cutoff = min(7200.0, nyquist - 200.0)
            b, a = signal.butter(6, cutoff / nyquist, btype="low")
            y_filtered = signal.filtfilt(b, a, y)

            # 2. Add subtle MDCT quantization noise at ~28 dB SNR
            signal_power = float(np.mean(y_filtered ** 2))
            if signal_power > 1e-9:
                snr_linear = 10.0 ** (28.0 / 10.0)
                noise_std = np.sqrt(signal_power / snr_linear)
                noise = np.random.normal(0.0, noise_std, len(y_filtered))
                y_filtered = y_filtered + noise

            # 3. Normalize
            max_val = float(np.max(np.abs(y_filtered)))
            if max_val > 0.0:
                y_filtered = (y_filtered / max_val * 0.95).astype(np.float32)

            return y_filtered, {
                "channel": "whatsapp_opus",
                "name": "WhatsApp Voice Note",
                "applied": True,
                "description": "Opus audio codec emulation: 7.2kHz lowpass filter with 28dB SNR MDCT quantization noise."
            }

        elif c_type == "cellular_telephony":
            # 1. Standard telephony bandpass filter (300 Hz - 3400 Hz)
            nyquist = 0.5 * sr
            low = 300.0 / nyquist
            high = 3400.0 / nyquist
            b, a = signal.butter(4, [low, high], btype="bandpass")
            y_bandpassed = signal.filtfilt(b, a, y)

            # 2. Downsample to 8000 Hz (telephony sampling rate)
            num_8k = int(len(y_bandpassed) * 8000 / sr)
            if num_8k > 10:
                y_8k = signal.resample(y_bandpassed, num_8k)
                # 3. 8-bit G.711 mu-law companding & quantization
                y_companded = _mu_law_companding(y_8k)
                # 4. Upsample back to 16000 Hz
                y_resampled = signal.resample(y_companded, len(y))
            else:
                y_resampled = _mu_law_companding(y_bandpassed)

            max_val = float(np.max(np.abs(y_resampled)))
            if max_val > 0.0:
                y_resampled = (y_resampled / max_val * 0.95).astype(np.float32)

            return y_resampled, {
                "channel": "cellular_telephony",
                "name": "Cellular Phone Call",
                "applied": True,
                "description": "G.711 Telephony emulation: 8kHz downsampling, 300-3400Hz bandpass filter, and 8-bit mu-law companding."
            }

        elif c_type == "voip_meeting":
            # 1. Short burst packet dropouts (15-30ms) simulating UDP jitter
            y_voip = y.copy()
            dropout_samples = int(0.02 * sr)  # ~20ms
            if len(y_voip) > dropout_samples * 10:
                num_dropouts = min(3, len(y_voip) // (sr * 2))
                for _ in range(max(1, num_dropouts)):
                    start = np.random.randint(sr // 2, len(y_voip) - dropout_samples)
                    y_voip[start:start + dropout_samples] *= 0.05  # Severe attenuation

            # 2. Conference room early reflections (mild room reverberation)
            rir_len = int(0.12 * sr)  # 120ms RT60
            if rir_len > 10:
                t = np.linspace(0, 0.12, rir_len, endpoint=False)
                envelope = np.exp(-18.0 * t)
                noise = np.random.normal(0, 1, rir_len)
                rir = envelope * noise
                rir[0] = 1.0
                rir = rir / np.sum(np.abs(rir))
                y_voip = signal.fftconvolve(y_voip, rir, mode="full")[:len(y_voip)]

            max_val = float(np.max(np.abs(y_voip)))
            if max_val > 0.0:
                y_voip = (y_voip / max_val * 0.95).astype(np.float32)

            return y_voip, {
                "channel": "voip_meeting",
                "name": "VoIP Conference Call",
                "applied": True,
                "description": "VoIP emulation: 20ms UDP packet dropouts and room acoustic early reflections."
            }

        return y, {"channel": "clean", "name": "Direct Studio Audio", "applied": False, "description": ""}
