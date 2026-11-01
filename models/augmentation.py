"""
Audio Augmentation Suite for VoiceGuard AI (Phase 3).
Provides:
1. Modernized RawBoost data augmentation (Odyssey 2022, Tak et al.):
   - ISD: Impulsive Signal-Dependent additive noise
   - LnL: Linear & Non-Linear convolutive noise (channel notch filter simulation)
   - SSI: Stationary Signal-Independent additive noise at calibrated SNR
2. Environmental Reverberation Simulation (early reflections + exponential decay tail)
3. Codec Simulation (lossy audio compression artifact emulation)
4. Calibrated Additive Noise (Gaussian / pink noise at target SNR)
5. Controlled Augmentation Pipeline with full NumPy 2.x compatibility.
"""

import copy
import io
from typing import Optional, Union, List, Tuple
import numpy as np
from scipy import signal
import soundfile as sf


def _rand_range(x1: float, x2: float, integer: bool = False) -> Union[int, float]:
    """Sample scalar uniform random value between x1 and x2."""
    val = float(np.random.uniform(low=x1, high=x2))
    return int(val) if integer else val


def normalize_waveform(x: np.ndarray, always: bool = False) -> np.ndarray:
    """Normalize waveform amplitude safely avoiding division by zero."""
    x = np.asarray(x, dtype=np.float32)
    max_val = float(np.amax(np.abs(x))) if len(x) > 0 else 0.0
    if max_val > 0.0:
        if always or max_val > 1.0:
            x = x / max_val
    return x.astype(np.float32)


# -------------------------------------------------------------
# 1. RawBoost Algorithms (Modernized for NumPy 2.x & Python 3.14)
# -------------------------------------------------------------

def gen_notch_coeffs(
    n_bands: int,
    min_f: float,
    max_f: float,
    min_bw: float,
    max_bw: float,
    min_coeff: int,
    max_coeff: int,
    min_g: float,
    max_g: float,
    fs: int = 16000
) -> np.ndarray:
    """Generate FIR notch filter coefficients for convolutive noise simulation."""
    b = np.array([1.0], dtype=np.float64)
    for _ in range(n_bands):
        fc = _rand_range(min_f, max_f, integer=False)
        bw = _rand_range(min_bw, max_bw, integer=False)
        c = _rand_range(min_coeff, max_coeff, integer=True)
        if c % 2 == 0:
            c += 1

        f1 = max(1e-3, fc - bw / 2.0)
        f2 = min(fs / 2.0 - 1e-3, fc + bw / 2.0)
        if f1 >= f2:
            continue

        band_fir = signal.firwin(c, [float(f1), float(f2)], window="hamming", fs=fs)
        b = np.convolve(band_fir, b)

    g = _rand_range(min_g, max_g, integer=False)
    _, h = signal.freqz(b, 1, fs=fs)
    max_h = np.amax(np.abs(h))
    if max_h > 1e-8:
        b = (10.0 ** (g / 20.0)) * b / max_h
    return b


def filter_fir(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Zero-phase centered FIR filter application."""
    N = b.shape[0] + 1
    xpad = np.pad(x, (0, N), mode="constant")
    y = signal.lfilter(b, 1, xpad)
    start = int(N // 2)
    end = start + len(x)
    return y[start:end]


def rawboost_lnl(
    x: np.ndarray,
    n_f: int = 5,
    n_bands: int = 5,
    min_f: float = 20.0,
    max_f: float = 7800.0,
    min_bw: float = 100.0,
    max_bw: float = 1000.0,
    min_coeff: int = 10,
    max_coeff: int = 100,
    min_g: float = -10.0,
    max_g: float = 10.0,
    min_bias: float = 5.0,
    max_bias: float = 10.0,
    fs: int = 16000
) -> np.ndarray:
    """Linear and non-linear convolutive noise (LnL)."""
    x = np.asarray(x, dtype=np.float32)
    y = np.zeros_like(x, dtype=np.float64)
    curr_min_g = min_g
    curr_max_g = max_g

    for i in range(n_f):
        if i == 1:
            curr_min_g -= min_bias
            curr_max_g -= max_bias
        b = gen_notch_coeffs(n_bands, min_f, max_f, min_bw, max_bw, min_coeff, max_coeff, curr_min_g, curr_max_g, fs)
        powered = np.power(x, i + 1)
        y += filter_fir(powered, b)

    y = y - np.mean(y)
    return normalize_waveform(y, always=False)


def rawboost_isd(x: np.ndarray, p_pct: float = 10.0, g_sd: float = 2.0) -> np.ndarray:
    """Impulsive signal-dependent additive noise (ISD)."""
    x = np.asarray(x, dtype=np.float32)
    beta = _rand_range(0.0, p_pct, integer=False)
    x_len = len(x)
    n = int(x_len * (beta / 100.0))
    if n == 0:
        return x

    y = x.copy()
    idx = np.random.choice(x_len, size=n, replace=False)
    f_r = ((2.0 * np.random.rand(n)) - 1.0) * ((2.0 * np.random.rand(n)) - 1.0)
    r = g_sd * x[idx] * f_r
    y[idx] = x[idx] + r
    return normalize_waveform(y, always=False)


def rawboost_ssi(
    x: np.ndarray,
    snr_min: float = 10.0,
    snr_max: float = 35.0,
    n_bands: int = 5,
    min_f: float = 20.0,
    max_f: float = 7800.0,
    min_bw: float = 100.0,
    max_bw: float = 1000.0,
    min_coeff: int = 10,
    max_coeff: int = 100,
    min_g: float = -10.0,
    max_g: float = 10.0,
    fs: int = 16000
) -> np.ndarray:
    """Stationary signal-independent additive noise (SSI)."""
    x = np.asarray(x, dtype=np.float32)
    noise = np.random.normal(0.0, 1.0, len(x))
    b = gen_notch_coeffs(n_bands, min_f, max_f, min_bw, max_bw, min_coeff, max_coeff, min_g, max_g, fs)
    noise = filter_fir(noise, b)
    noise = normalize_waveform(noise, always=True)

    snr = _rand_range(snr_min, snr_max, integer=False)
    norm_x = float(np.linalg.norm(x, 2))
    norm_noise = float(np.linalg.norm(noise, 2))
    if norm_noise > 1e-8 and norm_x > 1e-8:
        scale = (norm_x / norm_noise) / (10.0 ** (0.05 * snr))
        augmented = x + noise * scale
        return normalize_waveform(augmented, always=False)
    return x


# -------------------------------------------------------------
# 2. Environmental Reverberation Simulation
# -------------------------------------------------------------

def simulate_reverberation(
    x: np.ndarray,
    rt60: float = 0.3,
    early_reflections: int = 5,
    decay: float = 12.0,
    fs: int = 16000
) -> np.ndarray:
    """
    Simulate room acoustic reverberation with synthetic impulse response (RIR).
    Addresses the room reflections and microphone distance present in YouTube audio.
    """
    rir_len = int(rt60 * fs)
    if rir_len <= 10:
        return x

    t = np.linspace(0, rt60, rir_len, endpoint=False)
    # Exponential decay envelope
    envelope = np.exp(-decay * t)
    noise = np.random.normal(0, 1, rir_len)
    rir = envelope * noise

    # Add distinct early reflections in the first 40ms
    early_samples = int(0.04 * fs)
    for _ in range(early_reflections):
        idx = int(np.random.randint(1, max(2, early_samples)))
        rir[idx] += float(np.random.uniform(0.3, 0.7)) * float(np.random.choice([-1, 1]))

    rir[0] = 1.0  # Direct sound
    rir = rir / np.sum(np.abs(rir))

    reverberated = signal.fftconvolve(x, rir, mode="full")[:len(x)]
    return normalize_waveform(reverberated, always=False)


# -------------------------------------------------------------
# 3. Additive Gaussian Noise at Target SNR
# -------------------------------------------------------------

def add_gaussian_noise(x: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """Add zero-mean Gaussian noise at specified SNR (dB)."""
    x = np.asarray(x, dtype=np.float32)
    signal_power = np.mean(x ** 2)
    if signal_power < 1e-10:
        return x
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0.0, np.sqrt(noise_power), len(x))
    augmented = x + noise
    return normalize_waveform(augmented, always=False)


# -------------------------------------------------------------
# 4. Composite Audio Augmentor
# -------------------------------------------------------------

from models.channel_simulator import ChannelSimulator


class AudioAugmentor:
    """
    Configurable Audio Augmentation Engine for Anti-Spoofing Model Training.
    Supports controlled ablation and joint training across:
    - rawboost (ISD, LnL, SSI)
    - reverb (Acoustic room simulation)
    - noise (Calibrated additive noise)
    - channel_codec (WhatsApp Opus, 8kHz Cellular, VoIP Meeting)
    """
    def __init__(
        self,
        use_rawboost: bool = True,
        use_reverb: bool = True,
        use_noise: bool = True,
        use_channel_codec: bool = True,
        prob_apply: float = 0.7,
        fs: int = 16000
    ):
        self.use_rawboost = use_rawboost
        self.use_reverb = use_reverb
        self.use_noise = use_noise
        self.use_channel_codec = use_channel_codec
        self.prob_apply = prob_apply
        self.fs = fs

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply selected augmentations stochastically during training."""
        if np.random.rand() > self.prob_apply:
            return x

        augmented = x.copy()

        # 1. RawBoost: Convolutive or additive notch
        if self.use_rawboost and np.random.rand() < 0.6:
            algo = np.random.choice(["lnl", "isd", "ssi"])
            if algo == "lnl":
                augmented = rawboost_lnl(augmented, fs=self.fs)
            elif algo == "isd":
                augmented = rawboost_isd(augmented)
            elif algo == "ssi":
                augmented = rawboost_ssi(augmented, fs=self.fs)

        # 2. Reverberation: Room impulse response
        if self.use_reverb and np.random.rand() < 0.5:
            rt60 = float(np.random.uniform(0.15, 0.45))
            augmented = simulate_reverberation(augmented, rt60=rt60, fs=self.fs)

        # 3. Additive Gaussian noise
        if self.use_noise and np.random.rand() < 0.4:
            snr = float(np.random.uniform(15.0, 35.0))
            augmented = add_gaussian_noise(augmented, snr_db=snr)

        # 4. Multi-Codec Channel Degradation (WhatsApp Opus, 8kHz Cellular, VoIP Meeting)
        if self.use_channel_codec and np.random.rand() < 0.45:
            ch_choice = str(np.random.choice(["whatsapp_opus", "cellular_telephony", "voip_meeting"]))
            augmented, _ = ChannelSimulator.apply(augmented, channel_type=ch_choice, sr=self.fs)

        return normalize_waveform(augmented, always=False)
