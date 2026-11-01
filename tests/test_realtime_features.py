"""
Tests for Real-Time Features:
1. ChannelSimulator (clean, whatsapp_opus, cellular_telephony, voip_meeting)
2. Temporal Splicing & Sliding-Window Analysis
3. End-to-End API /api/predict with channel_sim and /api/channels
"""

import numpy as np
import pytest
import io
import soundfile as sf
from fastapi.testclient import TestClient

from models.channel_simulator import ChannelSimulator
from backend.app import app, get_engine

client = TestClient(app)


def create_dummy_audio_bytes(duration_sec: float = 3.0, sr: int = 16000) -> bytes:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # Sine wave with harmonics
    waveform = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, waveform, sr, format="WAV")
    buf.seek(0)
    return buf.read()


def test_channel_simulator_transforms():
    """Verify all 4 channel simulations execute and maintain valid audio output."""
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    orig = (0.4 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    channels = ["clean", "whatsapp_opus", "cellular_telephony", "voip_meeting"]
    for ch in channels:
        out, meta = ChannelSimulator.apply(orig, channel_type=ch, sr=sr)
        assert isinstance(out, np.ndarray), f"Failed for {ch}"
        assert len(out) == len(orig), f"Length mismatch for {ch}"
        assert not np.isnan(out).any(), f"NaN detected in {ch}"
        assert not np.isinf(out).any(), f"Inf detected in {ch}"
        assert meta["channel"] == ch
        if ch != "clean":
            assert meta["applied"] is True


def test_temporal_sliding_window_timeline():
    """Verify sliding-window temporal timeline generation on multi-second audio."""
    eng = get_engine()
    sr = 16000
    # 5.0 second audio -> multiple windows
    audio_bytes = create_dummy_audio_bytes(duration_sec=5.0, sr=sr)
    res = eng.predict(audio_bytes, "test_5s.wav", model_type="w2v2_large_scale", channel_sim="clean")

    assert "temporal_analysis" in res
    temporal = res["temporal_analysis"]
    assert "timeline" in temporal
    assert len(temporal["timeline"]) >= 3
    assert "splicing_detected" in temporal
    assert "splicing_summary" in temporal

    # Check structure of timeline segments
    first_seg = temporal["timeline"][0]
    assert "start_time" in first_seg
    assert "end_time" in first_seg
    assert "probability_fake" in first_seg
    assert first_seg["label"] in ["REAL", "FAKE"]


def test_api_predict_with_channel_sim():
    """Verify /api/predict accepts channel_sim and returns new diagnostic fields."""
    audio_bytes = create_dummy_audio_bytes(duration_sec=3.0)
    files = {"file": ("test_phone.wav", audio_bytes, "audio/wav")}
    response = client.post("/api/predict?model_type=w2v2_large_scale&channel_sim=cellular_telephony", files=files)
    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert data["prediction"]["channel_simulation"]["channel"] == "cellular_telephony"
    assert data["prediction"]["channel_simulation"]["applied"] is True
    assert "temporal_analysis" in data
    assert len(data["temporal_analysis"]["timeline"]) >= 1


def test_api_channels_endpoint():
    """Verify /api/channels returns registered channel list."""
    response = client.get("/api/channels")
    assert response.status_code == 200
    data = response.json()
    assert "channels" in data
    assert "whatsapp_opus" in data["channels"]
    assert "cellular_telephony" in data["channels"]
    assert "voip_meeting" in data["channels"]
