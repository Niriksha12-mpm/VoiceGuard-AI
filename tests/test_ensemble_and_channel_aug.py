"""
Automated Unit and Integration Tests for:
1. Option B: Multi-Codec Channel-Augmented Training (AudioAugmentor)
2. Option C: Multi-Model Soft-Voting Forensic Ensemble (InferenceEngine & REST API)
"""

import io
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from models.augmentation import AudioAugmentor
from backend.app import app, get_engine

client = TestClient(app)


def create_dummy_audio(duration: float = 3.0, sr: int = 16000) -> bytes:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = (0.3 * np.sin(2 * np.pi * 350 * t) + 0.2 * np.sin(2 * np.pi * 700 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    buf.seek(0)
    return buf.read()


def test_audio_augmentor_channel_codec():
    """Verify AudioAugmentor applies multi-codec channel degradation stochastically."""
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    x = (0.5 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)

    # Force channel codec application
    augmentor = AudioAugmentor(
        use_rawboost=False,
        use_reverb=False,
        use_noise=False,
        use_channel_codec=True,
        prob_apply=1.0,
        fs=sr
    )

    augmented = augmentor(x)
    assert isinstance(augmented, np.ndarray)
    assert len(augmented) == len(x)
    assert not np.isnan(augmented).any()
    assert not np.isinf(augmented).any()


def test_predict_w2v2_ensemble_engine():
    """Verify InferenceEngine produces consensus soft voting and ensemble breakdown."""
    eng = get_engine()
    audio_bytes = create_dummy_audio(duration=3.0)

    res = eng.predict(audio_bytes, "ensemble_test.wav", model_type="w2v2_ensemble")

    assert "prediction" in res
    pred = res["prediction"]
    assert "Ensemble" in pred["model_used"]
    assert pred["label"] in ["REAL", "FAKE"]
    assert 0.0 <= pred["confidence_score"] <= 1.0

    # Verify ensemble breakdown
    assert "ensemble_breakdown" in pred
    breakdown = pred["ensemble_breakdown"]
    assert breakdown is not None
    assert "consensus_status" in breakdown
    assert breakdown["consensus_status"] in ["3/3 Unanimous", "2/3 Majority Consensus"]
    assert len(breakdown["models"]) == 3

    # Check weights and votes of sub-models
    model_names = [m["name"] for m in breakdown["models"]]
    assert "Large-Scale Cloud Head" in model_names
    assert "Reverb & RawBoost Head" in model_names
    assert "TakHemlata Baseline" in model_names

    for m in breakdown["models"]:
        assert m["vote"] in ["REAL", "FAKE"]
        assert 0.0 <= m["spoof_prob"] <= 1.0

    # Verify temporal analysis is computed
    assert "temporal_analysis" in res
    assert len(res["temporal_analysis"]["timeline"]) >= 1


def test_api_predict_ensemble_endpoint():
    """Verify /api/predict accepts w2v2_ensemble and returns full ensemble payload."""
    audio_bytes = create_dummy_audio(duration=2.5)
    files = {"file": ("api_test.wav", audio_bytes, "audio/wav")}

    response = client.post("/api/predict?model_type=w2v2_ensemble", files=files)
    assert response.status_code == 200
    data = response.json()

    assert data["prediction"]["model_used"] == "VoiceGuard Forensic Ensemble (Multi-Head Consensus)"
    assert data["prediction"]["ensemble_breakdown"] is not None
    assert "consensus_status" in data["prediction"]["ensemble_breakdown"]
