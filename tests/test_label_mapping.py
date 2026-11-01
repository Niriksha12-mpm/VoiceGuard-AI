"""
Automated Unit Tests for Explicit Label Mapping in VoiceGuard AI.
Verifies compliance with Phase 2:
1. Real Hugging Face schemas (WaveFake 'real_or_fake', ASVspoof 'label'/'key').
2. Maps genuine audio to 0.0 and spoofed audio to 1.0.
3. Rejects unknown or ambiguous labels with informative ValueError.
"""

import pytest
from colab_train_streaming import map_audio_label


class TestLabelMapping:
    def test_wavefake_genuine_labels(self):
        """WaveFake bonafide records must map strictly to 0.0 (REAL)."""
        dataset = "ajaykarthick/wavefake-audio"
        for lbl in ["real", "bonafide", "original", "human", "LJSpeech", "0"]:
            sample = {"real_or_fake": lbl}
            assert map_audio_label(sample, dataset) == 0.0

    def test_wavefake_spoofed_labels(self):
        """WaveFake WF1..WF6 vocoders and algorithms must map strictly to 1.0 (FAKE)."""
        dataset = "ajaykarthick/wavefake-audio"
        for lbl in ["WF1", "WF2", "WF3", "WF4", "WF5", "WF6", "melgan", "hifigan", "parallel_wavegan", "fake", "1"]:
            sample = {"real_or_fake": lbl}
            assert map_audio_label(sample, dataset) == 1.0

    def test_asvspoof_labels(self):
        """ASVspoof 2019 benchmark genuine vs spoofed mapping."""
        dataset = "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA"
        assert map_audio_label({"key": "bonafide"}, dataset) == 0.0
        assert map_audio_label({"label": "bonafide"}, dataset) == 0.0
        assert map_audio_label({"key": "spoof"}, dataset) == 1.0
        assert map_audio_label({"label": "spoof"}, dataset) == 1.0

    def test_unknown_label_rejection(self):
        """Ambiguous or unknown label must raise informative ValueError."""
        dataset = "ajaykarthick/wavefake-audio"
        with pytest.raises(ValueError) as exc_info:
            map_audio_label({"real_or_fake": "dog_bark_alien_sound"}, dataset)
        assert "Unknown or ambiguous label" in str(exc_info.value)

    def test_missing_label_field_rejection(self):
        """Records without any label key must raise informative ValueError."""
        dataset = "ajaykarthick/wavefake-audio"
        with pytest.raises(ValueError) as exc_info:
            map_audio_label({"irrelevant_field": 123}, dataset)
        assert "missing label/vocoder key" in str(exc_info.value)
