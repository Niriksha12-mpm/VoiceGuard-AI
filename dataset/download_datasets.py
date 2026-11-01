"""
Dataset Download and Ingestion Module.
Downloads genuine, verified audio samples from official open repositories:
1. WaveFake (RUB-SysSec / Zenodo benchmark) - Authentic human LJSpeech + Neural Vocoder Deepfakes
2. Gary Stafford Deepfake Audio (GitHub verified mirrors) - YouTube human speech + ElevenLabs, Hume AI, Speechify
3. zo9999 Deepfake Repository - ElevenLabs Aria & Rachel high-fidelity neural voice clones
4. Coqui TTS Corpus - Multi-format real human speech (WAV, MP3, FLAC) for format & duplicate testing

Preserves full metadata: source dataset, speaker/source ID, algorithm, format, and label (REAL or FAKE).
"""

import os
import json
import hashlib
import requests
import soundfile as sf
from typing import List, Dict, Any


VERIFIED_AUDIO_SOURCES = [
    # --- REAL SAMPLES (AUTHENTIC HUMAN SPEECH) ---
    {
        "url": "https://raw.githubusercontent.com/RUB-SysSec/WaveFake/main/tests/sample_data/LJSpeech/LJ001-0001.wav",
        "filename": "wavefake_real_lj001_0001.wav",
        "label": "REAL",
        "dataset_source": "WaveFake (LJSpeech Bonafide)",
        "speaker_id": "LJSpeech_Narrator",
        "algorithm": "Human Natural Speech",
        "format": "wav"
    },
    {
        "url": "https://raw.githubusercontent.com/RUB-SysSec/WaveFake/main/tests/sample_data/LJSpeech/LJ001-0002.wav",
        "filename": "wavefake_real_lj001_0002.wav",
        "label": "REAL",
        "dataset_source": "WaveFake (LJSpeech Bonafide)",
        "speaker_id": "LJSpeech_Narrator",
        "algorithm": "Human Natural Speech",
        "format": "wav"
    },
    {
        "url": "https://raw.githubusercontent.com/RUB-SysSec/WaveFake/main/tests/sample_data/LJSpeech/LJ001-0003.wav",
        "filename": "wavefake_real_lj001_0003.wav",
        "label": "REAL",
        "dataset_source": "WaveFake (LJSpeech Bonafide)",
        "speaker_id": "LJSpeech_Narrator",
        "algorithm": "Human Natural Speech",
        "format": "wav"
    },
    {
        "url": "https://raw.githubusercontent.com/RUB-SysSec/WaveFake/main/tests/sample_data/LJSpeech/LJ001-0004.wav",
        "filename": "wavefake_real_lj001_0004.wav",
        "label": "REAL",
        "dataset_source": "WaveFake (LJSpeech Bonafide)",
        "speaker_id": "LJSpeech_Narrator",
        "algorithm": "Human Natural Speech",
        "format": "wav"
    },
    {
        "url": "https://raw.githubusercontent.com/RUB-SysSec/WaveFake/main/tests/sample_data/LJSpeech/LJ001-0005.wav",
        "filename": "wavefake_real_lj001_0005.wav",
        "label": "REAL",
        "dataset_source": "WaveFake (LJSpeech Bonafide)",
        "speaker_id": "LJSpeech_Narrator",
        "algorithm": "Human Natural Speech",
        "format": "wav"
    },
    {
        "url": "https://raw.githubusercontent.com/garystafford/deepfake-voice-detection-public/main/audio_samples/yt_0001_part_007.flac",
        "filename": "garystafford_real_yt0001_part007.flac",
        "label": "REAL",
        "dataset_source": "Gary Stafford Deepfake Dataset (YouTube Human)",
        "speaker_id": "YT_Speaker_0001",
        "algorithm": "Human Natural Speech",
        "format": "flac"
    },
    {
        "url": "https://raw.githubusercontent.com/garystafford/deepfake-voice-detection-public/main/audio_samples/yt_0005_part_007.flac",
        "filename": "garystafford_real_yt0005_part007.flac",
        "label": "REAL",
        "dataset_source": "Gary Stafford Deepfake Dataset (YouTube Human)",
        "speaker_id": "YT_Speaker_0005",
        "algorithm": "Human Natural Speech",
        "format": "flac"
    },
    {
        "url": "https://raw.githubusercontent.com/garystafford/deepfake-voice-detection-public/main/audio_samples/yt_0006_part_053.flac",
        "filename": "garystafford_real_yt0006_part053.flac",
        "label": "REAL",
        "dataset_source": "Gary Stafford Deepfake Dataset (YouTube Human)",
        "speaker_id": "YT_Speaker_0006",
        "algorithm": "Human Natural Speech",
        "format": "flac"
    },
    {
        "url": "https://raw.githubusercontent.com/garystafford/deepfake-voice-detection-public/main/audio_samples/yt_0006_part_113.flac",
        "filename": "garystafford_real_yt0006_part113.flac",
        "label": "REAL",
        "dataset_source": "Gary Stafford Deepfake Dataset (YouTube Human)",
        "speaker_id": "YT_Speaker_0006",
        "algorithm": "Human Natural Speech",
        "format": "flac"
    },
    {
        "url": "https://raw.githubusercontent.com/garystafford/deepfake-voice-detection-public/main/audio_samples/yt_0006_part_142.flac",
        "filename": "garystafford_real_yt0006_part142.flac",
        "label": "REAL",
        "dataset_source": "Gary Stafford Deepfake Dataset (YouTube Human)",
        "speaker_id": "YT_Speaker_0006",
        "algorithm": "Human Natural Speech",
        "format": "flac"
    },
    # Coqui TTS authentic audio in multiple formats (WAV, MP3, FLAC)
    {
        "url": "https://raw.githubusercontent.com/coqui-ai/TTS/dev/tests/data/ljspeech/wavs/LJ001-0001.flac",
        "filename": "coqui_real_lj001_0001.flac",
        "label": "REAL",
        "dataset_source": "Coqui TTS Corpus (LJSpeech)",
        "speaker_id": "LJSpeech_Narrator",
        "algorithm": "Human Natural Speech",
        "format": "flac"
    },
    {
        "url": "https://raw.githubusercontent.com/coqui-ai/TTS/dev/tests/data/ljspeech/wavs/LJ001-0001.mp3",
        "filename": "coqui_real_lj001_0001.mp3",
        "label": "REAL",
        "dataset_source": "Coqui TTS Corpus (LJSpeech)",
        "speaker_id": "LJSpeech_Narrator",
        "algorithm": "Human Natural Speech",
        "format": "mp3"
    },
    {
        "url": "https://raw.githubusercontent.com/coqui-ai/TTS/dev/tests/data/ljspeech/wavs/LJ001-0002.wav",
        "filename": "coqui_real_lj001_0002.wav",
        "label": "REAL",
        "dataset_source": "Coqui TTS Corpus (LJSpeech)",
        "speaker_id": "LJSpeech_Narrator",
        "algorithm": "Human Natural Speech",
        "format": "wav"
    },
    {
        "url": "https://raw.githubusercontent.com/coqui-ai/TTS/dev/tests/data/ljspeech/wavs/LJ001-0002.mp3",
        "filename": "coqui_real_lj001_0002.mp3",
        "label": "REAL",
        "dataset_source": "Coqui TTS Corpus (LJSpeech)",
        "speaker_id": "LJSpeech_Narrator",
        "algorithm": "Human Natural Speech",
        "format": "mp3"
    },
    {
        "url": "https://raw.githubusercontent.com/coqui-ai/TTS/dev/tests/data/ljspeech/wavs/LJ001-0003.wav",
        "filename": "coqui_real_lj001_0003.wav",
        "label": "REAL",
        "dataset_source": "Coqui TTS Corpus (LJSpeech)",
        "speaker_id": "LJSpeech_Narrator",
        "algorithm": "Human Natural Speech",
        "format": "wav"
    },

    # --- FAKE SAMPLES (AI-GENERATED / SYNTHETIC / SPOOFED SPEECH) ---
    {
        "url": "https://raw.githubusercontent.com/RUB-SysSec/WaveFake/main/tests/sample_data/generated/LJ001-0001_gen.wav",
        "filename": "wavefake_fake_lj001_0001_gen.wav",
        "label": "FAKE",
        "dataset_source": "WaveFake (Neural Vocoder Benchmark)",
        "speaker_id": "LJSpeech_Impersonated",
        "algorithm": "MelGAN / Parallel WaveGAN",
        "format": "wav"
    },
    {
        "url": "https://raw.githubusercontent.com/RUB-SysSec/WaveFake/main/tests/sample_data/generated/LJ001-0002_gen.wav",
        "filename": "wavefake_fake_lj001_0002_gen.wav",
        "label": "FAKE",
        "dataset_source": "WaveFake (Neural Vocoder Benchmark)",
        "speaker_id": "LJSpeech_Impersonated",
        "algorithm": "MelGAN / Parallel WaveGAN",
        "format": "wav"
    },
    {
        "url": "https://raw.githubusercontent.com/RUB-SysSec/WaveFake/main/tests/sample_data/generated/LJ001-0003_gen.wav",
        "filename": "wavefake_fake_lj001_0003_gen.wav",
        "label": "FAKE",
        "dataset_source": "WaveFake (Neural Vocoder Benchmark)",
        "speaker_id": "LJSpeech_Impersonated",
        "algorithm": "MelGAN / Parallel WaveGAN",
        "format": "wav"
    },
    {
        "url": "https://raw.githubusercontent.com/RUB-SysSec/WaveFake/main/tests/sample_data/generated/LJ001-0004_gen.wav",
        "filename": "wavefake_fake_lj001_0004_gen.wav",
        "label": "FAKE",
        "dataset_source": "WaveFake (Neural Vocoder Benchmark)",
        "speaker_id": "LJSpeech_Impersonated",
        "algorithm": "MelGAN / Parallel WaveGAN",
        "format": "wav"
    },
    {
        "url": "https://raw.githubusercontent.com/RUB-SysSec/WaveFake/main/tests/sample_data/generated/LJ001-0005_gen.wav",
        "filename": "wavefake_fake_lj001_0005_gen.wav",
        "label": "FAKE",
        "dataset_source": "WaveFake (Neural Vocoder Benchmark)",
        "speaker_id": "LJSpeech_Impersonated",
        "algorithm": "MelGAN / Parallel WaveGAN",
        "format": "wav"
    },
    {
        "url": "https://raw.githubusercontent.com/garystafford/deepfake-voice-detection-public/main/audio_samples/el_0027_part_003.flac",
        "filename": "garystafford_fake_el0027_part003.flac",
        "label": "FAKE",
        "dataset_source": "Gary Stafford Deepfake Dataset (ElevenLabs)",
        "speaker_id": "ElevenLabs_Clone_0027",
        "algorithm": "ElevenLabs Instant Voice Cloning",
        "format": "flac"
    },
    {
        "url": "https://raw.githubusercontent.com/garystafford/deepfake-voice-detection-public/main/audio_samples/el_0027_part_004.flac",
        "filename": "garystafford_fake_el0027_part004.flac",
        "label": "FAKE",
        "dataset_source": "Gary Stafford Deepfake Dataset (ElevenLabs)",
        "speaker_id": "ElevenLabs_Clone_0027",
        "algorithm": "ElevenLabs Instant Voice Cloning",
        "format": "flac"
    },
    {
        "url": "https://raw.githubusercontent.com/garystafford/deepfake-voice-detection-public/main/audio_samples/el_0027_part_006.flac",
        "filename": "garystafford_fake_el0027_part006.flac",
        "label": "FAKE",
        "dataset_source": "Gary Stafford Deepfake Dataset (ElevenLabs)",
        "speaker_id": "ElevenLabs_Clone_0027",
        "algorithm": "ElevenLabs Instant Voice Cloning",
        "format": "flac"
    },
    {
        "url": "https://raw.githubusercontent.com/garystafford/deepfake-voice-detection-public/main/audio_samples/he_0006_part_001.flac",
        "filename": "garystafford_fake_he0006_part001.flac",
        "label": "FAKE",
        "dataset_source": "Gary Stafford Deepfake Dataset (Hume AI)",
        "speaker_id": "Hume_Synthetic_0006",
        "algorithm": "Hume AI Empathic Voice Interface",
        "format": "flac"
    },
    {
        "url": "https://raw.githubusercontent.com/garystafford/deepfake-voice-detection-public/main/audio_samples/sf_0006_part_001.flac",
        "filename": "garystafford_fake_sf0006_part001.flac",
        "label": "FAKE",
        "dataset_source": "Gary Stafford Deepfake Dataset (Speechify)",
        "speaker_id": "Speechify_Synthetic_0006",
        "algorithm": "Speechify AI Voice Generator",
        "format": "flac"
    },
    {
        "url": "https://raw.githubusercontent.com/zo9999/deepfake-audio-detector/main/audio/ElevenLabs_2025-01-24T05_38_43_Aria_pre_s50_sb75_se0_b_m2.flac",
        "filename": "zo9999_fake_elevenlabs_aria.flac",
        "label": "FAKE",
        "dataset_source": "zo9999 Deepfake Audio Repository",
        "speaker_id": "ElevenLabs_Aria",
        "algorithm": "ElevenLabs Neural Voice Cloning",
        "format": "flac"
    },
    {
        "url": "https://raw.githubusercontent.com/zo9999/deepfake-audio-detector/main/audio/ElevenLabs_2025-01-24T06_12_40_Rachel_pre_s50_sb75_se0_b_m2.flac",
        "filename": "zo9999_fake_elevenlabs_rachel.flac",
        "label": "FAKE",
        "dataset_source": "zo9999 Deepfake Audio Repository",
        "speaker_id": "ElevenLabs_Rachel",
        "algorithm": "ElevenLabs Neural Voice Cloning",
        "format": "flac"
    }
]


def download_all_datasets(raw_dir: str = "data/raw") -> List[Dict[str, Any]]:
    """Download verified audio samples into raw directory and return ingestion manifest."""
    os.makedirs(os.path.join(raw_dir, "real"), exist_ok=True)
    os.makedirs(os.path.join(raw_dir, "fake"), exist_ok=True)

    ingested_manifest = []

    print(f"[Downloader] Downloading {len(VERIFIED_AUDIO_SOURCES)} verified genuine audio samples...")

    for item in VERIFIED_AUDIO_SOURCES:
        subfolder = "real" if item["label"] == "REAL" else "fake"
        dest_path = os.path.join(raw_dir, subfolder, item["filename"])

        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            print(f"  [Cached] {item['filename']} already downloaded.")
        else:
            print(f"  [Downloading] {item['filename']} from {item['dataset_source']}...")
            try:
                resp = requests.get(item["url"], timeout=30)
                resp.raise_for_status()
                with open(dest_path, "wb") as f:
                    f.write(resp.content)
            except Exception as e:
                print(f"  [Error] Failed downloading {item['filename']}: {e}")
                continue

        # Verify file is valid audio
        try:
            info = sf.info(dest_path)
            with open(dest_path, "rb") as f:
                sha256 = hashlib.sha256(f.read()).hexdigest()

            record = {
                "filename": item["filename"],
                "file_path": os.path.abspath(dest_path),
                "label": item["label"],
                "dataset_source": item["dataset_source"],
                "speaker_id": item["speaker_id"],
                "algorithm": item["algorithm"],
                "format": item["format"],
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "duration_seconds": round(info.duration, 3),
                "frames": info.frames,
                "sha256_hash": sha256
            }
            ingested_manifest.append(record)
        except Exception as e:
            print(f"  [Warning] Audio read error on {dest_path}: {e}")

    manifest_path = os.path.join(raw_dir, "raw_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(ingested_manifest, f, indent=2)

    real_count = sum(1 for x in ingested_manifest if x["label"] == "REAL")
    fake_count = sum(1 for x in ingested_manifest if x["label"] == "FAKE")
    print(f"[Downloader] Completed! Total valid files: {len(ingested_manifest)} (REAL: {real_count}, FAKE: {fake_count})")
    print(f"[Downloader] Manifest saved to {manifest_path}")
    return ingested_manifest


if __name__ == "__main__":
    download_all_datasets()
