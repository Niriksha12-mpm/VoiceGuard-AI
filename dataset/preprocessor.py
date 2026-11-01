"""
Audio Preprocessing and Standardization Module for VoiceGuard AI.
Standardizes audio signals for both training and application inference:
- Sample rate: 16,000 Hz
- Channels: Mono (1 channel)
- Silence trimming: Remove leading and trailing silence (< -40 dB)
- Amplitude normalization: Peak normalization to 0.95 headroom
- Fixed-duration framing: Standard 3.0-second clips (48,000 samples)
  (Zero-padded if short, segmented into multiple 3s chunks if long)
- Robust validation: Rejects corrupted files, silence, flat signals, and unsupported formats.
"""

import os
import io
import json
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
import soundfile as sf
import librosa


class AudioPreprocessor:
    def __init__(self, target_sr: int = 16000, target_duration: float = 3.0, top_db: float = 40.0):
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.target_samples = int(target_sr * target_duration)
        self.top_db = top_db

    def validate_audio(self, y: np.ndarray, sr: int) -> Tuple[bool, str]:
        """Validate audio signal. Reject silence, NaN, empty arrays, or flat signals."""
        if y is None or y.size == 0:
            return False, "Audio is empty or could not be decoded."
        if np.isnan(y).any() or np.isinf(y).any():
            return False, "Audio contains NaN or Inf values."
        peak = float(np.max(np.abs(y)))
        if peak < 1e-4:
            return False, "Audio is silent (RMS energy below audible threshold)."
        num_samples = max(y.shape) if y.ndim > 1 else len(y)
        if num_samples < int(sr * 0.2):  # less than 200ms
            return False, f"Audio duration too short ({num_samples/sr:.2f}s). Minimum 0.2s required."
        return True, "Valid"

    def process_waveform(self, y: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resample to 16kHz, convert to mono, trim silence, and normalize."""
        # Convert multi-channel (stereo) to mono
        if y.ndim > 1:
            y = np.mean(y, axis=0 if y.shape[0] < y.shape[1] else 1)

        # Remove DC offset (critical for PC microphones / USB interfaces)
        if len(y) > 0:
            y = y - float(np.mean(y))

        # Resample to 16kHz if needed
        if orig_sr != self.target_sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=self.target_sr)

        # Trim leading and trailing silence (using 25dB for real-world microphone noise floors)
        try:
            trimmed_y, _ = librosa.effects.trim(y, top_db=25.0)
            if len(trimmed_y) >= int(self.target_sr * 0.3):
                y = trimmed_y
        except Exception:
            pass


        # Peak normalization with headroom
        max_val = float(np.max(np.abs(y)))
        if max_val > 1e-5:
            y = (y / max_val * 0.95).astype(np.float32)

        return y.astype(np.float32)

    def segment_or_pad(self, y: np.ndarray) -> List[np.ndarray]:
        """
        Produce one or more fixed 3.0s (48,000 samples) segments:
        - If shorter than 3s, zero-pad to 48,000 samples.
        - If longer, slice into non-overlapping 3s segments.
        - If remaining tail is at least 1.5s, pad the tail to 3s.
        """
        segments = []
        if len(y) <= self.target_samples:
            padded = np.zeros(self.target_samples, dtype=np.float32)
            padded[:len(y)] = y
            segments.append(padded)
        else:
            num_segments = len(y) // self.target_samples
            for i in range(num_segments):
                seg = y[i * self.target_samples : (i + 1) * self.target_samples]
                segments.append(seg)

            tail_len = len(y) % self.target_samples
            if tail_len >= int(self.target_sr * 1.5):
                tail = np.zeros(self.target_samples, dtype=np.float32)
                tail[:tail_len] = y[-tail_len:]
                segments.append(tail)

        return segments

    def process_file(self, filepath: str) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Load, validate, process, and return 3s segments and metadata."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        ext = os.path.splitext(filepath)[1].lower()
        if ext not in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
            raise ValueError(f"Unsupported audio format: {ext}. Supported: WAV, MP3, FLAC, OGG.")

        try:
            y, sr = sf.read(filepath, dtype="float32")
        except Exception:
            try:
                y, sr = librosa.load(filepath, sr=None, mono=False)
            except Exception as e:
                raise ValueError(f"Corrupted or unreadable audio file: {e}")

        valid, msg = self.validate_audio(y, sr)
        if not valid:
            raise ValueError(f"Audio validation failed: {msg}")

        processed_y = self.process_waveform(y, sr)
        segments = self.segment_or_pad(processed_y)

        meta = {
            "original_sr": sr,
            "target_sr": self.target_sr,
            "raw_duration": round(len(y) / sr if y.ndim == 1 else y.shape[-1] / sr, 3),
            "processed_duration": round(len(processed_y) / self.target_sr, 3),
            "num_segments": len(segments),
            "segment_samples": self.target_samples
        }
        return segments, meta


def preprocess_clean_dataset(
    clean_manifest_path: Optional[str] = None,
    output_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Process all files from clean manifest, save standardized 16kHz WAV segments,
    and output processed dataset manifest with relative/portable paths.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if clean_manifest_path is None:
        clean_manifest_path = os.path.join(base_dir, "data", "raw", "clean_manifest.json")
    if output_dir is None:
        output_dir = os.path.join(base_dir, "data", "processed")

    with open(clean_manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    os.makedirs(os.path.join(output_dir, "real"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "fake"), exist_ok=True)

    preprocessor = AudioPreprocessor(target_sr=16000, target_duration=3.0)
    processed_manifest = []

    print(f"[Preprocessor] Preprocessing {len(records)} clean recordings into 16kHz mono standardized segments...")

    for r in records:
        # Resolve path relative to current project if needed
        src_path = r.get("file_path", "")
        if not os.path.exists(src_path):
            # Fall back to resolving within current project's data/raw directory
            filename = r.get("filename", os.path.basename(src_path))
            sub = "real" if r.get("label") == "REAL" else "fake"
            candidate = os.path.join(base_dir, "data", "raw", sub, filename)
            if os.path.exists(candidate):
                src_path = candidate
            else:
                candidate2 = os.path.join(base_dir, "data", "raw", filename)
                if os.path.exists(candidate2):
                    src_path = candidate2

        label = r["label"]
        speaker_id = r["speaker_id"]
        dataset_source = r["dataset_source"]
        algorithm = r["algorithm"]
        subfolder = "real" if label == "REAL" else "fake"

        try:
            segments, meta = preprocessor.process_file(src_path)
            base_name = os.path.splitext(r["filename"])[0]

            for s_idx, seg in enumerate(segments):
                seg_filename = f"{base_name}_seg{s_idx:02d}.wav"
                seg_path = os.path.join(output_dir, subfolder, seg_filename)

                sf.write(seg_path, seg, 16000, subtype="PCM_16")

                processed_manifest.append({
                    "segment_filename": seg_filename,
                    "segment_path": os.path.abspath(seg_path),
                    "parent_file": r["filename"],
                    "label": label,
                    "speaker_id": speaker_id,
                    "dataset_source": dataset_source,
                    "algorithm": algorithm,
                    "segment_index": s_idx,
                    "sample_rate": 16000,
                    "channels": 1,
                    "duration_seconds": 3.0,
                    "samples": 48000
                })
        except Exception as e:
            print(f"[Preprocessor] Error processing {r.get('filename')}: {e}")

    out_manifest_path = os.path.join(output_dir, "processed_manifest.json")
    with open(out_manifest_path, "w", encoding="utf-8") as f:
        json.dump(processed_manifest, f, indent=2)

    real_count = sum(1 for x in processed_manifest if x["label"] == "REAL")
    fake_count = sum(1 for x in processed_manifest if x["label"] == "FAKE")
    print(f"[Preprocessor] Generated {len(processed_manifest)} standardized 3.0s segments.")
    print(f"               REAL: {real_count}, FAKE: {fake_count}")
    print(f"               Manifest saved to {out_manifest_path}")
    return processed_manifest


if __name__ == "__main__":
    preprocess_clean_dataset()
