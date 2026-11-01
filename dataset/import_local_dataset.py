"""
Local Dataset Importer Module.
Enables direct offline import of existing large-scale audio datasets (e.g., ASVspoof 2019,
WaveFake, FoR, or custom real/fake folders) without downloading over the internet.

Usage examples:
  1. Generic Real/Fake folder structure:
     python dataset/import_local_dataset.py --source "D:/my_dataset" --max_samples 5000

  2. ASVspoof 2019 directory:
     python dataset/import_local_dataset.py --source "D:/ASVspoof2019" --format asvspoof --max_samples 5000

  3. WaveFake directory:
     python dataset/import_local_dataset.py --source "D:/WaveFake" --format wavefake --max_samples 5000
"""

import os
import sys
import glob
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional


def scan_generic_folder(source_dir: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Scans a folder that contains 'real' (or bonafide/human) and 'fake' (or spoof/synthetic/ai) subdirectories.
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    records = []

    # Map possible folder names to REAL / FAKE
    label_mapping = {
        "real": "REAL",
        "human": "REAL",
        "bonafide": "REAL",
        "original": "REAL",
        "fake": "FAKE",
        "spoof": "FAKE",
        "synthetic": "FAKE",
        "ai": "FAKE",
        "deepfake": "FAKE",
        "generated": "FAKE"
    }

    real_files = []
    fake_files = []

    for file_path in source_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
            # Check parent directory names for label cues
            parts = [p.lower() for p in file_path.parts]
            assigned_label = None
            for part in reversed(parts[:-1]):
                for cue, norm_label in label_mapping.items():
                    if cue in part:
                        assigned_label = norm_label
                        break
                if assigned_label:
                    break

            if assigned_label == "REAL":
                real_files.append(file_path)
            elif assigned_label == "FAKE":
                fake_files.append(file_path)

    print(f"[Import] Found {len(real_files)} REAL and {len(fake_files)} FAKE candidate audio files in {source_dir}")

    if not real_files and not fake_files:
        print("[Import] No explicit 'real' or 'fake' subfolders detected. Scanning all audio files...")
        all_files = [f for f in source_path.rglob("*") if f.is_file() and f.suffix.lower() in audio_extensions]
        # Treat first half as REAL, second half as FAKE or prompt
        half = len(all_files) // 2
        real_files = all_files[:half]
        fake_files = all_files[half:]
        print(f"[Import] Default split applied: {len(real_files)} REAL, {len(fake_files)} FAKE.")

    # Balance and cap to max_samples
    if max_samples:
        per_class = max_samples // 2
        real_files = real_files[:per_class]
        fake_files = fake_files[:per_class]

    for f in real_files:
        records.append({
            "source_path": str(f.resolve()),
            "filename": f.name,
            "label": "REAL",
            "speaker_id": f.parent.name,
            "dataset_source": "Local-Import",
            "algorithm": "Human-Speech"
        })

    for f in fake_files:
        records.append({
            "source_path": str(f.resolve()),
            "filename": f.name,
            "label": "FAKE",
            "speaker_id": f.parent.name,
            "dataset_source": "Local-Import",
            "algorithm": "Synthetic/Vocoded"
        })

    return records


def scan_asvspoof(source_dir: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Scans an ASVspoof 2019 Logical Access (LA) directory using its protocol file."""
    source_path = Path(source_dir)
    protocol_files = list(source_path.rglob("*.txt"))
    records = []

    audio_dir = None
    for p in source_path.rglob("flac"):
        if p.is_dir():
            audio_dir = p
            break
    if not audio_dir:
        for p in source_path.rglob("*.flac"):
            audio_dir = p.parent
            break

    # Look for protocol lines: SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY (bonafide / spoof)
    for proto in protocol_files:
        try:
            with open(proto, "r", encoding="utf-8") as pf:
                for line in pf:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        speaker = parts[0]
                        audio_name = parts[1]
                        system_id = parts[3]
                        key = parts[4].lower()

                        if key in ["bonafide", "real"]:
                            label = "REAL"
                        elif key in ["spoof", "fake"]:
                            label = "FAKE"
                        else:
                            continue

                        # Check if audio file exists
                        candidates = [
                            source_path / f"{audio_name}.flac",
                            source_path / f"{audio_name}.wav",
                            audio_dir / f"{audio_name}.flac" if audio_dir else None,
                            audio_dir / f"{audio_name}.wav" if audio_dir else None
                        ]
                        found_path = next((c for c in candidates if c and c.exists()), None)
                        if found_path:
                            records.append({
                                "source_path": str(found_path.resolve()),
                                "filename": found_path.name,
                                "label": label,
                                "speaker_id": speaker,
                                "dataset_source": "ASVspoof2019-LA",
                                "algorithm": "Human" if label == "REAL" else system_id
                            })
        except Exception:
            continue

    if records and max_samples:
        real_recs = [r for r in records if r["label"] == "REAL"][:max_samples // 2]
        fake_recs = [r for r in records if r["label"] == "FAKE"][:max_samples // 2]
        records = real_recs + fake_recs

    return records


def import_dataset(source_dir: str, format_type: str = "generic", max_samples: Optional[int] = 5000,
                   target_raw_dir: str = "data/raw"):
    """Imports files from a local directory into data/raw/ without downloading."""
    os.makedirs(os.path.join(target_raw_dir, "real"), exist_ok=True)
    os.makedirs(os.path.join(target_raw_dir, "fake"), exist_ok=True)

    print(f"\n=======================================================")
    print(f"       OFFLINE DATASET IMPORTER (Zero Download)")
    print(f"=======================================================")
    print(f" Source:      {source_dir}")
    print(f" Format:      {format_type}")
    print(f" Max Samples: {max_samples}")

    if format_type.lower() == "asvspoof":
        records = scan_asvspoof(source_dir, max_samples)
    else:
        records = scan_generic_folder(source_dir, max_samples)

    if not records:
        print(f"[Import Error] No audio files could be matched from {source_dir}.")
        print("Please ensure your directory contains .wav, .mp3, or .flac files organized by 'real' and 'fake' folders.")
        return

    print(f"[Import] Selected {len(records)} files for import.")
    real_cnt = sum(1 for r in records if r["label"] == "REAL")
    fake_cnt = sum(1 for r in records if r["label"] == "FAKE")
    print(f"         REAL: {real_cnt}, FAKE: {fake_cnt}")

    manifest = []
    print("[Import] Linking/copying files into data/raw/...")

    for i, r in enumerate(records):
        src = r["source_path"]
        subfolder = "real" if r["label"] == "REAL" else "fake"
        dest_filename = f"imported_{i:06d}_{os.path.basename(src)}"
        dest_path = os.path.join(target_raw_dir, subfolder, dest_filename)

        # Create hard link or copy
        try:
            if not os.path.exists(dest_path):
                # Try creating hard link first to save disk space
                try:
                    os.link(src, dest_path)
                except Exception:
                    shutil.copy2(src, dest_path)
        except Exception as e:
            print(f"[Import Error] Failed to link/copy {src}: {e}")
            continue

        manifest.append({
            "filename": dest_filename,
            "file_path": os.path.abspath(dest_path),
            "label": r["label"],
            "speaker_id": r["speaker_id"],
            "dataset_source": r["dataset_source"],
            "algorithm": r["algorithm"]
        })

    manifest_path = os.path.join(target_raw_dir, "clean_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[Import Complete] Successfully imported {len(manifest)} audio files.")
    print(f"                  Manifest saved to {manifest_path}")
    print("\nNext Steps to Preprocess & Train:")
    print("  1. python dataset/preprocessor.py")
    print("  2. python dataset/split_data.py")
    print("  3. python -m models.train --epochs 10 --batch_size 16")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Local Dataset Importer")
    parser.add_argument("--source", type=str, required=True, help="Path to local dataset directory")
    parser.add_argument("--format", type=str, default="generic", choices=["generic", "asvspoof", "wavefake"],
                        help="Dataset format: generic (folders real/fake), asvspoof, or wavefake")
    parser.add_argument("--max_samples", type=int, default=5000, help="Maximum number of audio files to import")
    args = parser.parse_args()

    import_dataset(args.source, args.format, args.max_samples)
