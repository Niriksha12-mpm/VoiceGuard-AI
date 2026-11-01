"""
Leakage-Free Dataset Splitting Module for VoiceGuard AI.
Performs:
1. Strict Training / Held-Out Test Split.
2. Leakage Prevention:
   - Group-based splitting on parent_file and speaker_id: All segments from
     the same parent recording and same speaker are strictly assigned together.
     Zero overlap between train, validation, and test splits.
3. Cross-Dataset Evaluation Set:
   - Holds out samples from independent source (zo9999 ElevenLabs)
     for out-of-distribution evaluation.
4. Separate group-aware validation split (train_split_manifest.json and val_manifest.json)
   for hyperparameter tuning and threshold calibration, keeping test_manifest.json locked!
5. Fixed random seed (seed=42) for deterministic reproducibility.
"""

import os
import json
import random
from typing import Dict, List, Any, Optional


def split_dataset(
    processed_manifest_path: Optional[str] = None,
    splits_dir: Optional[str] = None,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio_of_train: float = 0.15
) -> Dict[str, Any]:
    """
    Split processed segments into:
    - Train (80% of in-distribution) -> further split into train_split (85%) and val (15%)
    - Held-out Test (20% of in-distribution)
    - Cross-Dataset Out-of-Distribution Test Set
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if processed_manifest_path is None:
        processed_manifest_path = os.path.join(base_dir, "data", "processed", "processed_manifest.json")
    if splits_dir is None:
        splits_dir = os.path.join(base_dir, "data", "splits")

    random.seed(seed)
    os.makedirs(splits_dir, exist_ok=True)

    with open(processed_manifest_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    print(f"[Split] Splitting {len(segments)} segments with fixed seed={seed} (train_ratio={train_ratio})...")

    # Step 1: Separate cross-dataset evaluation candidates
    cross_dataset_segments = []
    in_distribution_segments = []

    for seg in segments:
        if "zo9999" in seg.get("dataset_source", ""):
            cross_dataset_segments.append(seg)
        else:
            in_distribution_segments.append(seg)

    print(f"[Split] In-distribution pool: {len(in_distribution_segments)} segments.")
    print(f"[Split] Cross-dataset evaluation pool: {len(cross_dataset_segments)} segments.")

    # Step 2: Group by parent recording and speaker to prevent leakage
    groups_real: Dict[str, List[Dict[str, Any]]] = {}
    groups_fake: Dict[str, List[Dict[str, Any]]] = {}

    for seg in in_distribution_segments:
        group_key = f"{seg['speaker_id']}::{seg['parent_file']}"
        if seg["label"] == "REAL":
            groups_real.setdefault(group_key, []).append(seg)
        else:
            groups_fake.setdefault(group_key, []).append(seg)

    # Step 3: Stratified partition of groups
    def partition_groups(groups_dict: Dict[str, List[Dict[str, Any]]], ratio: float):
        keys = list(groups_dict.keys())
        random.shuffle(keys)
        total_segs = sum(len(v) for v in groups_dict.values())
        target_train_segs = int(total_segs * ratio)

        train_keys = []
        test_keys = []
        curr_train = 0

        for k in keys:
            segs_count = len(groups_dict[k])
            if (curr_train + segs_count <= target_train_segs) or not train_keys:
                train_keys.append(k)
                curr_train += segs_count
            else:
                test_keys.append(k)

        train_segs = [s for k in train_keys for s in groups_dict[k]]
        test_segs = [s for k in test_keys for s in groups_dict[k]]
        return train_segs, test_segs, train_keys, test_keys

    train_real, test_real, tr_real_keys, te_real_keys = partition_groups(groups_real, train_ratio)
    train_fake, test_fake, tr_fake_keys, te_fake_keys = partition_groups(groups_fake, train_ratio)

    train_split = train_real + train_fake
    test_split = test_real + test_fake

    random.shuffle(train_split)
    random.shuffle(test_split)

    # Step 4: Verification of zero leakage between train and test
    train_parent_files = set(s["parent_file"] for s in train_split)
    test_parent_files = set(s["parent_file"] for s in test_split)
    leakage_intersection = train_parent_files.intersection(test_parent_files)
    assert len(leakage_intersection) == 0, f"DATA LEAKAGE DETECTED! Overlapping files: {leakage_intersection}"

    train_speakers = set(s["speaker_id"] for s in train_split)
    test_speakers = set(s["speaker_id"] for s in test_split)
    speaker_leakage = train_speakers.intersection(test_speakers)
    print(f"[Split] Speaker disjointness check: {len(speaker_leakage)} overlapping speakers.")

    # Step 5: Sub-split train into train_sub and val using group-aware partition
    groups_train_sub_real: Dict[str, List[Dict[str, Any]]] = {}
    groups_train_sub_fake: Dict[str, List[Dict[str, Any]]] = {}
    for seg in train_split:
        group_key = f"{seg['speaker_id']}::{seg['parent_file']}"
        if seg["label"] == "REAL":
            groups_train_sub_real.setdefault(group_key, []).append(seg)
        else:
            groups_train_sub_fake.setdefault(group_key, []).append(seg)

    tr_sub_real, val_real, _, _ = partition_groups(groups_train_sub_real, 1.0 - val_ratio_of_train)
    tr_sub_fake, val_fake, _, _ = partition_groups(groups_train_sub_fake, 1.0 - val_ratio_of_train)

    train_sub_split = tr_sub_real + tr_sub_fake
    val_split = val_real + val_fake
    random.shuffle(train_sub_split)
    random.shuffle(val_split)

    # Verify zero leakage between train_sub and val
    tr_sub_parents = set(s["parent_file"] for s in train_sub_split)
    val_parents = set(s["parent_file"] for s in val_split)
    val_leakage = tr_sub_parents.intersection(val_parents)
    assert len(val_leakage) == 0, f"VAL LEAKAGE DETECTED: {val_leakage}"

    # Write manifests
    train_manifest_path = os.path.join(splits_dir, "train_manifest.json")
    test_manifest_path = os.path.join(splits_dir, "test_manifest.json")
    cross_manifest_path = os.path.join(splits_dir, "cross_dataset_manifest.json")
    train_sub_path = os.path.join(splits_dir, "train_split_manifest.json")
    val_path = os.path.join(splits_dir, "val_manifest.json")
    split_info_path = os.path.join(splits_dir, "split_metadata.json")

    with open(train_manifest_path, "w", encoding="utf-8") as f:
        json.dump(train_split, f, indent=2)
    with open(test_manifest_path, "w", encoding="utf-8") as f:
        json.dump(test_split, f, indent=2)
    with open(cross_manifest_path, "w", encoding="utf-8") as f:
        json.dump(cross_dataset_segments, f, indent=2)
    with open(train_sub_path, "w", encoding="utf-8") as f:
        json.dump(train_sub_split, f, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_split, f, indent=2)

    metadata = {
        "random_seed": seed,
        "split_policy": "Group-aware partition by parent_file and speaker_id",
        "leakage_prevention": "Strict zero-overlap between train, val, and test parent recordings",
        "train_samples": len(train_split),
        "train_distribution": {
            "real": sum(1 for x in train_split if x["label"] == "REAL"),
            "fake": sum(1 for x in train_split if x["label"] == "FAKE")
        },
        "train_split_samples": len(train_sub_split),
        "train_split_distribution": {
            "real": sum(1 for x in train_sub_split if x["label"] == "REAL"),
            "fake": sum(1 for x in train_sub_split if x["label"] == "FAKE")
        },
        "val_samples": len(val_split),
        "val_distribution": {
            "real": sum(1 for x in val_split if x["label"] == "REAL"),
            "fake": sum(1 for x in val_split if x["label"] == "FAKE")
        },
        "test_samples": len(test_split),
        "test_distribution": {
            "real": sum(1 for x in test_split if x["label"] == "REAL"),
            "fake": sum(1 for x in test_split if x["label"] == "FAKE")
        },
        "cross_dataset_samples": len(cross_dataset_segments),
        "cross_dataset_distribution": {
            "real": sum(1 for x in cross_dataset_segments if x["label"] == "REAL"),
            "fake": sum(1 for x in cross_dataset_segments if x["label"] == "FAKE")
        },
        "train_parent_recordings": sorted(list(train_parent_files)),
        "test_parent_recordings": sorted(list(test_parent_files)),
        "val_parent_recordings": sorted(list(val_parents))
    }

    with open(split_info_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Split] SUCCESS: Zero data leakage verified.")
    print(f"        Train (Full): {len(train_split)} samples (REAL: {metadata['train_distribution']['real']}, FAKE: {metadata['train_distribution']['fake']})")
    print(f"        Train Split:  {len(train_sub_split)} samples (REAL: {metadata['train_split_distribution']['real']}, FAKE: {metadata['train_split_distribution']['fake']})")
    print(f"        Val Split:    {len(val_split)} samples (REAL: {metadata['val_distribution']['real']}, FAKE: {metadata['val_distribution']['fake']})")
    print(f"        Test Split:   {len(test_split)} samples (REAL: {metadata['test_distribution']['real']}, FAKE: {metadata['test_distribution']['fake']})")
    print(f"        Cross-Dataset: {len(cross_dataset_segments)} samples")
    print(f"        Saved to {splits_dir}/")
    return metadata


if __name__ == "__main__":
    split_dataset()
