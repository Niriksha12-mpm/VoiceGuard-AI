"""
Google Colab & Cloud GPU Training Script for XLS-R 300M + AASIST Anti-Spoofing.
Supports GPU acceleration, applying audio augmentations (RawBoost, RIR Reverberation, Additive Noise),
and fine-tuning the domain adaptation network.

Usage:
    python colab_w2v2_train.py --experiment exp_d2_reverb --epochs 25 --device cuda
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

from models.w2v2_aasist import W2V2AASISTClassifier, pad_fixed
from models.augmentation import AudioAugmentor, simulate_reverberation, add_gaussian_noise
from models.train_w2v2_adaptation import (
    W2V2DomainAdaptationHead,
    train_adaptation_head,
    evaluate_adaptation_model
)


def main():
    parser = argparse.ArgumentParser(description="Colab / Cloud GPU Trainer for XLS-R 300M + AASIST")
    parser.add_argument("--experiment", type=str, default="exp_d2_reverb",
                        choices=["exp_b_clean", "exp_c_augmented", "exp_d1_rawboost", "exp_d2_reverb", "exp_d3_noise"],
                        help="Experiment to train")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Compute device (cuda or cpu)")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save checkpoint")
    args = parser.parse_args()

    print("=" * 70)
    print("  VOICEGUARD AI: XLS-R 300M + AASIST CLOUD GPU TRAINER")
    print("=" * 70)
    print(f"Target Experiment:  {args.experiment}")
    print(f"Compute Device:     {args.device}")
    print(f"PyTorch Version:    {torch.__version__}")
    if args.device == "cuda" and torch.cuda.is_available():
        print(f"GPU Hardware:       {torch.cuda.get_device_name(0)}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    cached_dir = os.path.join(base_dir, "models", "saved", "w2v2_aasist")

    # Mapping experiment to cache file
    cache_map = {
        "exp_b_clean": "train_clean_features.npz",
        "exp_c_augmented": "train_aug_full_features.npz",
        "exp_d1_rawboost": "train_aug_rb_features.npz",
        "exp_d2_reverb": "train_aug_rev_features.npz",
        "exp_d3_noise": "train_aug_noise_features.npz"
    }

    clean_cache = os.path.join(cached_dir, "train_clean_features.npz")
    train_clean_data = np.load(clean_cache)
    train_clean_X, train_clean_y = train_clean_data["X"], train_clean_data["y"]

    val_cache = os.path.join(cached_dir, "val_features.npz")
    test_cache = os.path.join(cached_dir, "test_features.npz")
    val_data = np.load(val_cache)
    test_data = np.load(test_cache)

    if args.experiment == "exp_b_clean":
        train_X, train_y = train_clean_X, train_clean_y
    else:
        aug_cache = os.path.join(cached_dir, cache_map[args.experiment])
        aug_data = np.load(aug_cache)
        train_X = np.concatenate([train_clean_X, aug_data["X"]], axis=0)
        train_y = np.concatenate([train_clean_y, aug_data["y"]], axis=0)

    val_X, val_y = val_data["X"], val_data["y"]
    test_X, test_y = test_data["X"], test_data["y"]

    print(f"Loaded Train: {len(train_X)} samples | Val: {len(val_X)} samples | Test: {len(test_X)} samples")

    print("\nStarting Training & Validation Optimization...")
    model, cal_threshold, history = train_adaptation_head(
        train_X=train_X,
        train_y=train_y,
        val_X=val_X,
        val_y=val_y,
        epochs=args.epochs,
        lr=args.lr
    )

    print(f"\nTraining Complete! Calibrated Decision Threshold: {cal_threshold:.4f}")

    # Load test manifest for record metadata
    test_manifest_path = os.path.join(base_dir, "data", "splits", "test_manifest.json")
    with open(test_manifest_path, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    # Evaluate on held-out test partition
    eval_res = evaluate_adaptation_model(
        model=model,
        test_X=test_X,
        test_records=test_records,
        model_name=f"XLS-R 300M + AASIST ({args.experiment})",
        threshold=cal_threshold
    )

    metrics = eval_res["metrics"]
    print("\n" + "=" * 50)
    print(f"  HELD-OUT TEST EVALUATION METRICS ({args.experiment})")
    print("=" * 50)
    print(f"Accuracy:          {metrics['accuracy']:.2f}%")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.2f}%")
    print(f"Precision:         {metrics['precision']:.2f}%")
    print(f"Recall (Fake):     {metrics['recall']:.2f}%")
    print(f"F1-Score:          {metrics['f1_score']:.2f}%")
    print(f"ROC-AUC:           {metrics['roc_auc']:.4f}")
    print(f"Equal Error Rate:  {metrics['equal_error_rate']:.2f}%")
    print("=" * 50)

    # Save Checkpoint
    save_path = args.save_path or os.path.join(cached_dir, f"fine_tuned_{args.experiment}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved fine-tuned checkpoint to: {save_path}")


if __name__ == "__main__":
    main()
