"""
Controlled Training and Fine-Tuning Pipeline for Model B (VoiceGuard AI).
Fine-tunes the Four-Block CNN + BiLSTM deepfake detection model:
- Strictly isolates held-out test split (data/splits/test_manifest.json remains locked).
- Uses group-aware train and validation splits (data/splits/train_split_manifest.json and val_manifest.json).
- Applies verified audio augmentations (gain variation, mild noise, time shift).
- Optimizes raw BCEWithLogitsLoss using AdamW with Cosine Annealing, weight decay, and gradient clipping.
- Evaluates validation loss and calibrates optimal decision threshold on validation data ONLY.
- Saves complete v2.0 checkpoint bundle to models/saved/fine_tuned_model_b.pt.
- Persists genuine training logs and history in reports/training_history.json.
"""

import os
import json
import time
import random
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, roc_curve

from models.architecture import DeepfakeCNNBiLSTMClassifier, load_classifier
from models.feature_extractor import AudioFeatureExtractor


class AudioDeepfakeDataset(Dataset):
    """
    Dataset loader for audio deepfake spectrogram tensors.
    Applies controlled audio augmentations during training only.
    """
    def __init__(self, manifest_path: str, augment: bool = False, extractor: Optional[AudioFeatureExtractor] = None):
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)
        self.augment = augment
        self.extractor = extractor or AudioFeatureExtractor(sr=16000, n_mels=64, n_fft=512, hop_length=256)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        file_path = rec["segment_path"]
        y, sr = sf.read(file_path, dtype="float32")

        # Label: REAL -> 0.0, FAKE -> 1.0
        label_str = rec["label"].strip().upper()
        if label_str not in ["REAL", "FAKE"]:
            raise ValueError(f"Unknown label '{label_str}' in {file_path}. Expected 'REAL' or 'FAKE'.")
        label_val = 1.0 if label_str == "FAKE" else 0.0

        # Augmentation during training only (conservative, label-preserving)
        if self.augment:
            # 1. Subtle random time shift (up to 0.1s)
            max_shift = int(sr * 0.1)
            shift = random.randint(-max_shift, max_shift)
            y = np.roll(y, shift)

            # 2. Subtle gain / amplitude scaling (+/- 1.5 dB -> ~0.85 to 1.15)
            gain = random.uniform(0.85, 1.15)
            y = y * gain

            # 3. Subtle Gaussian noise (SNR ~ 35 dB) with 50% probability
            if random.random() > 0.5:
                noise_amp = 0.002 * random.uniform(0.5, 1.0)
                noise = np.random.normal(0, noise_amp, len(y)).astype(np.float32)
                y = y + noise

        # Extract 64-band log-mel spectrogram tensor (1, 64, 188)
        spec = self.extractor.extract_mel_spectrogram(y)
        tensor = torch.from_numpy(spec).unsqueeze(0)  # (1, 64, T)
        label_tensor = torch.tensor([label_val], dtype=torch.float32)

        return tensor, label_tensor, rec["label"], rec.get("parent_file", "")


def calibrate_threshold_on_val(model: nn.Module, val_loader: DataLoader, device: torch.device) -> Tuple[float, Dict[str, float]]:
    """
    Calibrate optimal decision threshold exclusively on the validation split.
    Finds threshold that maximizes Youden's J statistic = TPR - FPR.
    """
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for tensors, labels, _, _ in val_loader:
            tensors = tensors.to(device)
            logits = model(tensors)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            targets = labels.cpu().numpy().flatten()
            all_probs.extend(probs)
            all_targets.extend(targets)

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    if len(set(all_targets)) < 2:
        return 0.5, {"val_accuracy": 50.0, "val_roc_auc": 0.5, "optimal_threshold": 0.5}

    fpr, tpr, thresholds = roc_curve(all_targets, all_probs)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_threshold = float(thresholds[best_idx])

    # Clamp threshold to reasonable range [0.1, 0.9] to prevent degeneracies
    best_threshold = max(0.1, min(0.9, best_threshold))

    val_preds = (all_probs >= best_threshold).astype(int)
    val_acc = float(accuracy_score(all_targets, val_preds) * 100.0)
    val_auc = float(roc_auc_score(all_targets, all_probs))

    metrics = {
        "val_accuracy": round(val_acc, 2),
        "val_roc_auc": round(val_auc, 4),
        "optimal_threshold": round(best_threshold, 4),
        "val_samples": len(all_targets)
    }
    return best_threshold, metrics


def train_model_b(
    train_manifest: Optional[str] = None,
    val_manifest: Optional[str] = None,
    output_weights: Optional[str] = None,
    initial_weights: Optional[str] = None,
    history_path: Optional[str] = None,
    epochs: int = 15,
    batch_size: int = 8,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Controlled fine-tuning of Model B (Four-Block CNN + BiLSTM) on group-aware train split.
    Validation data is strictly used for model selection and threshold calibration.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if train_manifest is None:
        train_manifest = os.path.join(base_dir, "data", "splits", "train_split_manifest.json")
    if val_manifest is None:
        val_manifest = os.path.join(base_dir, "data", "splits", "val_manifest.json")
    if output_weights is None:
        output_weights = os.path.join(base_dir, "models", "saved", "fine_tuned_model_b.pt")
    if initial_weights is None:
        candidate_init = os.path.join(base_dir, "models", "saved", "fine_tuned_model_b_lakhs.pt")
        initial_weights = candidate_init if os.path.exists(candidate_init) else None
    if history_path is None:
        history_path = os.path.join(base_dir, "reports", "training_history.json")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("=" * 60)
    print("  VOICEGUARD AI: CONTROLLED FINE-TUNING PIPELINE")
    print("=" * 60)
    print(f"Train Manifest:       {train_manifest}")
    print(f"Validation Manifest:  {val_manifest}")
    print(f"Output Checkpoint:    {output_weights}")
    print(f"Learning Rate:        {learning_rate} (AdamW, weight_decay={weight_decay})")
    print(f"Batch Size / Epochs:  {batch_size} / {epochs}")
    print(f"Random Seed:          {seed}")
    print("=" * 60)

    extractor = AudioFeatureExtractor()
    train_dataset = AudioDeepfakeDataset(train_manifest, augment=True, extractor=extractor)
    val_dataset = AudioDeepfakeDataset(val_manifest, augment=False, extractor=extractor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = DeepfakeCNNBiLSTMClassifier(num_mels=64, lstm_hidden=128, dropout_rate=0.35)

    if initial_weights and os.path.exists(initial_weights):
        print(f"[Train] Initializing weights from checkpoint: {initial_weights}")
        model.load_weights(initial_weights)
    else:
        print("[Train] Initializing fresh Four-Block CNN + BiLSTM weights...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Execution Device: {device} ({'GPU Active' if device.type == 'cuda' else 'CPU'})")
    model.to(device)

    # Class-weighted loss for balance
    train_labels = [r["label"] for r in train_dataset.records]
    n_real = sum(1 for l in train_labels if l == "REAL")
    n_fake = sum(1 for l in train_labels if l == "FAKE")
    pos_weight = torch.tensor([n_real / max(n_fake, 1)], dtype=torch.float32, device=device)
    print(f"[Train] Class balance: {n_real} REAL, {n_fake} FAKE | pos_weight: {pos_weight.item():.3f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "seed": seed,
        "epochs": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_roc_auc": [],
        "learning_rates": [],
        "best_epoch": 0,
        "best_val_loss": float("inf"),
        "best_threshold": 0.5,
    }

    best_val_loss = float("inf")
    best_threshold = 0.5
    best_state_dict = None

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for tensors, labels, _, _ in train_loader:
            tensors = tensors.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(tensors)
            loss = criterion(logits, labels)
            loss.backward()

            # Gradient clipping to stabilize BiLSTM
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_train_loss += loss.item() * tensors.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()

        epoch_train_loss = round(running_train_loss / max(train_total, 1), 4)
        epoch_train_acc = round(train_correct / max(train_total, 1) * 100.0, 2)
        curr_lr = scheduler.get_last_lr()[0]

        # Validation loop
        model.eval()
        running_val_loss = 0.0
        val_total = 0
        with torch.no_grad():
            for v_tensors, v_labels, _, _ in val_loader:
                v_tensors = v_tensors.to(device)
                v_labels = v_labels.to(device)
                v_logits = model(v_tensors)
                v_loss = criterion(v_logits, v_labels)
                running_val_loss += v_loss.item() * v_tensors.size(0)
                val_total += v_labels.size(0)

        epoch_val_loss = round(running_val_loss / max(val_total, 1), 4)
        opt_thresh, val_metrics = calibrate_threshold_on_val(model, val_loader, device)

        history["epochs"].append(epoch)
        history["train_loss"].append(epoch_train_loss)
        history["train_accuracy"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_accuracy"].append(val_metrics["val_accuracy"])
        history["val_roc_auc"].append(val_metrics["val_roc_auc"])
        history["learning_rates"].append(curr_lr)

        print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.1f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_metrics['val_accuracy']:.1f}% (Thresh: {opt_thresh:.4f})")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_threshold = opt_thresh
            history["best_epoch"] = epoch
            history["best_val_loss"] = best_val_loss
            history["best_threshold"] = best_threshold
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    total_duration = round(time.time() - start_time, 2)
    history["training_time_seconds"] = total_duration

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    model.decision_threshold = best_threshold

    # Save complete v2.0 checkpoint bundle
    model.save_checkpoint_bundle(
        path=output_weights,
        feature_config=extractor.get_config(),
        class_mapping={"REAL": 0, "FAKE": 1},
        metrics={
            "best_epoch": history["best_epoch"],
            "best_val_loss": history["best_val_loss"],
            "calibrated_threshold": history["best_threshold"],
            "training_time_seconds": total_duration,
        }
    )

    print(f"\n[Complete] Best validation model from epoch {history['best_epoch']} saved to: {output_weights}")
    print(f"           Calibrated validation threshold: {best_threshold:.4f}")
    print(f"           Training completed in {total_duration}s.")

    os.makedirs(os.path.dirname(os.path.abspath(history_path)), exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"           Training history saved to: {history_path}")

    return history


if __name__ == "__main__":
    train_model_b()
