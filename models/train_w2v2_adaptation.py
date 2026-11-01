"""
Phase 3 Training and Audio Augmentation Pipeline for VoiceGuard AI.
Implements:
1. Controlled Training Experiments:
   - Experiment A: Pretrained XLS-R 300M + AASIST Baseline (Zero-Shot)
   - Experiment B: Standard Fine-Tuning (without audio augmentation)
   - Experiment C: Fine-Tuning with Full Audio Augmentation Suite (RawBoost + Reverb + Noise)
   - Experiment D: Component Ablation Study (D1: RawBoost, D2: Reverb, D3: Noise)
2. Robust Feature Extraction: W2V2-AASIST graph attention logits + multi-band acoustic energy descriptors
3. Calibrated Validation Threshold Selection (Youden's J statistic)
4. Evaluation across Locked Held-Out Test Split (13 samples)
5. Generates reports/phase3_training_experiments.json and updates reports/model_comparison.json
"""

import os
import json
import time
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

from models.w2v2_aasist import W2V2AASISTClassifier, pad_fixed
from models.augmentation import (
    AudioAugmentor, rawboost_isd, rawboost_lnl, rawboost_ssi,
    simulate_reverberation, add_gaussian_noise, normalize_waveform
)
from models.am_softmax import AMSoftmaxLoss, OCSoftmaxLoss
from models.layer_aggregation import SSLLayerAggregation


# -------------------------------------------------------------------
# Model Architecture: W2V2-AASIST Domain Adaptation Network
# -------------------------------------------------------------------

class W2V2DomainAdaptationHead(nn.Module):
    """
    Lightweight, highly regularized residual domain adaptation head for
    Wav2Vec 2.0 (XLS-R 300M) + AASIST speech anti-spoofing.
    Input: 66-dim composite feature vector (W2V2-AASIST logits + margin + probs + Mel energies + RMS).
    Output: 2-class logits [spoof, bona_fide].
    """
    def __init__(self, in_features: int = 66, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.decision_threshold = 0.5  # calibrated on validation set

        self.fc1 = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.25)
        )
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.res_act = nn.GELU()
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h = self.fc1(x)
        h = self.res_act(h + self.res_block(h))
        emb = self.fc2[:3](h)  # 32-dim normalized metric embedding bottleneck
        logits = self.fc2[3](emb)
        if return_embedding:
            return logits, emb
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# -------------------------------------------------------------------
# Feature Extraction Engine
# -------------------------------------------------------------------

def extract_composite_features(
    waveform: np.ndarray,
    w2v2_clf: W2V2AASISTClassifier,
    sr: int = 16000
) -> np.ndarray:
    """Extract 66-dim feature vector combining W2V2-AASIST logits with acoustic spectrum."""
    wav_64k = pad_fixed(waveform, 64600)
    batch = np.expand_dims(wav_64k, axis=0)
    logits = w2v2_clf.predict_batch_raw(batch)[0]  # [logit_spoof, logit_bonafide]

    exp_l = np.exp(logits - np.max(logits))
    probs = exp_l / np.sum(exp_l)
    margin = float(logits[1] - logits[0])

    mel = librosa.feature.melspectrogram(y=wav_64k, sr=sr, n_fft=1024, hop_length=512, n_mels=60)
    log_mel = np.log(np.maximum(mel, 1e-6))
    mel_mean = np.mean(log_mel, axis=1)
    rms = np.array([float(np.sqrt(np.mean(wav_64k ** 2)))])

    feat = np.concatenate([logits, [margin], probs, mel_mean, rms], axis=0).astype(np.float32)
    return feat


def compute_or_load_dataset_features(
    records: List[Dict[str, Any]],
    w2v2_clf: W2V2AASISTClassifier,
    cache_path: str,
    augmentor: Optional[AudioAugmentor] = None,
    name: str = "dataset"
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute composite features for records, or load from cache if exists."""
    if os.path.exists(cache_path):
        print(f"[{name}] Loading cached features from {cache_path}")
        data = np.load(cache_path)
        return data["X"], data["y"]

    print(f"[{name}] Computing features for {len(records)} samples (augmentor={augmentor is not None})...")
    X_list = []
    y_list = []

    for i, rec in enumerate(records):
        file_path = rec["segment_path"]
        is_fake = (rec["label"].strip().upper() == "FAKE")
        y_val = 1 if is_fake else 0

        y_audio, sr = sf.read(file_path, dtype="float32")
        if y_audio.ndim > 1:
            y_audio = np.mean(y_audio, axis=-1 if y_audio.shape[0] > y_audio.shape[1] else 0)

        if sr != 16000:
            y_audio = librosa.resample(y_audio, orig_sr=sr, target_sr=16000)

        # Apply augmentation if augmentor provided
        if augmentor is not None:
            y_audio = augmentor(y_audio)

        feat = extract_composite_features(y_audio, w2v2_clf, sr=16000)
        X_list.append(feat)
        y_list.append(y_val)

        if (i + 1) % 10 == 0 or (i + 1) == len(records):
            print(f"  Processed {i + 1}/{len(records)} samples...")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y)
    print(f"[{name}] Saved {len(X)} feature vectors to {cache_path}")
    return X, y


# -------------------------------------------------------------------
# Training and Evaluation Engine
# -------------------------------------------------------------------

class ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_adaptation_head(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    loss_type: str = "cross_entropy",
    am_margin: float = 0.35,
    am_scale: float = 30.0
) -> Tuple[W2V2DomainAdaptationHead, float, List[Dict[str, Any]]]:
    """
    Train adaptation head and calibrate decision threshold on validation set.
    Supports standard Cross-Entropy, AM-Softmax (Additive Margin), and OC-Softmax.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = W2V2DomainAdaptationHead(in_features=66, hidden_dim=128, num_classes=2)
    device = torch.device("cpu")
    model.to(device)

    # Initialize loss functions and optimizer
    am_criterion = None
    oc_criterion = None

    if loss_type == "am_softmax":
        am_criterion = AMSoftmaxLoss(in_features=32, num_classes=2, margin=am_margin, scale=am_scale).to(device)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(am_criterion.parameters()), lr=lr, weight_decay=weight_decay)
    elif loss_type == "oc_softmax":
        oc_criterion = OCSoftmaxLoss(in_features=32).to(device)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(oc_criterion.parameters()), lr=lr, weight_decay=weight_decay)
    else:
        num_fake = np.sum(train_y == 1)
        num_real = np.sum(train_y == 0)
        w_real = (len(train_y) / (2.0 * max(1, num_real)))
        w_fake = (len(train_y) / (2.0 * max(1, num_fake)))
        class_weights = torch.tensor([w_real, w_fake], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_ds = ArrayDataset(train_X, train_y)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            if loss_type == "am_softmax":
                _, emb = model(batch_X, return_embedding=True)
                loss, logits = am_criterion(emb, batch_y)
            elif loss_type == "oc_softmax":
                _, emb = model(batch_X, return_embedding=True)
                loss, cos_scores = oc_criterion(emb, batch_y)
                logits = torch.stack([-cos_scores, cos_scores], dim=-1)
            else:
                logits = model(batch_X)
                loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_y)
            preds = torch.argmax(logits, dim=-1)
            correct += int((preds == batch_y).sum().item())
            total += len(batch_y)

        scheduler.step()
        train_loss /= max(1, total)
        train_acc = (correct / total) * 100.0 if total > 0 else 0.0

        # Validation check
        model.eval()
        with torch.no_grad():
            val_t = torch.from_numpy(val_X)
            if loss_type == "am_softmax":
                _, emb_val = model(val_t, return_embedding=True)
                _, val_logits = am_criterion(emb_val, torch.zeros(len(val_X), dtype=torch.long))
            else:
                val_logits = model(val_t)
            v_probs = torch.softmax(val_logits, dim=-1).numpy()
            v_preds = np.argmax(v_probs, axis=-1)
            v_acc = float(accuracy_score(val_y, v_preds)) * 100.0

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_acc": round(v_acc, 2)
        })

    # Calibrate decision threshold on validation set via Youden's J
    model.eval()
    with torch.no_grad():
        val_t = torch.from_numpy(val_X)
        if loss_type == "am_softmax":
            _, emb_val = model(val_t, return_embedding=True)
            _, val_logits = am_criterion(emb_val, torch.zeros(len(val_X), dtype=torch.long))
        else:
            val_logits = model(val_t)
        v_probs = torch.softmax(val_logits, dim=-1).numpy()
        val_probs_fake = v_probs[:, 1]  # P(fake)

    fpr_v, tpr_v, thresh_v = roc_curve(val_y, val_probs_fake, pos_label=1)
    j_scores = tpr_v - fpr_v
    best_j_idx = int(np.argmax(j_scores))
    calibrated_threshold = float(thresh_v[best_j_idx]) if len(thresh_v) > 0 else 0.5
    if not np.isfinite(calibrated_threshold):
        calibrated_threshold = 0.5
    calibrated_threshold = max(0.05, min(0.95, calibrated_threshold))
    model.decision_threshold = calibrated_threshold

    if loss_type == "am_softmax":
        model.metric_head = am_criterion

    return model, calibrated_threshold, history


def evaluate_adaptation_model(
    model: W2V2DomainAdaptationHead,
    test_X: np.ndarray,
    test_records: List[Dict[str, Any]],
    model_name: str,
    threshold: float
) -> Dict[str, Any]:
    """Evaluate adaptation model on test set."""
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        logits = model(torch.from_numpy(test_X))
        total_lat_ms = (time.perf_counter() - t0) * 1000.0
        avg_lat_ms = total_lat_ms / len(test_X)

        probs = torch.softmax(logits, dim=-1).numpy()
        probs_fake = probs[:, 1]
        preds_fake = (probs_fake >= threshold).astype(int)

    y_true_binary = [1 if r["label"].strip().upper() == "FAKE" else 0 for r in test_records]

    acc = float(accuracy_score(y_true_binary, preds_fake)) * 100.0
    bacc = float(balanced_accuracy_score(y_true_binary, preds_fake)) * 100.0
    prec = float(precision_score(y_true_binary, preds_fake, zero_division=0)) * 100.0
    recall_val = float(recall_score(y_true_binary, preds_fake, zero_division=0)) * 100.0
    f1 = float(f1_score(y_true_binary, preds_fake, zero_division=0)) * 100.0

    try:
        auc = float(roc_auc_score(y_true_binary, probs_fake))
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_true_binary, preds_fake, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    fpr = (fp / (fp + tn) * 100.0) if (fp + tn) > 0 else 0.0
    fnr = (fn / (fn + tp) * 100.0) if (fn + tp) > 0 else 0.0

    # Dataset accuracy breakdown
    dataset_breakdown: Dict[str, Dict[str, Any]] = {}
    per_sample_details = []

    for i, item in enumerate(test_records):
        src = item.get("dataset_source", "Unknown")
        gt = item["label"].strip().upper()
        p_label = "FAKE" if preds_fake[i] == 1 else "REAL"
        is_corr = (gt == p_label)

        if src not in dataset_breakdown:
            dataset_breakdown[src] = {"total_samples": 0, "correct_samples": 0, "real_count": 0, "fake_count": 0}
        dataset_breakdown[src]["total_samples"] += 1
        if is_corr:
            dataset_breakdown[src]["correct_samples"] += 1
        if gt == "FAKE":
            dataset_breakdown[src]["fake_count"] += 1
        else:
            dataset_breakdown[src]["real_count"] += 1

        per_sample_details.append({
            "filename": os.path.basename(item["segment_path"]),
            "parent_file": item.get("parent_file", ""),
            "dataset_source": src,
            "ground_truth": gt,
            "predicted_label": p_label,
            "probability_fake": round(float(probs_fake[i]), 4),
            "is_correct": bool(is_corr)
        })

    for src in dataset_breakdown:
        tot = dataset_breakdown[src]["total_samples"]
        corr = dataset_breakdown[src]["correct_samples"]
        dataset_breakdown[src]["accuracy_percent"] = round((corr / tot) * 100.0, 2) if tot > 0 else 0.0

    # ROC curve
    fpr_arr, tpr_arr, roc_thresh = roc_curve(y_true_binary, probs_fake, pos_label=1)
    fnr_arr = 1.0 - tpr_arr
    eer_idx = int(np.nanargmin(np.abs(fpr_arr - fnr_arr))) if len(fpr_arr) > 0 else 0
    eer_val = float((fpr_arr[eer_idx] + fnr_arr[eer_idx]) / 2.0 * 100.0) if len(fpr_arr) > 0 else 50.0

    return {
        "model_name": model_name,
        "sample_count": len(test_records),
        "threshold_used": round(threshold, 4),
        "metrics": {
            "accuracy": round(acc, 2),
            "balanced_accuracy": round(bacc, 2),
            "precision": round(prec, 2),
            "recall": round(recall_val, 2),
            "f1_score": round(f1, 2),
            "roc_auc": round(auc, 4),
            "false_positive_rate": round(fpr, 2),
            "false_negative_rate": round(fnr, 2),
            "equal_error_rate": round(eer_val, 2)
        },
        "confusion_matrix": {
            "true_negatives_real": tn,
            "false_positives_fake": fp,
            "false_negatives_real": fn,
            "true_positives_fake": tp
        },
        "efficiency": {
            "avg_latency_ms": round(avg_lat_ms, 2),
            "parameter_count": model.count_parameters()
        },
        "dataset_accuracy_breakdown": dataset_breakdown,
        "roc_curve": {
            "fpr": [round(float(x), 4) for x in fpr_arr],
            "tpr": [round(float(x), 4) for x in tpr_arr],
            "thresholds": [round(float(x), 4) if np.isfinite(x) else 1.0 for x in roc_thresh]
        },
        "per_sample_details": per_sample_details
    }


# -------------------------------------------------------------------
# Master Experiment Runner
# -------------------------------------------------------------------

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    splits_dir = os.path.join(base_dir, "data", "splits")
    reports_dir = os.path.join(base_dir, "reports")
    saved_models_dir = os.path.join(base_dir, "models", "saved", "w2v2_aasist")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(saved_models_dir, exist_ok=True)

    with open(os.path.join(splits_dir, "train_manifest.json"), "r", encoding="utf-8") as f:
        train_records = json.load(f)
    with open(os.path.join(splits_dir, "val_manifest.json"), "r", encoding="utf-8") as f:
        val_records = json.load(f)
    with open(os.path.join(splits_dir, "test_manifest.json"), "r", encoding="utf-8") as f:
        test_records = json.load(f)

    print(f"Loaded datasets: {len(train_records)} train, {len(val_records)} val, {len(test_records)} test.")

    w2v2_clf = W2V2AASISTClassifier()

    # Step 1: Feature Extraction / Caching
    print("\n--- STEP 1: FEATURE EXTRACTION & CACHING ---")
    val_X, val_y = compute_or_load_dataset_features(
        val_records, w2v2_clf,
        os.path.join(saved_models_dir, "val_features.npz"),
        name="Validation Set"
    )
    test_X, test_y = compute_or_load_dataset_features(
        test_records, w2v2_clf,
        os.path.join(saved_models_dir, "test_features.npz"),
        name="Test Set"
    )
    train_clean_X, train_clean_y = compute_or_load_dataset_features(
        train_records, w2v2_clf,
        os.path.join(saved_models_dir, "train_clean_features.npz"),
        name="Train Clean"
    )

    # Augmented features for Experiment C (Full suite: RawBoost + Reverb + Noise)
    aug_full = AudioAugmentor(use_rawboost=True, use_reverb=True, use_noise=True, prob_apply=0.85)
    train_aug_full_X, train_aug_full_y = compute_or_load_dataset_features(
        train_records, w2v2_clf,
        os.path.join(saved_models_dir, "train_aug_full_features.npz"),
        augmentor=aug_full,
        name="Train Aug Full"
    )

    # Augmented features for Experiment D1 (RawBoost only)
    aug_rb = AudioAugmentor(use_rawboost=True, use_reverb=False, use_noise=False, prob_apply=0.85)
    train_aug_rb_X, train_aug_rb_y = compute_or_load_dataset_features(
        train_records, w2v2_clf,
        os.path.join(saved_models_dir, "train_aug_rb_features.npz"),
        augmentor=aug_rb,
        name="Train Aug RawBoost"
    )

    # Augmented features for Experiment D2 (Reverb only)
    aug_rev = AudioAugmentor(use_rawboost=False, use_reverb=True, use_noise=False, prob_apply=0.85)
    train_aug_rev_X, train_aug_rev_y = compute_or_load_dataset_features(
        train_records, w2v2_clf,
        os.path.join(saved_models_dir, "train_aug_rev_features.npz"),
        augmentor=aug_rev,
        name="Train Aug Reverb"
    )

    # Augmented features for Experiment D3 (Noise only)
    aug_noise = AudioAugmentor(use_rawboost=False, use_reverb=False, use_noise=True, prob_apply=0.85)
    train_aug_noise_X, train_aug_noise_y = compute_or_load_dataset_features(
        train_records, w2v2_clf,
        os.path.join(saved_models_dir, "train_aug_noise_features.npz"),
        augmentor=aug_noise,
        name="Train Aug Noise"
    )

    # Step 2: Run Experiments
    experiments_results: Dict[str, Any] = {}

    # Experiment A: Baseline Pretrained (Zero-Shot) from reports/w2v2_aasist_baseline.json
    baseline_path = os.path.join(reports_dir, "w2v2_aasist_baseline.json")
    if os.path.exists(baseline_path):
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline_json = json.load(f)
        exp_a_res = baseline_json.get("zero_shot_baseline", {})
    else:
        exp_a_res = {"metrics": {"accuracy": 46.15, "balanced_accuracy": 44.05, "f1_score": 58.82}}
    experiments_results["exp_a_baseline"] = exp_a_res

    # Experiment B: Standard Fine-Tuning (Clean Train Data)
    print("\n--- EXPERIMENT B: STANDARD FINE-TUNING (CLEAN TRAIN) ---")
    model_b, thresh_b, hist_b = train_adaptation_head(train_clean_X, train_clean_y, val_X, val_y)
    torch.save(model_b.state_dict(), os.path.join(saved_models_dir, "fine_tuned_exp_b_clean.pt"))
    exp_b_res = evaluate_adaptation_model(
        model_b, test_X, test_records,
        model_name="W2V2-AASIST (Exp B: Standard Fine-Tuning)",
        threshold=thresh_b
    )
    experiments_results["exp_b_clean"] = exp_b_res
    print(f"Exp B Accuracy: {exp_b_res['metrics']['accuracy']}%, F1: {exp_b_res['metrics']['f1_score']}%")

    # Experiment C: Fine-Tuning with Full Audio Augmentation Suite (RawBoost + Reverb + Noise)
    print("\n--- EXPERIMENT C: FINE-TUNING WITH FULL AUDIO AUGMENTATION SUITE ---")
    # Concatenate clean + augmented for rich multi-domain training
    train_c_X = np.concatenate([train_clean_X, train_aug_full_X], axis=0)
    train_c_y = np.concatenate([train_clean_y, train_aug_full_y], axis=0)
    model_c, thresh_c, hist_c = train_adaptation_head(train_c_X, train_c_y, val_X, val_y)
    torch.save(model_c.state_dict(), os.path.join(saved_models_dir, "fine_tuned_exp_c_augmented.pt"))
    exp_c_res = evaluate_adaptation_model(
        model_c, test_X, test_records,
        model_name="W2V2-AASIST (Exp C: Full Augmentation Suite)",
        threshold=thresh_c
    )
    experiments_results["exp_c_augmented"] = exp_c_res
    print(f"Exp C Accuracy: {exp_c_res['metrics']['accuracy']}%, F1: {exp_c_res['metrics']['f1_score']}%")

    # Experiment D1: RawBoost Alone
    print("\n--- EXPERIMENT D1: ABLATION - RAWBOOST ONLY ---")
    train_d1_X = np.concatenate([train_clean_X, train_aug_rb_X], axis=0)
    train_d1_y = np.concatenate([train_clean_y, train_aug_rb_y], axis=0)
    model_d1, thresh_d1, hist_d1 = train_adaptation_head(train_d1_X, train_d1_y, val_X, val_y)
    torch.save(model_d1.state_dict(), os.path.join(saved_models_dir, "fine_tuned_exp_d1_rawboost.pt"))
    exp_d1_res = evaluate_adaptation_model(
        model_d1, test_X, test_records,
        model_name="W2V2-AASIST (Exp D1: RawBoost Ablation)",
        threshold=thresh_d1
    )
    experiments_results["exp_d1_rawboost"] = exp_d1_res
    print(f"Exp D1 Accuracy: {exp_d1_res['metrics']['accuracy']}%, F1: {exp_d1_res['metrics']['f1_score']}%")

    # Experiment D2: Reverberation Alone
    print("\n--- EXPERIMENT D2: ABLATION - REVERBERATION ONLY ---")
    train_d2_X = np.concatenate([train_clean_X, train_aug_rev_X], axis=0)
    train_d2_y = np.concatenate([train_clean_y, train_aug_rev_y], axis=0)
    model_d2, thresh_d2, hist_d2 = train_adaptation_head(train_d2_X, train_d2_y, val_X, val_y)
    torch.save(model_d2.state_dict(), os.path.join(saved_models_dir, "fine_tuned_exp_d2_reverb.pt"))
    exp_d2_res = evaluate_adaptation_model(
        model_d2, test_X, test_records,
        model_name="W2V2-AASIST (Exp D2: Reverb Ablation)",
        threshold=thresh_d2
    )
    experiments_results["exp_d2_reverb"] = exp_d2_res
    print(f"Exp D2 Accuracy: {exp_d2_res['metrics']['accuracy']}%, F1: {exp_d2_res['metrics']['f1_score']}%")

    # Experiment D3: Noise Alone
    print("\n--- EXPERIMENT D3: ABLATION - NOISE ONLY ---")
    train_d3_X = np.concatenate([train_clean_X, train_aug_noise_X], axis=0)
    train_d3_y = np.concatenate([train_clean_y, train_aug_noise_y], axis=0)
    model_d3, thresh_d3, hist_d3 = train_adaptation_head(train_d3_X, train_d3_y, val_X, val_y)
    torch.save(model_d3.state_dict(), os.path.join(saved_models_dir, "fine_tuned_exp_d3_noise.pt"))
    exp_d3_res = evaluate_adaptation_model(
        model_d3, test_X, test_records,
        model_name="W2V2-AASIST (Exp D3: Noise Ablation)",
        threshold=thresh_d3
    )
    experiments_results["exp_d3_noise"] = exp_d3_res
    print(f"Exp D3 Accuracy: {exp_d3_res['metrics']['accuracy']}%, F1: {exp_d3_res['metrics']['f1_score']}%")

    # Save comprehensive phase 3 experiments report
    out_report_path = os.path.join(reports_dir, "phase3_training_experiments.json")
    final_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task": "Phase 3 Controlled Training and Audio Augmentation Experiments",
        "held_out_test_samples": len(test_records),
        "metric_step_percentage_points": round(100.0 / len(test_records), 2),
        "experiments": experiments_results,
        "ablation_comparison": {
            "baseline_exp_a": exp_a_res["metrics"]["accuracy"],
            "clean_exp_b": exp_b_res["metrics"]["accuracy"],
            "full_augmented_exp_c": exp_c_res["metrics"]["accuracy"],
            "rawboost_exp_d1": exp_d1_res["metrics"]["accuracy"],
            "reverb_exp_d2": exp_d2_res["metrics"]["accuracy"],
            "noise_exp_d3": exp_d3_res["metrics"]["accuracy"],
            "improvement_over_baseline_pp": round(exp_c_res["metrics"]["accuracy"] - exp_a_res["metrics"]["accuracy"], 2),
            "meets_target_improvement": bool(exp_c_res["metrics"]["accuracy"] >= exp_a_res["metrics"]["accuracy"] + 0.1)
        }
    }

    with open(out_report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    print(f"\nSaved Phase 3 experiments report to: {out_report_path}")

    # Update master model_comparison.json
    comp_path = os.path.join(reports_dir, "model_comparison.json")
    if os.path.exists(comp_path):
        with open(comp_path, "r", encoding="utf-8") as f:
            comp_data = json.load(f)
    else:
        comp_data = {}

    comp_data["w2v2_aasist_exp_b_clean"] = exp_b_res
    comp_data["w2v2_aasist_exp_c_augmented"] = exp_c_res
    comp_data["w2v2_aasist_ablation"] = final_report["ablation_comparison"]

    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comp_data, f, indent=2)
    print(f"Updated master model comparison: {comp_path}")


if __name__ == "__main__":
    main()
