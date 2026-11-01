"""
Rigorous Evaluation Engine for Pretrained XLS-R (Wav2Vec 2.0 300M) + AASIST.
Evaluates the reference pretrained checkpoint (w2v2-aasist.onnx) on:
1. Validation Split (data/splits/val_manifest.json) - threshold calibration
2. Locked Held-Out Test Split (data/splits/test_manifest.json) - final benchmark

Computes non-fabricated metrics:
- Accuracy, Balanced Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion Matrix (TN, FP, FN, TP)
- False Positive Rate (FPR), False Negative Rate (FNR), Equal Error Rate (EER)
- Latency (ms) and Throughput (samples/sec)
- 95% Bootstrap Confidence Intervals for Accuracy
- Dataset accuracy breakdown (Gary Stafford YouTube, Hume AI, WaveFake)
- Per-sample predictions and outputs
- Exports reports/w2v2_aasist_baseline.json and updates reports/model_comparison.json
"""

import os
import sys
import json
import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import soundfile as sf
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

from models.w2v2_aasist import W2V2AASISTClassifier, pad_fixed, preprocess_waveform


def compute_bootstrap_ci(
    y_true: List[int],
    y_pred: List[int],
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> Tuple[float, float]:
    """Calculate non-parametric bootstrap confidence interval for test accuracy."""
    np.random.seed(seed)
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0
    scores = []
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    for _ in range(n_bootstraps):
        indices = np.random.randint(0, n, size=n)
        scores.append(float(accuracy_score(y_t[indices], y_p[indices])))
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(scores, alpha * 100)) * 100.0
    upper = float(np.percentile(scores, (1.0 - alpha) * 100)) * 100.0
    return round(lower, 2), round(upper, 2)


def compute_eer(y_true: List[int], y_scores: List[float], pos_label: int = 1) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) and the corresponding threshold.
    Returns: (eer_percent, eer_threshold)
    """
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=pos_label)
        fnr = 1.0 - tpr
        eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
        eer_val = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0 * 100.0)
        return round(eer_val, 2), float(thresholds[eer_idx])
    except Exception:
        return 50.0, 0.0


def evaluate_manifest(
    classifier: W2V2AASISTClassifier,
    manifest_records: List[Dict[str, Any]],
    model_display_name: str,
    threshold: float,
    threshold_type: str = "score_logit1"
) -> Dict[str, Any]:
    """
    Evaluate W2V2-AASIST on manifest records.
    Binary definition matching project convention:
      Class 1 = FAKE (Synthetic/Deepfake)
      Class 0 = REAL (Bona fide/Genuine)
    """
    y_true_binary = []  # 1 for FAKE, 0 for REAL
    y_pred_binary = []
    y_probs_fake = []   # P(fake)
    y_scores = []       # bona fide score = logit[1]
    latencies = []

    per_sample_details = []
    dataset_breakdown: Dict[str, Dict[str, Any]] = {}

    for rec in manifest_records:
        file_path = rec["segment_path"]
        gt_label = rec["label"].strip().upper()
        is_fake = (gt_label == "FAKE")
        y_true_binary.append(1 if is_fake else 0)

        t0 = time.perf_counter()
        res = classifier.predict_proba(file_path)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)

        prob_fake = res["prob_spoof"]
        prob_real = res["prob_bonafide"]
        score = res["score"]  # bona fide logit

        # Decision rule:
        # If threshold_type == "prob_fake": classify FAKE if prob_fake >= threshold
        # If threshold_type == "score_logit1": classify REAL if score >= threshold, else FAKE
        if threshold_type == "prob_fake":
            pred_fake = (prob_fake >= threshold)
        else:
            # score is bona fide logit. REAL if score >= threshold, FAKE if score < threshold
            pred_fake = (score < threshold)

        pred_label = "FAKE" if pred_fake else "REAL"
        y_pred_binary.append(1 if pred_fake else 0)
        y_probs_fake.append(prob_fake)
        y_scores.append(score)

        is_correct = (pred_label == gt_label)
        dataset_src = rec.get("dataset_source", "Unknown")

        if dataset_src not in dataset_breakdown:
            dataset_breakdown[dataset_src] = {
                "total_samples": 0,
                "correct_samples": 0,
                "real_count": 0,
                "fake_count": 0
            }
        dataset_breakdown[dataset_src]["total_samples"] += 1
        if is_correct:
            dataset_breakdown[dataset_src]["correct_samples"] += 1
        if is_fake:
            dataset_breakdown[dataset_src]["fake_count"] += 1
        else:
            dataset_breakdown[dataset_src]["real_count"] += 1

        per_sample_details.append({
            "filename": os.path.basename(file_path),
            "parent_file": rec.get("parent_file", ""),
            "dataset_source": dataset_src,
            "ground_truth": gt_label,
            "predicted_label": pred_label,
            "score_bonafide": round(score, 4),
            "probability_fake": round(prob_fake, 4),
            "probability_real": round(prob_real, 4),
            "is_correct": bool(is_correct),
            "latency_ms": round(lat_ms, 2)
        })

    for src in dataset_breakdown:
        tot = dataset_breakdown[src]["total_samples"]
        corr = dataset_breakdown[src]["correct_samples"]
        dataset_breakdown[src]["accuracy_percent"] = round((corr / tot) * 100.0, 2) if tot > 0 else 0.0

    acc = float(accuracy_score(y_true_binary, y_pred_binary)) * 100.0
    bacc = float(balanced_accuracy_score(y_true_binary, y_pred_binary)) * 100.0
    prec = float(precision_score(y_true_binary, y_pred_binary, zero_division=0)) * 100.0
    rec = float(recall_score(y_true_binary, y_pred_binary, zero_division=0)) * 100.0
    f1 = float(f1_score(y_true_binary, y_pred_binary, zero_division=0)) * 100.0

    try:
        auc = float(roc_auc_score(y_true_binary, y_probs_fake))
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    # cm layout: [[TN, FP], [FN, TP]] where 0=REAL, 1=FAKE
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    fpr = (fp / (fp + tn) * 100.0) if (fp + tn) > 0 else 0.0
    fnr = (fn / (fn + tp) * 100.0) if (fn + tp) > 0 else 0.0
    eer_val, eer_thresh = compute_eer(y_true_binary, y_probs_fake, pos_label=1)
    ci_lower, ci_upper = compute_bootstrap_ci(y_true_binary, y_pred_binary)

    # Compute ROC points
    fpr_arr, tpr_arr, roc_thresh = roc_curve(y_true_binary, y_probs_fake, pos_label=1)

    real_prec = float(precision_score(y_true_binary, y_pred_binary, pos_label=0, zero_division=0)) * 100.0
    real_rec = float(recall_score(y_true_binary, y_pred_binary, pos_label=0, zero_division=0)) * 100.0
    real_f1 = float(f1_score(y_true_binary, y_pred_binary, pos_label=0, zero_division=0)) * 100.0

    avg_lat = float(np.mean(latencies))
    std_lat = float(np.std(latencies))
    throughput = round(1000.0 / avg_lat, 2) if avg_lat > 0 else 0.0

    return {
        "model_name": model_display_name,
        "sample_count": len(manifest_records),
        "threshold_used": round(threshold, 4),
        "threshold_type": threshold_type,
        "metrics": {
            "accuracy": round(acc, 2),
            "balanced_accuracy": round(bacc, 2),
            "accuracy_95ci": [ci_lower, ci_upper],
            "precision": round(prec, 2),
            "recall": round(rec, 2),
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
        "class_wise_metrics": {
            "real": {
                "precision": round(real_prec, 2),
                "recall": round(real_rec, 2),
                "f1_score": round(real_f1, 2),
                "support": tn + fp
            },
            "fake": {
                "precision": round(prec, 2),
                "recall": round(rec, 2),
                "f1_score": round(f1, 2),
                "support": fn + tp
            }
        },
        "efficiency": {
            "avg_latency_ms": round(avg_lat, 2),
            "std_latency_ms": round(std_lat, 2),
            "throughput_samples_per_sec": throughput,
            "parameter_count": 317837800
        },
        "dataset_accuracy_breakdown": dataset_breakdown,
        "roc_curve": {
            "fpr": [round(float(x), 4) for x in fpr_arr],
            "tpr": [round(float(x), 4) for x in tpr_arr],
            "thresholds": [round(float(x), 4) if np.isfinite(x) else 1.0 for x in roc_thresh]
        },
        "per_sample_details": per_sample_details
    }


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    splits_dir = os.path.join(base_dir, "data", "splits")
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    val_path = os.path.join(splits_dir, "val_manifest.json")
    test_path = os.path.join(splits_dir, "test_manifest.json")

    with open(val_path, "r", encoding="utf-8") as f:
        val_records = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    print(f"Loaded {len(val_records)} validation samples and {len(test_records)} test samples.")

    classifier = W2V2AASISTClassifier()

    # Step 1: Validation Set Evaluation & Threshold Calibration
    print("\n--- EVALUATING VALIDATION SPLIT & CALIBRATING THRESHOLD ---")
    val_y_true = [1 if r["label"].strip().upper() == "FAKE" else 0 for r in val_records]
    val_scores = []
    val_probs_fake = []

    for r in val_records:
        res = classifier.predict_proba(r["segment_path"])
        val_scores.append(res["score"])
        val_probs_fake.append(res["prob_spoof"])

    val_scores_np = np.array(val_scores)
    val_probs_fake_np = np.array(val_probs_fake)
    val_y_true_np = np.array(val_y_true)

    # Youden J index calibration on validation P(fake)
    fpr_v, tpr_v, thresh_v = roc_curve(val_y_true_np, val_probs_fake_np, pos_label=1)
    j_scores = tpr_v - fpr_v
    best_j_idx = int(np.argmax(j_scores))
    calibrated_prob_thresh = float(thresh_v[best_j_idx])

    # Also calibrate threshold on raw bona fide score (logit[1])
    fpr_s, tpr_s, thresh_s = roc_curve(val_y_true_np, -val_scores_np, pos_label=1)
    j_scores_s = tpr_s - fpr_s
    best_s_idx = int(np.argmax(j_scores_s))
    calibrated_score_thresh = -float(thresh_s[best_s_idx])

    val_eer, _ = compute_eer(val_y_true, val_probs_fake, pos_label=1)
    print(f"Validation EER: {val_eer:.2f}%")
    print(f"Calibrated P(fake) Threshold (Youden's J): {calibrated_prob_thresh:.4f}")
    print(f"Calibrated Score Threshold (Youden's J): {calibrated_score_thresh:.4f}")

    # Step 2: Test Set Evaluation - Default Zero-Shot (score=0.0)
    print("\n--- EVALUATING PRETRAINED BASELINE (DEFAULT THRESHOLD score=0.0) ---")
    baseline_default = evaluate_manifest(
        classifier=classifier,
        manifest_records=test_records,
        model_display_name="XLS-R 300M + AASIST (Zero-Shot Baseline, thr=0.0)",
        threshold=0.0,
        threshold_type="score_logit1"
    )
    print(f"Zero-Shot Baseline Test Accuracy: {baseline_default['metrics']['accuracy']}%")
    print(f"Balanced Accuracy: {baseline_default['metrics']['balanced_accuracy']}%")
    print(f"Precision: {baseline_default['metrics']['precision']}%, Recall: {baseline_default['metrics']['recall']}%")
    print(f"F1 Score: {baseline_default['metrics']['f1_score']}%, ROC-AUC: {baseline_default['metrics']['roc_auc']}")

    # Step 3: Test Set Evaluation - Calibrated Validation Threshold
    print(f"\n--- EVALUATING PRETRAINED BASELINE (CALIBRATED THRESHOLD thr={calibrated_prob_thresh:.4f}) ---")
    baseline_calibrated = evaluate_manifest(
        classifier=classifier,
        manifest_records=test_records,
        model_display_name=f"XLS-R 300M + AASIST (Calibrated Baseline, thr={calibrated_prob_thresh:.4f})",
        threshold=calibrated_prob_thresh,
        threshold_type="prob_fake"
    )
    print(f"Calibrated Baseline Test Accuracy: {baseline_calibrated['metrics']['accuracy']}%")
    print(f"Balanced Accuracy: {baseline_calibrated['metrics']['balanced_accuracy']}%")
    print(f"Precision: {baseline_calibrated['metrics']['precision']}%, Recall: {baseline_calibrated['metrics']['recall']}%")
    print(f"F1 Score: {baseline_calibrated['metrics']['f1_score']}%, ROC-AUC: {baseline_calibrated['metrics']['roc_auc']}")

    # Step 4: Save Baseline Report
    baseline_report_path = os.path.join(reports_dir, "w2v2_aasist_baseline.json")
    baseline_full_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_architecture": "XLS-R 300M + AASIST (SpeechAntiSpoofingBenchmarks/W2V2-AASIST)",
        "reference_checkpoint": "SpeechAntiSpoofingBenchmarks/W2V2-AASIST",
        "parameters": 317837800,
        "input_sample_rate": 16000,
        "window_samples": 64600,
        "validation_samples": len(val_records),
        "test_samples": len(test_records),
        "validation_calibrated_threshold_prob_fake": round(calibrated_prob_thresh, 4),
        "validation_calibrated_threshold_score": round(calibrated_score_thresh, 4),
        "zero_shot_baseline": baseline_default,
        "calibrated_baseline": baseline_calibrated
    }
    with open(baseline_report_path, "w", encoding="utf-8") as f:
        json.dump(baseline_full_report, f, indent=2)
    print(f"\nSaved W2V2-AASIST baseline report to: {baseline_report_path}")

    # Step 5: Update master model_comparison.json
    comp_path = os.path.join(reports_dir, "model_comparison.json")
    if os.path.exists(comp_path):
        with open(comp_path, "r", encoding="utf-8") as f:
            comp_data = json.load(f)
    else:
        comp_data = {}

    comp_data["w2v2_aasist_pretrained"] = baseline_default
    comp_data["w2v2_aasist_calibrated"] = baseline_calibrated

    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comp_data, f, indent=2)
    print(f"Updated master model comparison report: {comp_path}")


if __name__ == "__main__":
    main()
