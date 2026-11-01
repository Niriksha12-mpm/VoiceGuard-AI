"""
Rigorous Evaluation and Model Comparison Engine for VoiceGuard AI.
Evaluates:
1. Model A (Pretrained CNN Baseline)
2. Model B Baseline (Colab Lakhs weights with default 0.5 threshold)
3. Model B Improved (Fine-Tuned Four-Block CNN-BiLSTM with calibrated validation threshold)

Evaluated on the exact same locked held-out test split (data/splits/test_manifest.json).
Computes genuine, non-fabricated metrics:
- Accuracy, Balanced Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion Matrix (TN, FP, FN, TP)
- False Positive Rate (FPR), False Negative Rate (FNR), Equal Error Rate (EER)
- Latency (ms) and Throughput (samples/sec)
- 95% Bootstrap Confidence Intervals for Accuracy
- Metric resolution calculation (e.g. 1/13 = 7.69 pp step)
- Exports JSON and CSV comparison reports to reports/
"""

import os
import json
import csv
import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import soundfile as sf
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

from models.architecture import DeepfakeCNNBiLSTMClassifier, load_classifier
from models.feature_extractor import AudioFeatureExtractor


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


def compute_eer(y_true: List[int], y_probs: List[float]) -> float:
    """Compute Equal Error Rate (point where FPR == FNR)."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        fnr = 1.0 - tpr
        eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
        return round(float((fpr[eer_idx] + fnr[eer_idx]) / 2.0 * 100.0), 2)
    except Exception:
        return 50.0


def evaluate_single_model(
    model: torch.nn.Module,
    manifest_records: list,
    model_name: str,
    extractor: AudioFeatureExtractor,
    device: torch.device,
    custom_threshold: Optional[float] = None
) -> dict:
    """Evaluate a model on a set of manifest records and compute full metrics."""
    y_true = []
    y_pred = []
    y_probs = []
    latencies = []

    per_sample_results = []
    dataset_breakdown: dict = {}

    threshold = custom_threshold if custom_threshold is not None else getattr(model, "decision_threshold", 0.5)
    model.eval()

    with torch.no_grad():
        for rec in manifest_records:
            file_path = rec["segment_path"]
            ground_truth = rec["label"].strip().upper()
            y_true_binary = 1 if ground_truth == "FAKE" else 0

            t0 = time.perf_counter()
            y, sr = sf.read(file_path, dtype="float32")
            tensor = extractor.extract_tensor(y).to(device)
            logit = model(tensor).item()
            prob_fake = 1.0 / (1.0 + np.exp(-logit))
            latency_ms = (time.perf_counter() - t0) * 1000.0

            predicted_label = "FAKE" if prob_fake >= threshold else "REAL"
            y_pred_binary = 1 if predicted_label == "FAKE" else 0

            y_true.append(y_true_binary)
            y_pred.append(y_pred_binary)
            y_probs.append(prob_fake)
            latencies.append(latency_ms)

            is_correct = (predicted_label == ground_truth)
            ds_src = rec.get("dataset_source", "Unknown")

            dataset_breakdown.setdefault(ds_src, {"total": 0, "correct": 0, "real": 0, "fake": 0})
            dataset_breakdown[ds_src]["total"] += 1
            if is_correct:
                dataset_breakdown[ds_src]["correct"] += 1
            if ground_truth == "REAL":
                dataset_breakdown[ds_src]["real"] += 1
            else:
                dataset_breakdown[ds_src]["fake"] += 1

            per_sample_results.append({
                "filename": rec.get("segment_filename", os.path.basename(file_path)),
                "parent_file": rec.get("parent_file", ""),
                "dataset_source": ds_src,
                "ground_truth": ground_truth,
                "predicted_label": predicted_label,
                "probability_fake": round(float(prob_fake), 4),
                "is_correct": is_correct,
                "latency_ms": round(float(latency_ms), 2)
            })

    acc = accuracy_score(y_true, y_pred) * 100.0
    bal_acc = balanced_accuracy_score(y_true, y_pred) * 100.0
    prec = precision_score(y_true, y_pred, zero_division=0) * 100.0
    rec = recall_score(y_true, y_pred, zero_division=0) * 100.0
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100.0
    ci_lower, ci_upper = compute_bootstrap_ci(y_true, y_pred)
    eer = compute_eer(y_true, y_probs)

    try:
        if len(set(y_true)) > 1:
            roc_auc = float(roc_auc_score(y_true, y_probs))
            fpr_vals, tpr_vals, thresholds = roc_curve(y_true, y_probs)
            safe_thresh = [1.0 if np.isinf(x) else round(float(x), 4) for x in thresholds]
            roc_curve_data = {
                "fpr": [round(float(x), 4) for x in fpr_vals],
                "tpr": [round(float(x), 4) for x in tpr_vals],
                "thresholds": safe_thresh
            }
        else:
            roc_auc = 0.5
            roc_curve_data = {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "thresholds": [1.0, 0.0]}
    except Exception:
        roc_auc = 0.5
        roc_curve_data = {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "thresholds": [1.0, 0.0]}

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    fpr = float(fp / (fp + tn)) * 100.0 if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) * 100.0 if (fn + tp) > 0 else 0.0

    real_precision = float(tn / (tn + fn)) * 100.0 if (tn + fn) > 0 else 0.0
    real_recall = float(tn / (tn + fp)) * 100.0 if (tn + fp) > 0 else 0.0
    real_f1 = (2 * real_precision * real_recall) / (real_precision + real_recall) if (real_precision + real_recall) > 0 else 0.0

    ds_summary = {}
    for ds_name, stats in dataset_breakdown.items():
        ds_summary[ds_name] = {
            "total_samples": stats["total"],
            "correct_samples": stats["correct"],
            "accuracy_percent": round(stats["correct"] / stats["total"] * 100.0, 2),
            "real_count": stats["real"],
            "fake_count": stats["fake"]
        }

    avg_latency = float(np.mean(latencies))
    std_latency = float(np.std(latencies))
    throughput = round(1000.0 / avg_latency, 1) if avg_latency > 0 else 0.0

    return {
        "model_name": model_name,
        "sample_count": len(manifest_records),
        "threshold_used": round(threshold, 4),
        "metrics": {
            "accuracy": round(acc, 2),
            "balanced_accuracy": round(bal_acc, 2),
            "accuracy_95ci": [ci_lower, ci_upper],
            "precision": round(prec, 2),
            "recall": round(rec, 2),
            "f1_score": round(f1, 2),
            "roc_auc": round(roc_auc, 4),
            "false_positive_rate": round(fpr, 2),
            "false_negative_rate": round(fnr, 2),
            "equal_error_rate": round(eer, 2)
        },
        "confusion_matrix": {
            "true_negatives_real": tn,
            "false_positives_fake": fp,
            "false_negatives_real": fn,
            "true_positives_fake": tp
        },
        "class_wise_metrics": {
            "real": {"precision": round(real_precision, 2), "recall": round(real_recall, 2), "f1_score": round(real_f1, 2), "support": int(tn + fp)},
            "fake": {"precision": round(prec, 2), "recall": round(rec, 2), "f1_score": round(f1, 2), "support": int(tp + fn)}
        },
        "efficiency": {
            "avg_latency_ms": round(avg_latency, 2),
            "std_latency_ms": round(std_latency, 2),
            "throughput_samples_per_sec": throughput,
            "parameter_count": model.count_parameters() if hasattr(model, "count_parameters") else sum(p.numel() for p in model.parameters())
        },
        "dataset_accuracy_breakdown": ds_summary,
        "roc_curve": roc_curve_data,
        "per_sample_details": per_sample_results
    }


def run_comparative_evaluation(
    test_manifest_path: Optional[str] = None,
    reports_dir: Optional[str] = None
) -> dict:
    """Run full evaluation comparing Model A, Model B Baseline, and Model B Improved."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if test_manifest_path is None:
        test_manifest_path = os.path.join(base_dir, "data", "splits", "test_manifest.json")
    if reports_dir is None:
        reports_dir = os.path.join(base_dir, "reports")

    os.makedirs(reports_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = AudioFeatureExtractor()

    with open(test_manifest_path, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    n_test = len(test_records)
    metric_resolution = round(100.0 / n_test, 2) if n_test > 0 else 0.0
    print("=" * 65)
    print("  VOICEGUARD AI: COMPARATIVE BENCHMARK EVALUATION")
    print("=" * 65)
    print(f"Held-Out Test Set: {test_manifest_path}")
    print(f"Total Test Samples: {n_test} (Metric Resolution: {metric_resolution} pp per sample)")
    print(f"Device:             {device}")
    print("=" * 65)

    # 1. Model A (Pretrained Baseline)
    path_a = os.path.join(base_dir, "models", "saved", "pretrained_model_a.pt")
    model_a = load_classifier(path_a, map_location=str(device))
    eval_a = evaluate_single_model(model_a, test_records, "Model A (Pretrained Baseline)", extractor, device, custom_threshold=0.5)

    # 2. Model B Baseline (Colab Lakhs weights at 0.5 threshold)
    path_b_lakhs = os.path.join(base_dir, "models", "saved", "fine_tuned_model_b_lakhs.pt")
    model_b_base = load_classifier(path_b_lakhs, map_location=str(device))
    eval_b_base = evaluate_single_model(model_b_base, test_records, "Model B (Colab Baseline)", extractor, device, custom_threshold=0.5)

    # 3. Model B Improved (Fine-Tuned with calibrated validation threshold)
    path_b_improved = os.path.join(base_dir, "models", "saved", "fine_tuned_model_b.pt")
    model_b_improved = load_classifier(path_b_improved, map_location=str(device))
    eval_b_imp = evaluate_single_model(model_b_improved, test_records, "Model B (Improved Fine-Tuned)", extractor, device)

    # Calculate exact delta
    acc_baseline = eval_b_base["metrics"]["accuracy"]
    acc_improved = eval_b_imp["metrics"]["accuracy"]
    acc_delta_pp = round(acc_improved - acc_baseline, 2)
    rel_change_pct = round((acc_improved - acc_baseline) / max(acc_baseline, 1e-5) * 100.0, 2)

    # Count changed predictions between baseline and improved
    preds_baseline = [s["predicted_label"] for s in eval_b_base["per_sample_details"]]
    preds_improved = [s["predicted_label"] for s in eval_b_imp["per_sample_details"]]
    changed_preds = sum(1 for p_b, p_i in zip(preds_baseline, preds_improved) if p_b != p_i)

    meets_target = acc_delta_pp >= 0.1

    print("\n" + "=" * 65)
    print("  EVALUATION SUMMARY & ACCURACY DELTA")
    print("=" * 65)
    print(f"Model A (Pretrained Baseline):       {eval_a['metrics']['accuracy']}% (ROC-AUC: {eval_a['metrics']['roc_auc']})")
    print(f"Model B Baseline (0.5 threshold):    {acc_baseline}% (ROC-AUC: {eval_b_base['metrics']['roc_auc']})")
    print(f"Model B Improved (Calibrated Thresh):{acc_improved}% (ROC-AUC: {eval_b_imp['metrics']['roc_auc']}, Thresh: {eval_b_imp['threshold_used']})")
    print("-" * 65)
    print(f"Accuracy Delta:                      +{acc_delta_pp} percentage points")
    print(f"Relative Accuracy Gain:              +{rel_change_pct}%")
    print(f"Changed Predictions Count:           {changed_preds} / {n_test}")
    print(f"Metric Resolution:                   {metric_resolution} pp per sample")
    print(f"Target Met (>= +0.1 pp):             {'[PASS] YES' if meets_target else '[FAIL] NO'}")
    print("=" * 65)

    comparison_bundle = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "held_out_test_samples": n_test,
        "metric_resolution_percentage_points": metric_resolution,
        "held_out_distribution": {
            "real": sum(1 for r in test_records if r["label"] == "REAL"),
            "fake": sum(1 for r in test_records if r["label"] == "FAKE")
        },
        "model_a_pretrained": eval_a,
        "model_b_baseline": eval_b_base,
        "model_b_improved": eval_b_imp,
        "improvement_delta": {
            "baseline_model": "Model B (Colab Baseline)",
            "improved_model": "Model B (Improved Fine-Tuned)",
            "accuracy_baseline": acc_baseline,
            "accuracy_improved": acc_improved,
            "accuracy_delta_percentage_points": acc_delta_pp,
            "accuracy_relative_change_percent": rel_change_pct,
            "changed_predictions_count": changed_preds,
            "precision_delta": round(eval_b_imp["metrics"]["precision"] - eval_b_base["metrics"]["precision"], 2),
            "recall_delta": round(eval_b_imp["metrics"]["recall"] - eval_b_base["metrics"]["recall"], 2),
            "f1_score_delta": round(eval_b_imp["metrics"]["f1_score"] - eval_b_base["metrics"]["f1_score"], 2),
            "fpr_reduction": round(eval_b_base["metrics"]["false_positive_rate"] - eval_b_imp["metrics"]["false_positive_rate"], 2),
            "fnr_reduction": round(eval_b_base["metrics"]["false_negative_rate"] - eval_b_imp["metrics"]["false_negative_rate"], 2),
            "meets_target_improvement": meets_target,
            "resolution_note": f"With {n_test} test samples, minimum detectable accuracy step is {metric_resolution} pp."
        }
    }

    # Save JSON reports
    comp_json_path = os.path.join(reports_dir, "model_comparison.json")
    with open(comp_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_bundle, f, indent=2)

    metrics_json_path = os.path.join(reports_dir, "evaluation_metrics.json")
    metrics_summary = {
        "test_evaluation": {
            "model_a_pretrained": eval_a["metrics"],
            "model_b_baseline": eval_b_base["metrics"],
            "model_b_improved": eval_b_imp["metrics"],
            "delta": comparison_bundle["improvement_delta"]
        },
        "confusion_matrices": {
            "model_a_pretrained": eval_a["confusion_matrix"],
            "model_b_baseline": eval_b_base["confusion_matrix"],
            "model_b_improved": eval_b_imp["confusion_matrix"]
        },
        "efficiency": {
            "model_a_pretrained": eval_a["efficiency"],
            "model_b_baseline": eval_b_base["efficiency"],
            "model_b_improved": eval_b_imp["efficiency"]
        }
    }
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    # Save CSV reports
    csv_path = os.path.join(reports_dir, "model_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Accuracy (%)", "Balanced Acc (%)", "95% CI Lower", "95% CI Upper",
                         "Precision (%)", "Recall (%)", "F1 (%)", "ROC-AUC", "FPR (%)", "FNR (%)", "Latency (ms)", "Params"])
        for m_key in ["model_a_pretrained", "model_b_baseline", "model_b_improved"]:
            item = comparison_bundle[m_key]
            m = item["metrics"]
            eff = item["efficiency"]
            writer.writerow([
                item["model_name"], m["accuracy"], m["balanced_accuracy"], m["accuracy_95ci"][0], m["accuracy_95ci"][1],
                m["precision"], m["recall"], m["f1_score"], m["roc_auc"], m["false_positive_rate"], m["false_negative_rate"],
                eff["avg_latency_ms"], eff["parameter_count"]
            ])

    metrics_csv_path = os.path.join(reports_dir, "evaluation_metrics.csv")
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Model A (Pretrained)", "Model B (Baseline)", "Model B (Improved)", "Delta (Imp - Base)"])
        writer.writerow(["Accuracy (%)", eval_a["metrics"]["accuracy"], acc_baseline, acc_improved, f"+{acc_delta_pp} pp"])
        writer.writerow(["Precision (%)", eval_a["metrics"]["precision"], eval_b_base["metrics"]["precision"], eval_b_imp["metrics"]["precision"], f"{round(eval_b_imp['metrics']['precision'] - eval_b_base['metrics']['precision'], 2)} pp"])
        writer.writerow(["Recall (%)", eval_a["metrics"]["recall"], eval_b_base["metrics"]["recall"], eval_b_imp["metrics"]["recall"], f"{round(eval_b_imp['metrics']['recall'] - eval_b_base['metrics']['recall'], 2)} pp"])
        writer.writerow(["F1-Score (%)", eval_a["metrics"]["f1_score"], eval_b_base["metrics"]["f1_score"], eval_b_imp["metrics"]["f1_score"], f"{round(eval_b_imp['metrics']['f1_score'] - eval_b_base['metrics']['f1_score'], 2)} pp"])
        writer.writerow(["ROC-AUC", eval_a["metrics"]["roc_auc"], eval_b_base["metrics"]["roc_auc"], eval_b_imp["metrics"]["roc_auc"], f"{round(eval_b_imp['metrics']['roc_auc'] - eval_b_base['metrics']['roc_auc'], 4)}"])
        writer.writerow(["FPR (%)", eval_a["metrics"]["false_positive_rate"], eval_b_base["metrics"]["false_positive_rate"], eval_b_imp["metrics"]["false_positive_rate"], f"{round(eval_b_imp['metrics']['false_positive_rate'] - eval_b_base['metrics']['false_positive_rate'], 2)} pp"])

    print(f"\n[Saved] Reports written to:\n  - {comp_json_path}\n  - {metrics_json_path}\n  - {csv_path}\n  - {metrics_csv_path}")
    return comparison_bundle


if __name__ == "__main__":
    run_comparative_evaluation()
