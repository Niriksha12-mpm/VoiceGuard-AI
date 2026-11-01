"""
Live Accuracy Test Runner for VoiceGuard AI.
Executes non-fabricated evaluations across the active TakHemlata models
using the locked held-out test split (data/splits/test_manifest.json).
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

from backend.inference import InferenceEngine


def run_accuracy_evaluation():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manifest_path = os.path.join(base_dir, "data", "splits", "test_manifest.json")

    if not os.path.exists(manifest_path):
        print(f"Error: Manifest not found at {manifest_path}")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    print("=" * 80)
    print(f"VOICEGUARD AI - LIVE EMPIRICAL ACCURACY TEST")
    print(f"Evaluating {len(records)} locked held-out test audio samples...")
    print("=" * 80)

    # Initialize Engine
    eng = InferenceEngine()

    models_to_test = [
        ("TakHemlata Baseline (Pretrained Zero-Shot)", "w2v2_aasist_pretrained"),
        ("VoiceGuard Fine-Tuned (Reverb Adaptation - Exp D2)", "w2v2_aasist"),
        ("VoiceGuard Fine-Tuned (Large-Scale Cloud Adaptation)", "w2v2_large_scale"),
        ("VoiceGuard Forensic Ensemble (Consensus Voting)", "w2v2_ensemble")
    ]

    results_table = []

    for model_name, model_id in models_to_test:
        print(f"\n[Running Evaluation] -> {model_name}...")
        y_true = []
        y_pred = []
        latencies = []

        per_sample_details = []

        for rec in records:
            file_path = rec.get("segment_path")
            true_label = rec.get("label", "UNKNOWN").strip().upper()

            if not os.path.exists(file_path):
                # Try relative to base_dir
                rel_path = os.path.join(base_dir, "data", "processed", "real" if true_label == "REAL" else "fake", rec.get("segment_filename"))
                if os.path.exists(rel_path):
                    file_path = rel_path
                else:
                    continue

            with open(file_path, "rb") as af:
                raw_bytes = af.read()

            t0 = time.perf_counter()
            pred_res = eng.predict(raw_bytes, os.path.basename(file_path), model_type=model_id)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)

            pred_label = pred_res["prediction"]["label"]
            conf = pred_res["prediction"]["confidence_score"]
            prob_fake = pred_res["prediction"]["probability_fake"]

            y_true.append(1 if true_label == "FAKE" else 0)
            y_pred.append(1 if pred_label == "FAKE" else 0)

            is_correct = (true_label == pred_label)
            per_sample_details.append({
                "file": rec.get("segment_filename"),
                "source": rec.get("dataset_source"),
                "true": true_label,
                "pred": pred_label,
                "confidence": conf,
                "correct": is_correct
            })

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        acc = float(accuracy_score(y_true, y_pred)) * 100.0
        prec = float(precision_score(y_true, y_pred, zero_division=0)) * 100.0
        rec = float(recall_score(y_true, y_pred, zero_division=0)) * 100.0
        f1 = float(f1_score(y_true, y_pred, zero_division=0)) * 100.0
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        avg_lat = float(np.mean(latencies))

        results_table.append({
            "model_name": model_name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "avg_latency_ms": round(avg_lat, 1),
            "samples": per_sample_details
        })

    # Print Summary Table
    print("\n" + "=" * 80)
    print("EMPIRICAL ACCURACY BENCHMARK SUMMARY")
    print("=" * 80)
    header = f"{'Model Architecture':<42} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<8} | {'F1':<8} | {'TN/FP/FN/TP'}"
    print(header)
    print("-" * 80)

    for res in results_table:
        cm_str = f"{res['tn']}/{res['fp']}/{res['fn']}/{res['tp']}"
        row = f"{res['model_name']:<42} | {res['accuracy']:>6.2f}%  | {res['precision']:>6.2f}%  | {res['recall']:>5.1f}% | {res['f1_score']:>5.1f}% | {cm_str:<12}"
        print(row)

    print("=" * 80)

    # Save to JSON
    out_path = os.path.join(base_dir, "reports", "live_accuracy_test_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_table, f, indent=2)
    print(f"\nDetailed per-sample accuracy report saved to: {out_path}\n")


if __name__ == "__main__":
    run_accuracy_evaluation()
