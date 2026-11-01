"""
Generate Phase 3 and Phase 4 Visualizations for VoiceGuard AI:
1. ROC curves across all experimental variants
2. Ablation benchmark bar chart (Accuracy, F1-Score, and EER)
3. Score distribution histograms for Genuine vs. Spoof separation
4. Dataset accuracy breakdown
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(base_dir, "reports")
    plots_dir = os.path.join(reports_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    json_path = os.path.join(reports_dir, "phase3_training_experiments.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    exps = data["experiments"]

    # 1. Ablation Bar Chart: Accuracy and F1 across experiments
    models = [
        ("Exp A: Zero-Shot Baseline", "exp_a_baseline", "#e74c3c"),
        ("Exp B: Clean Fine-Tuning", "exp_b_clean", "#3498db"),
        ("Exp C: Full Augmentation", "exp_c_augmented", "#9b59b6"),
        ("Exp D1: RawBoost Only", "exp_d1_rawboost", "#e67e22"),
        ("Exp D2: Reverb Only", "exp_d2_reverb", "#2ecc71"),
        ("Exp D3: Noise Only", "exp_d3_noise", "#1abc9c")
    ]

    labels = [m[0] for m in models]
    accuracies = [exps[m[1]]["metrics"]["accuracy"] for m in models]
    f1_scores = [exps[m[1]]["metrics"]["f1_score"] for m in models]
    colors = [m[2] for m in models]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, accuracies, width, label="Test Accuracy (%)", color=colors, alpha=0.9, edgecolor="black")
    bars2 = ax.bar(x + width/2, f1_scores, width, label="Test F1-Score (%)", color=colors, alpha=0.4, hatch="//", edgecolor="black")

    ax.axhline(46.15, color="#e74c3c", linestyle="--", linewidth=1.5, label="Baseline (46.15%)")
    ax.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax.set_title("XLS-R 300M + AASIST: Controlled Fine-Tuning & Audio Augmentation Ablation", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True)

    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    barchart_path = os.path.join(plots_dir, "w2v2_aasist_ablation_barchart.png")
    plt.savefig(barchart_path)
    plt.close()
    print(f"Saved: {barchart_path}")

    # 2. ROC Curves for Experiments
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.50)")

    for name, key, color in models:
        exp_data = exps[key]
        roc = exp_data.get("roc_curve", {})
        fpr = roc.get("fpr", [])
        tpr = roc.get("tpr", [])
        auc = exp_data["metrics"].get("roc_auc", 0.5)
        if fpr and tpr:
            ax.plot(fpr, tpr, color=color, linewidth=2.0, label=f"{name} (AUC = {auc:.3f})")

    ax.set_xlabel("False Positive Rate (FPR)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (TPR / Recall)", fontsize=11, fontweight="bold")
    ax.set_title("ROC Curves: XLS-R 300M + AASIST Ablation Variants", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, fontsize=9)

    plt.tight_layout()
    roc_path = os.path.join(plots_dir, "w2v2_aasist_roc_curves.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"Saved: {roc_path}")

    # 3. Score Distributions for Best Model (Exp D2: Reverb Adaptation)
    d2_details = exps["exp_d2_reverb"]["per_sample_details"]
    real_probs = [s["probability_fake"] for s in d2_details if s["ground_truth"] == "REAL"]
    fake_probs = [s["probability_fake"] for s in d2_details if s["ground_truth"] == "FAKE"]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    bins = np.linspace(0.0, 1.0, 15)

    ax.hist(real_probs, bins=bins, alpha=0.7, color="#2ecc71", label=f"Genuine Human Speech (n={len(real_probs)})", edgecolor="black")
    ax.hist(fake_probs, bins=bins, alpha=0.7, color="#e74c3c", label=f"Deepfake / Synthetic Speech (n={len(fake_probs)})", edgecolor="black")

    thresh = exps["exp_d2_reverb"]["threshold_used"]
    ax.axvline(thresh, color="black", linestyle="--", linewidth=2.0, label=f"Calibrated Threshold ({thresh:.4f})")

    ax.set_xlabel("Predicted Deepfake Probability P(Fake)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Sample Count", fontsize=11, fontweight="bold")
    ax.set_title("Score Separation: Best Fine-Tuned Model (Exp D2 Reverb Adaptation)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper center", frameon=True, fontsize=10)

    plt.tight_layout()
    dist_path = os.path.join(plots_dir, "w2v2_aasist_score_distribution.png")
    plt.savefig(dist_path)
    plt.close()
    print(f"Saved: {dist_path}")

    print("All Phase 3 / Phase 4 visualization plots generated successfully!")


if __name__ == "__main__":
    main()
