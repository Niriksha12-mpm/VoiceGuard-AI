"""
Plot Generation Script for VoiceGuard AI.
Generates publication-quality figures:
1. Training Curves (Loss and Accuracy over epochs)
2. Confusion Matrices (Model A Baseline vs Model B Baseline vs Model B Improved)
3. ROC Curves with AUC scores and operating points
Saves all figures to reports/plots/.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np


def generate_all_plots():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(base_dir, "reports")
    plots_dir = os.path.join(reports_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    history_path = os.path.join(reports_dir, "training_history.json")
    comparison_path = os.path.join(reports_dir, "model_comparison.json")

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Training Curves
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            hist = json.load(f)

        epochs = hist.get("epochs", list(range(1, len(hist.get("train_loss", [])) + 1)))
        tr_loss = hist.get("train_loss", [])
        val_loss = hist.get("val_loss", [])
        tr_acc = hist.get("train_accuracy", [])
        val_acc = hist.get("val_accuracy", [])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # Loss curve
        ax1.plot(epochs, tr_loss, "b-o", linewidth=2, label="Train Loss (BCE)")
        ax1.plot(epochs, val_loss, "r--s", linewidth=2, label="Validation Loss")
        best_ep = hist.get("best_epoch", 1)
        if best_ep <= len(val_loss):
            ax1.axvline(best_ep, color="green", linestyle=":", label=f"Best Checkpoint (Ep {best_ep})")
        ax1.set_title("VoiceGuard AI: Training & Validation Loss", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_ylabel("Binary Cross Entropy Loss", fontsize=11)
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        # Accuracy curve
        ax2.plot(epochs, tr_acc, "b-o", linewidth=2, label="Train Accuracy (%)")
        ax2.plot(epochs, val_acc, "r--s", linewidth=2, label="Validation Accuracy (%)")
        if best_ep <= len(val_acc):
            ax2.axvline(best_ep, color="green", linestyle=":", label=f"Best Checkpoint (Ep {best_ep})")
        ax2.set_title("VoiceGuard AI: Training & Validation Accuracy", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("Accuracy (%)", fontsize=11)
        ax2.legend(loc="lower right")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(plots_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[Plot] Training curves saved to {plot_path}")

    # 2. Confusion Matrices & ROC Curves
    if os.path.exists(comparison_path):
        with open(comparison_path, "r", encoding="utf-8") as f:
            comp = json.load(f)

        models = [
            ("Model A (Pretrained)", comp.get("model_a_pretrained", {})),
            ("Model B (Baseline 0.5)", comp.get("model_b_baseline", {})),
            ("Model B (Improved Calibrated)", comp.get("model_b_improved", {}))
        ]

        # Confusion Matrices
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for idx, (name, m_data) in enumerate(models):
            cm_dict = m_data.get("confusion_matrix", {"true_negatives_real": 0, "false_positives_fake": 0, "false_negatives_real": 0, "true_positives_fake": 0})
            cm_matrix = np.array([
                [cm_dict.get("true_negatives_real", 0), cm_dict.get("false_positives_fake", 0)],
                [cm_dict.get("false_negatives_real", 0), cm_dict.get("true_positives_fake", 0)]
            ])
            ax = axes[idx]
            im = ax.imshow(cm_matrix, interpolation="nearest", cmap=plt.cm.Blues)
            ax.set_title(f"{name}\nAcc: {m_data.get('metrics', {}).get('accuracy', 0)}%", fontsize=11, fontweight="bold")
            tick_marks = np.arange(2)
            ax.set_xticks(tick_marks)
            ax.set_xticklabels(["REAL", "FAKE"], fontsize=10)
            ax.set_yticks(tick_marks)
            ax.set_yticklabels(["REAL", "FAKE"], fontsize=10)
            ax.set_xlabel("Predicted Label", fontsize=10)
            ax.set_ylabel("Ground Truth", fontsize=10)

            thresh = cm_matrix.max() / 2.0
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, format(cm_matrix[i, j], "d"),
                            ha="center", va="center",
                            color="white" if cm_matrix[i, j] > thresh else "black",
                            fontsize=13, fontweight="bold")
            ax.grid(False)

        plt.tight_layout()
        cm_path = os.path.join(plots_dir, "confusion_matrices.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"[Plot] Confusion matrices saved to {cm_path}")

        # ROC Curves
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#7f7f7f", "#d62728", "#1f77b4"]
        for idx, (name, m_data) in enumerate(models):
            roc_data = m_data.get("roc_curve", {})
            fpr = roc_data.get("fpr", [0, 1])
            tpr = roc_data.get("tpr", [0, 1])
            auc = m_data.get("metrics", {}).get("roc_auc", 0.5)
            ax.plot(fpr, tpr, color=colors[idx], linewidth=2.2, label=f"{name} (AUC = {auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.500)")
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
        ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
        ax.set_title("VoiceGuard AI: Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        roc_path = os.path.join(plots_dir, "roc_curves.png")
        plt.savefig(roc_path, dpi=300)
        plt.close()
        print(f"[Plot] ROC curves saved to {roc_path}")


if __name__ == "__main__":
    generate_all_plots()
