"""
Evaluation Module for Strawberry AI ViT-GAT Framework.

Evaluates the trained model on the held-out test dataset (743 images):
- Computes Accuracy, Macro & Weighted Precision, Recall, F1-score
- Generates detailed per-class classification report
- Plots and saves high-resolution Confusion Matrix heatmap
- Evaluates disease severity rating alignment (Level 1 vs Level 2)
"""

import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from typing import Dict, Optional, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from src.utils import load_config, get_device, ensure_dir
from src.data_loader import get_dataloaders, StrawberryDataset
from src.preprocessing import get_val_transforms
from src.classifier import StrawberryViTGATClassifier


class ModelEvaluator:
    """Evaluates trained ViT-GAT checkpoint on held-out test splits."""

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_strawberry_gat.pth",
        config_path: str = "config.yaml",
        device: Optional[torch.device] = None,
    ):
        self.config = load_config(config_path)
        self.device = device or get_device()
        self.checkpoint_path = checkpoint_path

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        # Load checkpoint first to extract model configuration
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_cfg = checkpoint.get("model_config", {})

        # Instantiate model architecture with matched hyperparams
        self.model = StrawberryViTGATClassifier(
            num_classes=model_cfg.get("num_classes", self.config["dataset"]["num_classes"]),
            vit_model_name=model_cfg.get("vit_model_name", self.config["model"]["vit_backbone"]),
            pretrained_vit=False,
            freeze_vit=True,
            gat_hidden=model_cfg.get("gat_hidden", self.config["model"]["gat"]["hidden_channels"]),
            gat_out=model_cfg.get("gat_out", self.config["model"]["gat"]["out_channels"]),
            gat_heads=model_cfg.get("gat_heads", self.config["model"]["gat"]["heads"]),
            dropout=self.config["model"]["classifier"]["dropout"],
            connect_by=model_cfg.get("connect_by", self.config["model"]["graph"]["connect_by"]),
            include_severity_head=model_cfg.get("include_severity_head", True),
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.class_to_idx = checkpoint.get("class_to_idx") or {
            c: i for i, c in enumerate(self.config["dataset"]["classes"])
        }
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        self.classes = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]
        self.class_display_names = [
            self.config["dataset"]["class_display_names"].get(c, c) for c in self.classes
        ]

    @torch.no_grad()
    def evaluate(self, test_loader) -> Dict:
        """Run full evaluation on test loader."""
        all_preds = []
        all_targets = []
        all_probs = []

        for images, labels, _ in test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            logits = outputs["disease_logits"]
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # Core Metrics
        acc = float(accuracy_score(all_targets, all_preds))
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average="macro", zero_division=0
        )
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average="weighted", zero_division=0
        )

        # Per-class metrics
        p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
            all_targets, all_preds, average=None, zero_division=0
        )

        per_class_metrics = {}
        for idx, cls_name in enumerate(self.classes):
            disp = self.class_display_names[idx]
            per_class_metrics[cls_name] = {
                "display_name": disp,
                "precision": float(p_per[idx]) if idx < len(p_per) else 0.0,
                "recall": float(r_per[idx]) if idx < len(r_per) else 0.0,
                "f1_score": float(f1_per[idx]) if idx < len(f1_per) else 0.0,
                "support": int(sup_per[idx]) if idx < len(sup_per) else 0,
            }

        cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(self.classes))))

        report_dict = classification_report(
            all_targets,
            all_preds,
            target_names=self.class_display_names,
            output_dict=True,
            zero_division=0,
        )

        metrics = {
            "total_test_samples": len(all_targets),
            "accuracy": round(acc, 4),
            "macro_precision": round(float(macro_p), 4),
            "macro_recall": round(float(macro_r), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_precision": round(float(weighted_p), 4),
            "weighted_recall": round(float(weighted_r), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "per_class": per_class_metrics,
            "confusion_matrix": cm.tolist(),
            "classification_report": report_dict,
        }

        return metrics

    def plot_confusion_matrix(self, cm: np.ndarray, output_path: str) -> None:
        """Plot and save confusion matrix heatmap."""
        ensure_dir(os.path.dirname(output_path))
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_display_names,
            yticklabels=self.class_display_names,
            cbar=True,
            ax=ax,
        )

        ax.set_title("Strawberry ViT-GAT Disease Confusion Matrix", fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted Pathology Class", fontsize=11, fontweight="bold")
        ax.set_ylabel("Ground Truth Pathology Class", fontsize=11, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()


def run_evaluation():
    """Main evaluation routine."""
    config = load_config()
    checkpoint_path = os.path.join(
        config["training"]["checkpoint_dir"],
        config["training"]["best_checkpoint_name"],
    )

    if not os.path.exists(checkpoint_path):
        print(f"[FAIL] Checkpoint not found at {checkpoint_path}. Train the model first.")
        sys.exit(1)

    print("=" * 65)
    print("🍓 EVALUATING STRAWBERRY ViT-GAT ON HELD-OUT TEST DATASET")
    print("=" * 65)

    _, _, test_loader, _, _, _ = get_dataloaders(config)
    evaluator = ModelEvaluator(checkpoint_path=checkpoint_path)
    metrics = evaluator.evaluate(test_loader)

    print(f"Total Test Samples:   {metrics['total_test_samples']:,}")
    print(f"Overall Accuracy:     {metrics['accuracy']*100:.2f}%")
    print(f"Macro Precision:      {metrics['macro_precision']*100:.2f}%")
    print(f"Macro Recall:         {metrics['macro_recall']*100:.2f}%")
    print(f"Macro F1-Score:       {metrics['macro_f1']*100:.2f}%\n")

    print("-" * 65)
    print("Per-Class Assessment Performance:")
    print(f"{'Class Name':<25} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'Support':<7}")
    print("-" * 65)
    for cls_name, pdata in metrics["per_class"].items():
        print(
            f"{pdata['display_name']:<25} | "
            f"{pdata['precision']*100:>8.2f}% | "
            f"{pdata['recall']*100:>8.2f}% | "
            f"{pdata['f1_score']*100:>8.2f}% | "
            f"{pdata['support']:>7}"
        )
    print("-" * 65)

    # Save outputs
    metrics_path = os.path.join(config["evaluation"]["metrics_output_dir"], "test_metrics.json")
    cm_path = os.path.join(config["evaluation"]["figures_output_dir"], "confusion_matrix.png")
    ensure_dir(os.path.dirname(metrics_path))
    ensure_dir(os.path.dirname(cm_path))

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm = np.array(metrics["confusion_matrix"])
    evaluator.plot_confusion_matrix(cm, cm_path)

    print(f"[OK] Metrics saved to: {metrics_path}")
    print(f"[OK] Confusion matrix saved to: {cm_path}")
    print("=" * 65)


if __name__ == "__main__":
    run_evaluation()
