"""
Training Pipeline for Strawberry AI ViT-GAT Assessment Framework.

Features:
- Weighted CrossEntropyLoss to handle class imbalance across strawberry disease classes.
- AdamW optimizer with Cosine Annealing learning rate schedule.
- Tracking of loss, accuracy, and macro-averaged F1 score per epoch.
- Best model checkpointing based on validation macro F1.
- Early stopping to prevent overfitting.
- Export of training curves to PNG and metrics to JSON.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

from src.utils import set_seed, get_device, ensure_dir, load_config
from src.data_loader import get_dataloaders
from src.classifier import StrawberryViTGATClassifier


class StrawberryTrainer:
    """Manages model training, validation, checkpointing, and evaluation tracking."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        class_weights: Optional[torch.Tensor] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "checkpoints",
        best_model_name: str = "best_strawberry_gat.pth",
    ):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = ensure_dir(checkpoint_dir)
        self.best_model_path = os.path.join(self.checkpoint_dir, best_model_name)

        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimize only parameters requiring gradients (GAT + Classifier head)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1_macro": [],
            "lr": [],
        }

    def train_epoch(self) -> float:
        """Execute one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for images, labels, _ in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs["disease_logits"], labels)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

        return total_loss / max(total_samples, 1)

    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float]:
        """Evaluate on validation set. Returns (val_loss, accuracy, macro_f1)."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_targets = []

        for images, labels, _ in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs["disease_logits"], labels)

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            preds = torch.argmax(outputs["disease_logits"], dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

        avg_loss = total_loss / max(total_samples, 1)
        acc = accuracy_score(all_targets, all_preds)
        macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

        return avg_loss, float(acc), float(macro_f1)

    def fit(
        self,
        epochs: int = 10,
        patience: int = 3,
        class_to_idx: Optional[dict] = None,
    ) -> dict:
        """Run complete training loop with validation, learning rate scheduling, and early stopping."""
        scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-5)
        best_val_f1 = -1.0
        patience_counter = 0

        print(f"Starting training on device: {self.device} for {epochs} epochs...")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            ep_start = time.time()
            train_loss = self.train_epoch()
            val_loss, val_acc, val_f1 = self.validate()
            scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            ep_time = time.time() - ep_start

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(round(train_loss, 4))
            self.history["val_loss"].append(round(val_loss, 4))
            self.history["val_acc"].append(round(val_acc, 4))
            self.history["val_f1_macro"].append(round(val_f1, 4))
            self.history["lr"].append(current_lr)

            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] ({ep_time:.1f}s) - "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc*100:.2f}% | "
                f"Val Macro-F1: {val_f1:.4f}"
            )

            # Checkpoint saving
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                self.save_checkpoint(epoch, val_f1, val_acc, class_to_idx)
                print(f"  [*] Best model saved! (Val Macro-F1: {val_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  [!] Early stopping triggered after {epoch} epochs.")
                    break

        total_duration = time.time() - start_time
        print(f"Training completed in {total_duration/60:.2f} minutes.")
        return self.history

    def save_checkpoint(
        self,
        epoch: int,
        val_f1: float,
        val_acc: float,
        class_to_idx: Optional[dict] = None,
    ) -> str:
        """Save full model checkpoint including weights, optimizer state, and class mappings."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_f1": val_f1,
            "val_acc": val_acc,
            "class_to_idx": class_to_idx,
            "num_classes": self.model.num_classes,
            "model_config": {
                "num_classes": self.model.num_classes,
                "gat_hidden": getattr(self.model, "gat_hidden", 128),
                "gat_out": getattr(self.model, "gat_out", 64),
                "gat_heads": getattr(self.model, "gat_heads", 4),
                "include_severity_head": getattr(self.model, "include_severity_head", True),
                "vit_model_name": getattr(self.model.encoder, "model_name", "vit_tiny_patch16_224"),
                "connect_by": getattr(self.model, "connect_by", "hybrid"),
            },
        }
        torch.save(checkpoint, self.best_model_path)
        return self.best_model_path

    def plot_and_save_history(self, output_png: str, output_json: str) -> None:
        """Generate and save loss and accuracy/F1 training curves."""
        ensure_dir(os.path.dirname(output_png))
        ensure_dir(os.path.dirname(output_json))

        # Save metrics JSON
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        # Plot training curves
        epochs = self.history["epoch"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Loss Curve
        axes[0].plot(epochs, self.history["train_loss"], "o-", label="Train Loss", color="#d9534f")
        axes[0].plot(epochs, self.history["val_loss"], "s--", label="Val Loss", color="#0275d8")
        axes[0].set_title("Cross-Entropy Loss vs. Epochs", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].legend()

        # Metric Curves
        axes[1].plot(epochs, [a * 100 for a in self.history["val_acc"]], "o-", label="Val Accuracy (%)", color="#5cb85c")
        axes[1].plot(epochs, [f * 100 for f in self.history["val_f1_macro"]], "^-", label="Val Macro-F1 (%)", color="#f0ad4e")
        axes[1].set_title("Validation Performance vs. Epochs", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Percentage (%)")
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(output_png, dpi=200)
        plt.close()


def run_training():
    """Main training routine."""
    config = load_config()
    set_seed(config["project"]["seed"])

    print("=" * 65)
    print("🍓 TRAINING STRAWBERRY ViT-GAT CLASSIFIER")
    print("=" * 65)

    train_loader, val_loader, test_loader, class_weights, class_to_idx, _ = get_dataloaders(config)

    print(f"Dataset splits loaded:")
    print(f"  - Train samples: {len(train_loader.dataset)}")
    print(f"  - Val samples:   {len(val_loader.dataset)}")
    print(f"  - Test samples:  {len(test_loader.dataset)}")

    model = StrawberryViTGATClassifier(
        num_classes=config["dataset"]["num_classes"],
        vit_model_name=config["model"]["vit_backbone"],
        pretrained_vit=config["model"]["pretrained"],
        freeze_vit=config["model"]["freeze_vit"],
        gat_hidden=config["model"]["gat"]["hidden_channels"],
        gat_out=config["model"]["gat"]["out_channels"],
        gat_heads=config["model"]["gat"]["heads"],
        dropout=config["model"]["classifier"]["dropout"],
        connect_by=config["model"]["graph"]["connect_by"],
    )

    trainer = StrawberryTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        checkpoint_dir=config["training"]["checkpoint_dir"],
        best_model_name=config["training"]["best_checkpoint_name"],
    )

    trainer.fit(
        epochs=config["training"]["epochs"],
        patience=config["training"]["patience"],
        class_to_idx=class_to_idx,
    )

    trainer.plot_and_save_history(
        output_png="outputs/figures/training_history.png",
        output_json="outputs/metrics/train_history.json",
    )
    print("=" * 65)
    print(f"[OK] Training complete. Best model checkpoint saved to: {trainer.best_model_path}")
    print("=" * 65)


if __name__ == "__main__":
    run_training()
