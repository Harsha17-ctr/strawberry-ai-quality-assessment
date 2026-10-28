"""
Smoke test script for Strawberry AI.
Quickly validates the complete train-validate-save-load-inference pipeline on a mini subset.
"""

import os
import sys
import tempfile
import torch
from torch.utils.data import Subset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.utils import load_config, set_seed, get_device
from src.data_loader import get_dataloaders
from src.classifier import StrawberryViTGATClassifier
from src.train import StrawberryTrainer
from src.evaluate import ModelEvaluator


def main():
    print("=" * 65)
    print("[*] RUNNING END-TO-END SMOKE TEST")
    print("=" * 65)

    config = load_config()
    set_seed(42)
    device = get_device()

    print(f"Hardware: {device}")
    train_loader, val_loader, test_loader, _, class_to_idx, _ = get_dataloaders(config)

    # Subsample 16 samples for training and 8 for validation
    train_subset = Subset(train_loader.dataset, list(range(min(16, len(train_loader.dataset)))))
    val_subset = Subset(val_loader.dataset, list(range(min(8, len(val_loader.dataset)))))

    smoke_train_loader = torch.utils.data.DataLoader(train_subset, batch_size=4, shuffle=True)
    smoke_val_loader = torch.utils.data.DataLoader(val_subset, batch_size=4, shuffle=False)

    print("[1/5] Building ViT-GAT Classifier...")
    model = StrawberryViTGATClassifier(
        num_classes=config["dataset"]["num_classes"],
        vit_model_name=config["model"]["vit_backbone"],
        pretrained_vit=True,
        freeze_vit=True,
        gat_hidden=64,
        gat_out=32,
        gat_heads=2,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = StrawberryTrainer(
            model=model,
            train_loader=smoke_train_loader,
            val_loader=smoke_val_loader,
            lr=1e-3,
            device=device,
            checkpoint_dir=tmp_dir,
            best_model_name="smoke_test.pth",
        )

        print("[2/5] Running mini training epoch...")
        train_loss = trainer.train_epoch()
        print(f"  Train loss: {train_loss:.4f}")

        print("[3/5] Running validation step...")
        val_loss, val_acc, val_f1 = trainer.validate()
        print(f"  Val loss: {val_loss:.4f}, Val Acc: {val_acc*100:.1f}%, Val F1: {val_f1:.4f}")

        print("[4/5] Testing checkpoint save & reload...")
        saved_path = trainer.save_checkpoint(epoch=1, val_f1=val_f1, val_acc=val_acc, class_to_idx=class_to_idx)
        assert os.path.exists(saved_path), "Checkpoint was not saved!"

        evaluator = ModelEvaluator(checkpoint_path=saved_path, device=device)
        print("  Checkpoint reloaded successfully into fresh model instance.")

        print("[5/5] Testing single image forward pass...")
        dummy_img = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            out = evaluator.model(dummy_img, return_explanations=True)
            logits = out["disease_logits"]
            probs = torch.softmax(logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()

        print(f"  Inference logits shape: {logits.shape}")
        print(f"  Predicted class index: {pred_class} ({evaluator.idx_to_class[pred_class]})")

    print("=" * 65)
    print("[OK] Smoke test passed 100%! Full pipeline is verified and operational.")
    print("=" * 65)


if __name__ == "__main__":
    main()
