"""
Unit tests for GAT Model and End-to-End ViT-GAT Classifier.
Tests forward pass, tensor shapes, gradient flow, and loss computation.
"""

import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gat_model import StrawberryGAT
from src.classifier import StrawberryViTGATClassifier


def test_gat_model():
    """Verify StrawberryGAT forward pass and shapes."""
    gat = StrawberryGAT(in_channels=192, hidden_channels=64, out_channels=32, heads=2)
    dummy_x = torch.randn(196 * 2, 192)
    # Simple self-loop edges
    edges = torch.arange(196 * 2).repeat(2, 1)
    batch = torch.tensor([0] * 196 + [1] * 196)

    graph_rep, node_rep = gat(dummy_x, edges, batch)
    assert graph_rep.shape == (2, 64)  # 2 * out_channels
    assert node_rep.shape == (392, 32)


def test_classifier_forward_and_backward():
    """Verify complete forward pass, loss computation, and backprop gradient flow."""
    model = StrawberryViTGATClassifier(
        num_classes=5,
        pretrained_vit=False,  # faster for test without re-downloading
        freeze_vit=True,
    )
    dummy_images = torch.randn(2, 3, 224, 224)
    out = model(dummy_images, return_explanations=True)

    disease_logits = out["disease_logits"]
    assert disease_logits.shape == (2, 5)
    assert not torch.isnan(disease_logits).any(), "Logits must not contain NaN"
    assert torch.isfinite(disease_logits).all(), "Logits must be finite"

    # Verify backpropagation
    targets = torch.tensor([1, 2], dtype=torch.long)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(disease_logits, targets)

    assert torch.isfinite(loss)
    loss.backward()

    # Verify that GAT and classifier linear weights received gradients
    has_grad = False
    for param in model.classifier.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "Classifier parameters must receive gradients during backprop!"


if __name__ == "__main__":
    test_gat_model()
    test_classifier_forward_and_backward()
    print("[OK] All model & gradient flow tests passed successfully!")
