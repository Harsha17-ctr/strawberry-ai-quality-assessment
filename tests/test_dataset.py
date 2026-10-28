"""
Unit tests for Strawberry AI dataset and preprocessing pipeline.
"""

import os
import sys
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_config
from src.preprocessing import get_train_transforms, get_val_transforms
from src.data_loader import StrawberryDataset, get_dataloaders


def test_preprocessing_transforms():
    """Verify transform outputs have correct shape and type."""
    dummy_img = Image.new("RGB", (300, 400), (255, 100, 50))
    train_tf = get_train_transforms(image_size=224)
    val_tf = get_val_transforms(image_size=224)

    t_tensor = train_tf(dummy_img)
    v_tensor = val_tf(dummy_img)

    assert isinstance(t_tensor, torch.Tensor)
    assert t_tensor.shape == (3, 224, 224)
    assert isinstance(v_tensor, torch.Tensor)
    assert v_tensor.shape == (3, 224, 224)


def test_strawberry_dataset():
    """Verify loading samples from real dataset manifest."""
    config = load_config()
    manifest_path = config["dataset"]["splits_manifest"]
    raw_dir = config["dataset"]["raw_dir"]

    if not os.path.exists(manifest_path):
        pytest.skip(f"Manifest not found at {manifest_path}")

    dataset = StrawberryDataset(
        manifest_path=manifest_path,
        split="train",
        raw_dir=raw_dir,
        transform=get_val_transforms(224),
    )

    assert len(dataset) > 0, "Dataset should contain training samples"
    img, label, meta = dataset[0]

    assert img.shape == (3, 224, 224)
    assert isinstance(label, int)
    assert 0 <= label < config["dataset"]["num_classes"]
    assert "disease" in meta
    assert "filename" in meta


def test_class_weights():
    """Verify computation of class imbalance weights."""
    config = load_config()
    manifest_path = config["dataset"]["splits_manifest"]
    raw_dir = config["dataset"]["raw_dir"]

    if not os.path.exists(manifest_path):
        pytest.skip("Manifest not found")

    dataset = StrawberryDataset(
        manifest_path=manifest_path,
        split="train",
        raw_dir=raw_dir,
    )
    weights = dataset.compute_class_weights()
    assert weights.shape[0] == config["dataset"]["num_classes"]
    assert (weights > 0).all()


if __name__ == "__main__":
    test_preprocessing_transforms()
    test_strawberry_dataset()
    test_class_weights()
    print("[OK] All dataset unit tests passed successfully!")
