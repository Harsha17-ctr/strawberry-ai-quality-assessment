"""
PyTorch Dataset and DataLoader implementation for Strawberry AI.
Loads images based on the verified dataset manifest with class imbalance handling.
"""

import os
import json
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from src.preprocessing import get_train_transforms, get_val_transforms, load_image_safely


class StrawberryDataset(Dataset):
    """Dataset for Strawberry pathology and freshness assessment."""

    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        raw_dir: Optional[str] = None,
        transform=None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.split = split
        self.transform = transform

        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        if raw_dir is None:
            self.raw_dir = manifest.get("root_dir", "data/raw")
        else:
            self.raw_dir = raw_dir

        if split not in manifest:
            raise ValueError(f"Split '{split}' not found in manifest. Available: {list(manifest.keys())}")

        self.samples = manifest[split]

        # Determine class mappings
        if class_to_idx is None:
            # Use sorted unique disease classes
            all_diseases = sorted(list({s["disease"] for s in self.samples if s.get("disease")}))
            self.class_to_idx = {cls: idx for idx, cls in enumerate(all_diseases)}
        else:
            self.class_to_idx = class_to_idx

        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, dict]:
        sample = self.samples[idx]
        image_path = os.path.join(self.raw_dir, sample["rel_path"])

        try:
            image = load_image_safely(image_path)
        except Exception as e:
            # Fallback to an empty black image in case of I/O error
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform is not None:
            image = self.transform(image)

        label_name = sample["disease"]
        label_idx = self.class_to_idx.get(label_name, 0)

        metadata = {
            "rel_path": sample["rel_path"],
            "filename": sample["filename"],
            "disease": label_name,
            "severity_level": str(sample.get("severity_level") or "N/A"),
            "has_json": bool(sample.get("has_json", False)),
        }

        return image, label_idx, metadata

    def compute_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency class weights to counteract class imbalance."""
        counts = torch.zeros(len(self.class_to_idx))
        for sample in self.samples:
            cls_name = sample["disease"]
            if cls_name in self.class_to_idx:
                counts[self.class_to_idx[cls_name]] += 1
        
        # Avoid division by zero
        counts = torch.clamp(counts, min=1.0)
        total = counts.sum()
        weights = total / (len(self.class_to_idx) * counts)
        return weights


def get_dataloaders(
    config: dict,
    manifest_path: Optional[str] = None,
    raw_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, Dict[str, int], Dict[int, str]]:
    """Build train, val, and test DataLoaders along with balanced loss weights."""
    if manifest_path is None:
        manifest_path = config["dataset"]["splits_manifest"]
    if raw_dir is None:
        raw_dir = config["dataset"]["raw_dir"]

    img_size = config["dataset"]["image_size"]
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    train_tf = get_train_transforms(img_size)
    val_tf = get_val_transforms(img_size)

    # Define standard class to idx mapping from config
    classes = config["dataset"]["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for i, c in enumerate(classes)}

    train_dataset = StrawberryDataset(manifest_path, split="train", raw_dir=raw_dir, transform=train_tf, class_to_idx=class_to_idx)
    val_dataset = StrawberryDataset(manifest_path, split="val", raw_dir=raw_dir, transform=val_tf, class_to_idx=class_to_idx)
    test_dataset = StrawberryDataset(manifest_path, split="test", raw_dir=raw_dir, transform=val_tf, class_to_idx=class_to_idx)

    class_weights = train_dataset.compute_class_weights()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader, class_weights, class_to_idx, idx_to_class
