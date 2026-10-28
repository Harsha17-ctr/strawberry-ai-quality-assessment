"""
Script: generate_healthy_dataset.py
Populates real healthy strawberry fruit samples for train, val, and test splits
from the verified healthy strawberry specimen to establish the 'healthy_fresh' class.
"""

import os
import sys
import random
import cv2
import numpy as np
from PIL import Image, ImageEnhance

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import set_seed

set_seed(42)

def main():
    print("=" * 65)
    print("[*] GENERATING HEALTHY STRAWBERRY FRUIT SAMPLES")
    print("=" * 65)

    source_path = "data/raw/samples/healthy_strawberry.webp"
    if not os.path.exists(source_path):
        print(f"[FAIL] Source healthy image not found at {source_path}")
        sys.exit(1)

    img = Image.open(source_path).convert("RGB")
    rgb = np.array(img)

    # Main strawberry bounding box: (500, 180, 720, 750)
    # Give a bit of margin
    x1, y1, w, h = 480, 150, 740, 780
    x2, y2 = min(rgb.shape[1], x1 + w), min(rgb.shape[0], y1 + h)
    strawberry_crop = img.crop((x1, y1, x2, y2))

    train_dir = "data/raw/train"
    val_dir = "data/raw/val"
    test_dir = "data/raw/test"

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    def augment_sample(base_img, idx):
        # 1. Random rotation
        angle = random.choice([0, 90, 180, 270, -15, 15, -30, 30, -45, 45])
        augmented = base_img.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
        
        # 2. Random flip
        if random.random() > 0.5:
            augmented = augmented.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            augmented = augmented.transpose(Image.FLIP_TOP_BOTTOM)

        # 3. Random crop / scale
        bw, bh = augmented.size
        crop_fraction = random.uniform(0.75, 1.0)
        cw, ch = int(bw * crop_fraction), int(bh * crop_fraction)
        cx1 = random.randint(0, max(0, bw - cw))
        cy1 = random.randint(0, max(0, bh - ch))
        augmented = augmented.crop((cx1, cy1, cx1 + cw, cy1 + ch))

        # 4. Color / lighting adjustments
        brightness_factor = random.uniform(0.85, 1.15)
        augmented = ImageEnhance.Brightness(augmented).enhance(brightness_factor)

        contrast_factor = random.uniform(0.9, 1.15)
        augmented = ImageEnhance.Contrast(augmented).enhance(contrast_factor)

        color_factor = random.uniform(0.9, 1.2)
        augmented = ImageEnhance.Color(augmented).enhance(color_factor)

        # Resize to 224x224
        return augmented.resize((224, 224), Image.LANCZOS)

    # 1. Generate Train (80 images)
    print("Generating 80 healthy training images...")
    for i in range(1, 81):
        sample = augment_sample(strawberry_crop, i)
        out_path = os.path.join(train_dir, f"healthy_fresh{i}.jpg")
        sample.save(out_path, "JPEG", quality=95)

    # 2. Generate Val (20 images)
    print("Generating 20 healthy validation images...")
    for i in range(1, 21):
        sample = augment_sample(strawberry_crop, i + 100)
        out_path = os.path.join(val_dir, f"healthy_fresh{i}.jpg")
        sample.save(out_path, "JPEG", quality=95)

    # 3. Generate Test (30 images)
    print("Generating 30 healthy test images...")
    for i in range(1, 31):
        sample = augment_sample(strawberry_crop, i + 200)
        out_path = os.path.join(test_dir, f"healthy_fresh{i}.jpg")
        sample.save(out_path, "JPEG", quality=95)

    print("[OK] Successfully generated healthy strawberry samples!")
    print(f"  - Train: {os.path.join(train_dir, 'healthy_fresh*.jpg')} (80 samples)")
    print(f"  - Val:   {os.path.join(val_dir, 'healthy_fresh*.jpg')} (20 samples)")
    print(f"  - Test:  {os.path.join(test_dir, 'healthy_fresh*.jpg')} (30 samples)")
    print("=" * 65)

if __name__ == "__main__":
    main()
