"""
Script: prepare_dataset.py
Safely extracts raw dataset archive and generates verified dataset manifest.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.utils import load_config
from src.dataset_inspector import DatasetInspector


def main():
    print("=" * 65)
    print("[*] PREPARING STRAWBERRY DATASET")
    print("=" * 65)

    config = load_config()
    zip_path = config["dataset"]["archive_path"]
    raw_dir = config["dataset"]["raw_dir"]
    manifest_path = config["dataset"]["splits_manifest"]

    if not os.path.isabs(raw_dir):
        raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", raw_dir))
    if not os.path.isabs(manifest_path):
        manifest_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", manifest_path))

    # Check if raw_dir already contains extracted images
    existing_jpgs = 0
    if os.path.exists(raw_dir):
        for _, _, files in os.walk(raw_dir):
            existing_jpgs += sum(1 for f in files if f.lower().endswith(".jpg"))

    if existing_jpgs >= 3000:
        print(f"[INFO] Raw directory already populated with {existing_jpgs:,} images.")
        print(f"       Skipping extraction: {raw_dir}")
    else:
        print(f"Extracting archive safely from: {zip_path}")
        print(f"Target destination:             {raw_dir}")
        extracted, failed = DatasetInspector.extract_archive_safely(zip_path, raw_dir)
        print(f"[OK] Extraction complete: {extracted:,} files extracted, {failed} errors.")

    print("\nBuilding and verifying dataset manifest...")
    manifest = DatasetInspector.build_dataset_manifest(raw_dir, manifest_path)

    print(f"[OK] Manifest saved to: {manifest_path}")
    print("\nDataset Split Summary:")
    for split, count in manifest["split_summary"].items():
        print(f"  - {split:<15}: {count:>5} images")

    print("\nClass Breakdown Across Splits:")
    for cls_name, count in sorted(manifest["class_summary"].items(), key=lambda x: -x[1]):
        print(f"  - {cls_name:<25}: {count:>5} images")

    corrupted_count = len(manifest["corrupted_images"])
    if corrupted_count > 0:
        print(f"[WARN] Detected {corrupted_count} corrupted images (excluded from training).")
    else:
        print("[OK] All scanned images are 100% valid and readable!")

    print("=" * 65)


if __name__ == "__main__":
    main()
