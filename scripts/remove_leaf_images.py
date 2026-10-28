"""
Script: remove_leaf_images.py
Removes all foliar leaf images and annotations (angular_leafspot, leaf_spot, powdery_mildew_leaf)
from data/raw/train, data/raw/val, and data/raw/test so the dataset workspace contains
strictly strawberry fruit images.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dirs = [
        os.path.join(base_dir, "data", "raw", "train"),
        os.path.join(base_dir, "data", "raw", "val"),
        os.path.join(base_dir, "data", "raw", "test"),
    ]

    leaf_prefixes = ["angular_leafspot", "leaf_spot", "powdery_mildew_leaf"]
    deleted_count = 0
    remaining_count = 0

    print("=" * 65)
    print("🍓 REMOVING LEAF IMAGES FROM DATASET WORKSPACE")
    print("=" * 65)

    for d in dirs:
        if not os.path.exists(d):
            continue
        files = os.listdir(d)
        dir_deleted = 0
        for f in files:
            if any(f.startswith(prefix) for prefix in leaf_prefixes):
                fpath = os.path.join(d, f)
                try:
                    os.remove(fpath)
                    dir_deleted += 1
                except Exception as e:
                    print(f"Error removing {fpath}: {e}")
        remaining = len(os.listdir(d))
        print(f"Directory {os.path.basename(d)}: removed {dir_deleted} leaf files | {remaining} fruit files remain.")
        deleted_count += dir_deleted
        remaining_count += remaining

    print("=" * 65)
    print(f"Total leaf files removed: {deleted_count}")
    print(f"Total strawberry fruit files remaining in data/raw: {remaining_count}")
    print("Dataset now strictly contains only strawberry fruit!")
    print("=" * 65)

if __name__ == "__main__":
    main()
