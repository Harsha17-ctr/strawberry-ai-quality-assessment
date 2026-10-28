"""
Dataset Inspector and Verification Module for Strawberry AI.

Performs robust integrity checks, parsing of Dataset.txt and Labelme JSON annotations,
image readability checks with Pillow, and builds reproducible dataset manifests.
"""

import os
import re
import json
import zipfile
from collections import Counter
from typing import Dict, List, Tuple, Optional
from PIL import Image


class DatasetInspector:
    """Inspects, extracts, verifies, and catalogs strawberry image datasets."""

    KNOWN_DISEASES = [
        "healthy_fresh",
        "anthracnose_fruit_rot",
        "blossom_blight",
        "gray_mold",
        "powdery_mildew_fruit",
    ]

    @staticmethod
    def verify_zip_integrity(zip_path: str) -> Tuple[bool, Optional[str]]:
        """Verify ZIP archive CRC checksums. Returns (is_valid, error_message)."""
        if not os.path.exists(zip_path):
            return False, f"File does not exist at {zip_path}"
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                first_bad = z.testzip()
                if first_bad is not None:
                    return False, f"Corrupted file inside archive: {first_bad}"
            return True, None
        except Exception as e:
            return False, f"Zip integrity verification error: {str(e)}"

    @staticmethod
    def parse_disease_label(filename: str) -> Optional[str]:
        """Extract standardized disease label from image or json filename."""
        base = os.path.basename(filename).lower()
        base = re.sub(r"\.(jpg|jpeg|png|json)$", "", base)
        # Remove trailing digits and punctuation
        prefix = re.sub(r"\d+$", "", base).rstrip("_- ")
        
        # Exact or substring match against known diseases
        for disease in DatasetInspector.KNOWN_DISEASES:
            if prefix == disease or disease in prefix:
                return disease
        return None

    @classmethod
    def inspect_archive(cls, zip_path: str) -> dict:
        """Inspect contents of ZIP archive without full extraction."""
        is_valid, err = cls.verify_zip_integrity(zip_path)
        if not is_valid:
            return {"valid": False, "error": err}

        summary = {
            "valid": True,
            "total_files": 0,
            "jpg_count": 0,
            "json_count": 0,
            "txt_count": 0,
            "subfolders": Counter(),
            "detected_classes": Counter(),
            "dataset_txt_found": False,
            "dataset_txt_preview": "",
            "sample_json_labels": Counter(),
        }

        with zipfile.ZipFile(zip_path, "r") as z:
            names = z.namelist()
            summary["total_files"] = len(names)

            for name in names:
                if name.endswith("/"):
                    continue
                parts = name.split("/")
                if len(parts) > 1:
                    summary["subfolders"][parts[0]] += 1

                if name.endswith(".jpg"):
                    summary["jpg_count"] += 1
                    disease = cls.parse_disease_label(name)
                    if disease:
                        summary["detected_classes"][disease] += 1
                elif name.endswith(".json"):
                    summary["json_count"] += 1
                elif name.endswith("Dataset.txt") or name.endswith(".txt"):
                    summary["txt_count"] += 1
                    if "Dataset.txt" in name:
                        summary["dataset_txt_found"] = True
                        try:
                            content = z.read(name).decode("utf-8", errors="ignore")
                            summary["dataset_txt_preview"] = content[:300].strip()
                        except Exception:
                            pass

            # Inspect sample JSON annotations
            sample_jsons = [f for f in names if f.endswith(".json")][:20]
            for jf in sample_jsons:
                try:
                    data = json.loads(z.read(jf).decode("utf-8"))
                    for shape in data.get("shapes", []):
                        lbl = shape.get("label")
                        if lbl:
                            summary["sample_json_labels"][lbl] += 1
                except Exception:
                    pass

        # Convert counters to standard dicts
        summary["subfolders"] = dict(summary["subfolders"])
        summary["detected_classes"] = dict(summary["detected_classes"])
        summary["sample_json_labels"] = dict(summary["sample_json_labels"])
        return summary

    @classmethod
    def extract_archive_safely(cls, zip_path: str, target_dir: str) -> Tuple[int, int]:
        """Safely extract ZIP archive into target directory without deleting any original data.
        Returns (extracted_count, corrupted_or_failed_count)."""
        is_valid, err = cls.verify_zip_integrity(zip_path)
        if not is_valid:
            raise ValueError(f"Cannot extract invalid ZIP: {err}")

        os.makedirs(target_dir, exist_ok=True)
        extracted = 0
        failed = 0

        with zipfile.ZipFile(zip_path, "r") as z:
            members = z.infolist()
            for member in members:
                try:
                    z.extract(member, target_dir)
                    extracted += 1
                except Exception:
                    failed += 1

        return extracted, failed

    @staticmethod
    def verify_single_image(image_path: str) -> bool:
        """Check if an image can be opened and verified without truncation/corruption."""
        try:
            with Image.open(image_path) as img:
                img.verify()
            # Double check conversion to RGB
            with Image.open(image_path) as img:
                img.convert("RGB")
            return True
        except Exception:
            return False

    @classmethod
    def build_dataset_manifest(cls, root_dir: str, manifest_save_path: str) -> dict:
        """Scan extracted directory, verify image integrity, parse ground truth and severity,
        and write structured manifest JSON."""
        manifest = {
            "root_dir": root_dir,
            "train": [],
            "val": [],
            "test": [],
            "severity_test": [],
            "corrupted_images": [],
            "class_summary": Counter(),
            "split_summary": Counter(),
        }

        # Scan all jpg files
        for root, _, files in os.walk(root_dir):
            for file in files:
                if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, root_dir).replace("\\", "/")

                # Verify image readability
                if not cls.verify_single_image(full_path):
                    manifest["corrupted_images"].append(rel_path)
                    continue

                disease = cls.parse_disease_label(file)
                if not disease:
                    continue

                # Check accompanying JSON for polygon annotations
                json_path = os.path.splitext(full_path)[0] + ".json"
                has_annotations = os.path.exists(json_path)

                # Determine split
                parts = rel_path.split("/")
                top_folder = parts[0].lower()
                split = None
                severity_level = None

                if top_folder == "train":
                    split = "train"
                elif top_folder == "val":
                    split = "val"
                elif top_folder == "test":
                    split = "test"
                elif "severity" in top_folder or (len(parts) > 1 and "level" in parts[1].lower()):
                    split = "severity_test"
                    if "level 1" in rel_path.lower():
                        severity_level = "Level 1"
                    elif "level 2" in rel_path.lower():
                        severity_level = "Level 2"

                record = {
                    "rel_path": rel_path,
                    "filename": file,
                    "disease": disease,
                    "has_json": has_annotations,
                    "split": split,
                    "severity_level": severity_level,
                }

                if split in manifest:
                    manifest[split].append(record)
                    manifest["class_summary"][disease] += 1
                    manifest["split_summary"][split] += 1

        manifest["class_summary"] = dict(manifest["class_summary"])
        manifest["split_summary"] = dict(manifest["split_summary"])

        os.makedirs(os.path.dirname(manifest_save_path), exist_ok=True)
        with open(manifest_save_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest
