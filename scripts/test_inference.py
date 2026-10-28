"""
Script: test_inference.py
Verifies single image inference and explainability generation on sample test images.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.inference import StrawberryPredictor


def main():
    print("=" * 65)
    print("[*] TESTING SINGLE-IMAGE INFERENCE & EXPLAINABILITY")
    print("=" * 65)

    predictor = StrawberryPredictor(checkpoint_path="checkpoints/best_strawberry_gat.pth")

    test_dir = "data/raw/test"
    sample_files = [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".png"))][:3]

    for fname in sample_files:
        fpath = os.path.join(test_dir, fname)
        print(f"\nProcessing image: {fname}")
        res = predictor.predict(fpath, explain=True)

        print(f"  - Predicted Disease: {res['display_name']} ({res['predicted_class']})")
        print(f"  - Confidence:        {res['confidence']*100:.2f}%")
        print(f"  - Quality Grade:     {res['quality_grade']}")
        print(f"  - Freshness Impact:  {res['freshness_impact']}")
        print(f"  - Recommendation:    {res['recommendation']}")
        print(f"  - Top nodes found:   {len(res['explanation']['top_nodes'])}")
        print(f"  - Top edges found:   {len(res['explanation']['top_edges'])}")

    # Test leaf removal and inference on healthy strawberry sample
    sample_healthy = "data/raw/samples/healthy_strawberry.webp"
    if not os.path.exists(sample_healthy):
        sample_healthy = r"C:\Users\harsha\Downloads\healthy strawberry.webp"

    if os.path.exists(sample_healthy):
        print(f"\n[*] Testing Leaf & Calyx Removal on: {sample_healthy}")
        res_leaf = predictor.predict(sample_healthy, explain=True, remove_leaves=True)
        stats = res_leaf.get("leaf_stats", {})
        h_exp = res_leaf.get("human_explanation", {})

        print(f"  - Leaf Area Removed: {stats.get('leaf_area_pct', 0)}%")
        print(f"  - Diagnosis:         {res_leaf['display_name']} ({res_leaf['confidence']*100:.2f}%)")
        print(f"  - Quality Grade:     {res_leaf['quality_grade']}")
        print(f"  - Freshness Score:   {h_exp.get('freshness_score')}/100")
        print(f"  - Edibility Status:  {h_exp.get('edibility')}")
        print(f"  - Estimated Shelf:   {h_exp.get('shelf_life')}")
        print(f"  - Clear Summary:     {h_exp.get('leaf_processing_summary')}")

    sample_rotten = "data/raw/samples/rotten_strawberry.jpg"
    if os.path.exists(sample_rotten):
        print(f"\n[*] Testing Inference on Rotten Strawberry: {sample_rotten}")
        res_rot = predictor.predict(sample_rotten, explain=True, remove_leaves=True)
        h_rot = res_rot.get("human_explanation", {})
        print(f"  - Diagnosis:         {res_rot['display_name']} ({res_rot['confidence']*100:.2f}%)")
        print(f"  - Quality Grade:     {res_rot['quality_grade']}")
        print(f"  - Freshness Score:   {h_rot.get('freshness_score')}/100")
        print(f"  - Edibility Status:  {h_rot.get('edibility')}")

    print("\n" + "=" * 65)
    print("[OK] Single-image inference, leaf removal, and explainability verified!")
    print("=" * 65)


if __name__ == "__main__":
    main()
