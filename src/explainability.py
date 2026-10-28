"""
Explainable AI (XAI) Module for Strawberry ViT-GAT Architecture.

Provides multi-level visual and graph explanations:
1. ViT Image-level Saliency / Attention Map:
   Calculates gradient attribution over patch embeddings and upsamples to 224x224 heatmap overlay.
2. GAT Graph-level Node & Edge Explanations:
   Extracts top attended edges (alpha_ij) and influential lesion nodes,
   mapping them directly back to bounding boxes on the original strawberry image.
3. Agronomic Freshness & Pathology Interpretation:
   Translates detected disease and severity into commercial quality grades.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class StrawberryExplainer:
    """Generates visual and graph-theoretic explanations for model predictions."""

    def __init__(self, model, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()

    def explain_image(
        self,
        image_tensor: torch.Tensor,
        original_pil_image: Image.Image,
        target_class_idx: Optional[int] = None,
        top_k_nodes: int = 10,
    ) -> Dict:
        """
        Generate comprehensive image-level and graph-level explanation for a single strawberry image.
        
        Args:
            image_tensor: (1, 3, 224, 224)
            original_pil_image: PIL Image object
            target_class_idx: class index to explain (defaults to model's predicted class)
            top_k_nodes: number of top influential patch nodes to highlight
            
        Returns:
            dict containing:
                - 'predicted_class': int
                - 'predicted_prob': float
                - 'saliency_map': np.ndarray (224, 224)
                - 'overlay_image': PIL.Image
                - 'top_nodes': list of dicts with patch coords and importance scores
                - 'top_edges': list of dicts with (src, dst) coords and attention weights
        """
        self.model.zero_grad()
        img = image_tensor.to(self.device).requires_grad_(True)

        # 1. Forward pass extracting patch tokens and graph outputs
        out = self.model(img, return_explanations=True)
        logits = out["disease_logits"]
        probs = torch.softmax(logits, dim=-1)

        if target_class_idx is None:
            target_class_idx = int(torch.argmax(probs, dim=-1).item())

        prob_score = float(probs[0, target_class_idx].item())
        target_score = logits[0, target_class_idx]

        # 2. Compute gradients with respect to patch tokens for image-level saliency
        patch_tokens = out["patch_tokens"]  # (1, 196, D)
        grads = torch.autograd.grad(target_score, patch_tokens, retain_graph=True)[0]  # (1, 196, D)

        # Patch importance is L2 norm of gradients
        patch_importance = torch.norm(grads[0], dim=-1).detach().cpu().numpy()  # (196,)
        
        # Reshape to 14x14 grid
        grid_size = self.model.encoder.grid_size
        heatmap_grid = patch_importance.reshape((grid_size, grid_size))

        # Normalize to [0, 1]
        norm_min, norm_max = heatmap_grid.min(), heatmap_grid.max()
        if norm_max > norm_min:
            heatmap_grid = (heatmap_grid - norm_min) / (norm_max - norm_min)
        else:
            heatmap_grid = np.zeros_like(heatmap_grid)

        # Upsample 14x14 heatmap to 224x224
        heatmap_tensor = torch.tensor(heatmap_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        upsampled_heatmap = F.interpolate(
            heatmap_tensor, size=(224, 224), mode="bicubic", align_corners=False
        ).squeeze().clamp(0, 1).numpy()

        # Generate blended heatmap overlay
        resized_pil = original_pil_image.convert("RGB").resize((224, 224))
        rgb_np = np.array(resized_pil) / 255.0

        # Use modern matplotlib colormaps API (compatible with Matplotlib >= 3.8)
        try:
            colormap = plt.get_cmap("jet")
        except AttributeError:
            colormap = cm.jet

        colored_heatmap = colormap(upsampled_heatmap)[:, :, :3]
        blended = 0.55 * rgb_np + 0.45 * colored_heatmap
        blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
        overlay_pil = Image.fromarray(blended)

        # 3. Graph-level node and edge explanation
        # Identify top-K nodes
        sorted_node_indices = np.argsort(patch_importance)[::-1]
        top_node_records = []
        for rank, node_idx in enumerate(sorted_node_indices[:top_k_nodes]):
            y1, y2, x1, x2 = self.model.encoder.get_patch_coordinates(int(node_idx))
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            top_node_records.append({
                "rank": rank + 1,
                "node_idx": int(node_idx),
                "importance": float(patch_importance[node_idx]),
                "bbox": [x1, y1, x2, y2],
                "center": [center_x, center_y],
            })

        # Extract top attended GAT edges
        edge_index = out["edge_index"].detach().cpu()
        top_edge_records = []
        if getattr(self.model.gat, "last_attention_weights", None) is not None:
            _, alpha = self.model.gat.last_attention_weights
            alpha_np = alpha.detach().cpu().numpy()
            if alpha_np.ndim > 1:
                alpha_np = alpha_np.mean(axis=-1)
            sorted_edge_indices = np.argsort(alpha_np)[::-1]

            for e_idx in sorted_edge_indices[:15]:
                src = int(edge_index[0, e_idx].item())
                dst = int(edge_index[1, e_idx].item())
                if src != dst:  # Ignore self-loops for visualization clarity
                    src_y1, src_y2, src_x1, src_x2 = self.model.encoder.get_patch_coordinates(src)
                    dst_y1, dst_y2, dst_x1, dst_x2 = self.model.encoder.get_patch_coordinates(dst)
                    top_edge_records.append({
                        "src_node": src,
                        "dst_node": dst,
                        "weight": float(alpha_np[e_idx]),
                        "src_center": [(src_x1 + src_x2) // 2, (src_y1 + src_y2) // 2],
                        "dst_center": [(dst_x1 + dst_x2) // 2, (dst_y1 + dst_y2) // 2],
                    })

        return {
            "target_class_idx": target_class_idx,
            "confidence": prob_score,
            "raw_saliency_grid": heatmap_grid,
            "upsampled_saliency": upsampled_heatmap,
            "overlay_image": overlay_pil,
            "top_nodes": top_node_records,
            "top_edges": top_edge_records,
        }

    @staticmethod
    def get_quality_assessment(disease_class: str, confidence: float, severity_pred: Optional[str] = None) -> Dict:
        """
        Map predicted disease and confidence into an agricultural quality grade,
        freshness score, and management recommendation.
        """
        disease_info = {
            "healthy_fresh": {
                "name": "Healthy Fresh Strawberry",
                "category": "Prime Quality / Disease-Free",
                "grade": "Grade A (Prime Fresh / Healthy)",
                "freshness_impact": "None: Intact glossy pericarp, firm flesh, vibrant natural red color.",
                "freshness_score": 98,
                "shelf_life": "5 to 7 days (refrigerated at 2-4°C)",
                "edibility": "100% Safe to eat! Fresh, ripe, and delicious. Wash with clean water before eating.",
                "symptoms": "No visible fungal lesions, mold spores, or sunken necrotic decay. Achenes (seeds) and fruit surface are completely intact.",
                "recommendation": "Store unwashed in cold refrigeration (2–4°C) with breathable ventilation to maintain maximum freshness and firmness.",
            },
            "anthracnose_fruit_rot": {
                "name": "Anthracnose Fruit Rot (Colletotrichum acutatum)",
                "category": "Fruit Necrosis / Decay",
                "grade": "Grade C (Substandard / Post-harvest Rot)",
                "freshness_impact": "Severe: Sunken water-soaked lesions leading to complete fruit decay.",
                "freshness_score": 20,
                "shelf_life": "0 days (actively rotting)",
                "edibility": "DO NOT EAT. Fungal mycelium penetrates deeply into the pulp, creating bitter rot.",
                "symptoms": "Round, sunken, brown-to-black firm lesions on the berry that develop salmon-pink spore masses in moist conditions.",
                "recommendation": "Cull infected strawberries immediately; avoid picking wet plants; sanitize harvest trays.",
            },
            "blossom_blight": {
                "name": "Blossom Blight (Botrytis / Colletotrichum complex)",
                "category": "Floral Organ Infection",
                "grade": "Grade B (Compromised Flowering)",
                "freshness_impact": "Moderate to High: Petal browning causes unfertilized, misshapen fruit.",
                "freshness_score": 65,
                "shelf_life": "2 to 3 days (inspect berry stem)",
                "edibility": "Mature berries are safe if calyx and stem are clean, but check for early rot at the stem attachment.",
                "symptoms": "Brown necrosis on petals, pistils, and floral calyx. Blighted blossoms dry up and fail to set healthy fruit.",
                "recommendation": "Improve canopy aeration and maintain proactive bloom protective sprays.",
            },
            "gray_mold": {
                "name": "Gray Mold (Botrytis cinerea)",
                "category": "Fruit Soft Rot & Sporulation",
                "grade": "Grade C (Unmarketable / High Post-harvest Spoilage)",
                "freshness_impact": "Critical: Velvet gray fungal spores rapidly cross-contaminate stored berries.",
                "freshness_score": 10,
                "shelf_life": "0 days (spoilage hazard)",
                "edibility": "DO NOT EAT. The fruit is colonized by fungal hyphae and produces allergen-provoking gray mold spores.",
                "symptoms": "Soft, watery brown decay that rapidly covers the entire berry with a dense, velvety gray-brown spore carpet.",
                "recommendation": "Rapid pre-cooling (below 4°C); remove culled berries from cold chain immediately to prevent nest rot.",
            },
            "powdery_mildew_fruit": {
                "name": "Powdery Mildew Fruit (Podosphaera aphanis)",
                "category": "Fruit Mycelial Colonization",
                "grade": "Grade C (Severe Surface Coating)",
                "freshness_impact": "Severe: Powdery white coating diminishes market value and flavor quality.",
                "freshness_score": 35,
                "shelf_life": "1 to 2 days (surface compromised)",
                "edibility": "Not recommended. The white powdery coating alters taste, promotes drying, and diminishes consumer safety.",
                "symptoms": "Patches of white talcum powder-like fungal growth on fruit surface, leading to hard, seed-exposed, unripened patches.",
                "recommendation": "Apply biological or sulfur control agents; ensure adequate air circulation.",
            },
        }

        info = disease_info.get(disease_class, {
            "name": disease_class,
            "category": "General Quality Assessment",
            "grade": "Grade B (Commercial)",
            "freshness_impact": "Potential shelf-life reduction.",
            "freshness_score": 60,
            "shelf_life": "2 to 3 days",
            "edibility": "Inspect carefully before consumption.",
            "symptoms": "Tissue irregularities detected on strawberry surface.",
            "recommendation": "Inspect and isolate affected produce.",
        })

        return info

    def generate_human_explanation(
        self,
        predicted_class: str,
        confidence: float,
        leaf_stats: Optional[Dict] = None,
        xai_result: Optional[Dict] = None,
    ) -> Dict:
        """
        Generate an easy-to-understand, crystal-clear explanation for a college student,
        farmer, or consumer uploading a strawberry photo.
        """
        assessment = self.get_quality_assessment(predicted_class, confidence)
        conf_pct = round(confidence * 100, 2)
        freshness_score = assessment.get("freshness_score", 70)

        # Confidence qualitative tier
        if conf_pct >= 90:
            conf_tier = "High Confidence"
            conf_desc = f"The model is very confident ({conf_pct}%) in this diagnosis."
        elif conf_pct >= 70:
            conf_tier = "Moderate Confidence"
            conf_desc = f"The model diagnosed this with reasonable confidence ({conf_pct}%)."
        else:
            conf_tier = "Low/Ambiguous"
            conf_desc = f"Confidence is low ({conf_pct}%); subtle or overlapping symptoms may be present."

        # Leaf removal commentary
        if leaf_stats and leaf_stats.get("leaves_detected", False):
            leaf_text = (
                f"The leafy calyx and surrounding green foliage ({leaf_stats['leaf_area_pct']}% of the image) "
                f"were digitally isolated and removed. This allows the AI model to inspect the strawberry berry "
                f"skin, achenes (seeds), and pericarp tissue without foliage clutter."
            )
        else:
            leaf_text = (
                "Minimal or no green leaves were detected. The analysis was conducted across the strawberry fruit tissue."
            )

        # XAI Heatmap commentary
        saliency_text = (
            "The Vision Transformer (ViT) patch attention map highlights key diagnostic regions with bright red and "
            "yellow overlays. These areas represent where the model found distinctive tissue texture, spore growth, "
            "or discoloration that strongly influenced its decision."
        )

        return {
            "verdict_title": f"{assessment['name']}",
            "quality_grade": assessment["grade"],
            "freshness_score": freshness_score,
            "confidence_tier": conf_tier,
            "confidence_desc": conf_desc,
            "edibility": assessment["edibility"],
            "shelf_life": assessment["shelf_life"],
            "visual_symptoms": assessment["symptoms"],
            "freshness_impact": assessment["freshness_impact"],
            "management_recommendation": assessment["recommendation"],
            "leaf_processing_summary": leaf_text,
            "saliency_explanation": saliency_text,
        }

