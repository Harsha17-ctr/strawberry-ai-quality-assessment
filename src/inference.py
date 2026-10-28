"""
Inference Module for Strawberry AI.
Provides single-image prediction and explainability for CLI and Web interfaces.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from typing import Dict, Optional, Union
import torch
from PIL import Image

from src.utils import load_config, get_device
from src.preprocessing import get_inference_transforms, load_image_safely, remove_strawberry_leaves
from src.classifier import StrawberryViTGATClassifier
from src.explainability import StrawberryExplainer


class StrawberryPredictor:
    """Performs inference and generates explanations for uploaded strawberry images."""

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_strawberry_gat.pth",
        config_path: str = "config.yaml",
        device: Optional[torch.device] = None,
    ):
        self.config = load_config(config_path)
        self.device = device or get_device()
        self.checkpoint_path = checkpoint_path

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_cfg = checkpoint.get("model_config", {})

        self.model = StrawberryViTGATClassifier(
            num_classes=model_cfg.get("num_classes", self.config["dataset"]["num_classes"]),
            vit_model_name=model_cfg.get("vit_model_name", self.config["model"]["vit_backbone"]),
            pretrained_vit=False,
            freeze_vit=True,
            gat_hidden=model_cfg.get("gat_hidden", self.config["model"]["gat"]["hidden_channels"]),
            gat_out=model_cfg.get("gat_out", self.config["model"]["gat"]["out_channels"]),
            gat_heads=model_cfg.get("gat_heads", self.config["model"]["gat"]["heads"]),
            dropout=self.config["model"]["classifier"]["dropout"],
            connect_by=model_cfg.get("connect_by", self.config["model"]["graph"]["connect_by"]),
            include_severity_head=model_cfg.get("include_severity_head", True),
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.class_to_idx = checkpoint.get("class_to_idx") or {
            c: i for i, c in enumerate(self.config["dataset"]["classes"])
        }
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        self.transform = get_inference_transforms(self.config["dataset"]["image_size"])
        self.explainer = StrawberryExplainer(self.model, device=self.device)

    def predict(
        self,
        image_input: Union[str, Image.Image],
        explain: bool = True,
        remove_leaves: bool = False,
        background_mode: str = "white",
    ) -> Dict:
        """
        Predict strawberry disease class and return explainability package.
        
        Args:
            image_input: file path string or PIL Image
            explain: whether to compute visual saliency and graph attention overlays
            remove_leaves: whether to segment and remove green leaves/calyx before analysis
            background_mode: background color for leaf removal ('white', 'neutral', 'blur', 'black')
        """
        if isinstance(image_input, str):
            pil_image = load_image_safely(image_input)
        else:
            pil_image = image_input.convert("RGB")

        # Process leaf removal if requested
        leaf_stats = None
        leaf_mask = None
        analysis_image = pil_image

        if remove_leaves:
            fruit_only, leaf_mask, leaf_stats = remove_strawberry_leaves(
                pil_image, background_mode=background_mode
            )
            analysis_image = fruit_only

        tensor = self.transform(analysis_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(tensor)
            logits = out["disease_logits"]
            probs = torch.softmax(logits, dim=-1)[0]
            top_prob, top_idx = torch.topk(probs, k=len(self.idx_to_class))

        predicted_class = self.idx_to_class[top_idx[0].item()]
        confidence = float(top_prob[0].item())
        display_name = self.config["dataset"]["class_display_names"].get(predicted_class, predicted_class)

        # Build ranking list
        all_class_scores = []
        for i in range(len(top_idx)):
            cls_name = self.idx_to_class[top_idx[i].item()]
            disp = self.config["dataset"]["class_display_names"].get(cls_name, cls_name)
            all_class_scores.append({
                "class": cls_name,
                "display_name": disp,
                "confidence": float(top_prob[i].item()),
            })

        # Severity rating
        severity_est = None
        if "severity_logits" in out:
            sev_probs = torch.softmax(out["severity_logits"], dim=-1)[0]
            sev_idx = int(torch.argmax(sev_probs).item())
            severity_est = "Level 1 (Mild)" if sev_idx == 0 else "Level 2 (Severe)"

        quality_report = self.explainer.get_quality_assessment(predicted_class, confidence, severity_est)

        result = {
            "predicted_class": predicted_class,
            "display_name": display_name,
            "confidence": confidence,
            "severity_estimate": severity_est,
            "all_classes": all_class_scores,
            "quality_grade": quality_report["grade"],
            "category": quality_report["category"],
            "freshness_impact": quality_report["freshness_impact"],
            "freshness_score": quality_report.get("freshness_score", 70),
            "recommendation": quality_report["recommendation"],
            "original_image": pil_image.resize((224, 224)),
            "analysis_image": analysis_image.resize((224, 224)),
            "leaves_removed": remove_leaves,
            "leaf_stats": leaf_stats,
        }

        if explain:
            explanation = self.explainer.explain_image(
                tensor, analysis_image, target_class_idx=top_idx[0].item()
            )
            result["explanation"] = explanation

        # Generate human-friendly clear explanation
        result["human_explanation"] = self.explainer.generate_human_explanation(
            predicted_class=predicted_class,
            confidence=confidence,
            leaf_stats=leaf_stats,
            xai_result=result.get("explanation"),
        )

        return result
