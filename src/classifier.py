"""
Unified ViT-GAT Strawberry Freshness and Disease Assessment Model.

End-to-End Pipeline:
Input Image -> ViT Patch Encoder -> Patch Graph Builder -> GAT Network -> Fused Readout -> Classification Head
"""

from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.vit_encoder import ViTPatchEncoder
from src.graph_builder import PatchGraphBuilder
from src.gat_model import StrawberryGAT


class StrawberryViTGATClassifier(nn.Module):
    """
    Intelligent Strawberry Assessment System combining Vision Transformer
    and Graph Attention Network with Explainable feature representations.
    """

    def __init__(
        self,
        num_classes: int = 7,
        vit_model_name: str = "vit_tiny_patch16_224",
        pretrained_vit: bool = True,
        freeze_vit: bool = True,
        patch_size: int = 16,
        image_size: int = 224,
        gat_hidden: int = 128,
        gat_out: int = 64,
        gat_heads: int = 4,
        dropout: float = 0.25,
        connect_by: str = "hybrid",
        include_severity_head: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.include_severity_head = include_severity_head
        self.gat_hidden = gat_hidden
        self.gat_out = gat_out
        self.gat_heads = gat_heads
        self.connect_by = connect_by

        # 1. Vision Transformer Backbone
        self.encoder = ViTPatchEncoder(
            model_name=vit_model_name,
            pretrained=pretrained_vit,
            freeze=freeze_vit,
            patch_size=patch_size,
            image_size=image_size,
        )

        embed_dim = self.encoder.embed_dim  # 192 for vit_tiny

        # 2. Graph Construction
        self.graph_builder = PatchGraphBuilder(
            grid_size=self.encoder.grid_size,
            connect_by=connect_by,
            similarity_threshold=0.5,
            top_k_similarity=4,
            include_self_loops=True,
        )

        # 3. Graph Attention Network
        self.gat = StrawberryGAT(
            in_channels=embed_dim,
            hidden_channels=gat_hidden,
            out_channels=gat_out,
            heads=gat_heads,
            dropout=dropout,
        )

        # Graph representation size is (2 * gat_out) because of Mean + Max pooling
        graph_rep_dim = 2 * gat_out  # 128
        fused_dim = graph_rep_dim + embed_dim  # 128 + 192 = 320

        # 4. Fusion and Classification Head
        self.fusion_norm = nn.LayerNorm(fused_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # 5. Multi-task Severity Head (Level 1 vs Level 2)
        if include_severity_head:
            self.severity_classifier = nn.Sequential(
                nn.Linear(fused_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 2),
            )

    def forward(
        self,
        images: torch.Tensor,
        return_explanations: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: Tensor of shape (B, 3, 224, 224)
            return_explanations: whether to return intermediate graph & patch representations
            
        Returns:
            dict containing:
                - 'disease_logits': (B, num_classes)
                - 'severity_logits': (B, 2) if severity head enabled
                - 'node_rep': (B * 196, out_channels)
                - 'graph_rep': (B, 2 * out_channels)
                - 'edge_index': (2, Total_edges)
                - 'patch_tokens': (B, 196, embed_dim)
        """
        B = images.size(0)

        # Step A: ViT Feature Extraction
        patch_tokens, cls_token = self.encoder(images)  # (B, 196, 192), (B, 192)

        # Step B: Graph Construction
        x_nodes, edge_index, batch_idx = self.graph_builder.build_batch_graph(patch_tokens)

        # Step C: GAT Message Passing
        graph_rep, node_rep = self.gat(
            x=x_nodes,
            edge_index=edge_index,
            batch=batch_idx,
            return_attention=return_explanations,
        )

        # Step D: Fusion of GNN Graph Structure + Global ViT CLS context
        fused = torch.cat([graph_rep, cls_token], dim=-1)  # (B, 320)
        fused = self.fusion_norm(fused)

        # Step E: Disease Classification
        disease_logits = self.classifier(fused)

        out = {
            "disease_logits": disease_logits,
            "graph_rep": graph_rep,
            "fused_features": fused,
        }

        if self.include_severity_head:
            out["severity_logits"] = self.severity_classifier(fused)

        if return_explanations:
            out["node_rep"] = node_rep
            out["edge_index"] = edge_index
            out["batch_idx"] = batch_idx
            out["patch_tokens"] = patch_tokens

        return out
