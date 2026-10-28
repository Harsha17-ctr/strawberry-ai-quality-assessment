"""
Vision Transformer (ViT) Feature Extraction Module for Strawberry AI.

Extracts spatial patch embeddings from strawberry images using a pretrained
Vision Transformer (timm ViT), separates the CLS token from regional patch tokens,
and preserves spatial 2D grid correspondence.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import timm


class ViTPatchEncoder(nn.Module):
    """
    Vision Transformer Patch Feature Extractor.
    
    Transforms input images of shape (B, 3, 224, 224) into:
    - Regional patch tokens: (B, 196, embed_dim) where 196 = 14x14 grid of 16x16 patches.
    - Global CLS token: (B, embed_dim) representing whole-image aggregate context.
    """

    def __init__(
        self,
        model_name: str = "vit_tiny_patch16_224",
        pretrained: bool = True,
        freeze: bool = True,
        patch_size: int = 16,
        image_size: int = 224,
    ):
        super().__init__()
        self.model_name = model_name
        self.patch_size = patch_size
        self.image_size = image_size
        self.grid_size = image_size // patch_size  # 224 // 16 = 14
        self.num_patches = self.grid_size * self.grid_size  # 196

        # Initialize timm ViT backbone
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove default classification head
        )

        self.embed_dim = self.vit.num_features  # 192 for vit_tiny

        # Configurable freezing for transfer learning
        if freeze:
            for param in self.vit.parameters():
                param.requires_grad = False

        self._last_attention = None

    def unfreeze_last_n_blocks(self, n: int = 2) -> None:
        """Unfreeze the top N transformer blocks for fine-tuning."""
        if hasattr(self.vit, "blocks") and len(self.vit.blocks) >= n:
            for block in self.vit.blocks[-n:]:
                for param in block.parameters():
                    param.requires_grad = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input image tensor of shape (B, 3, 224, 224)
            
        Returns:
            patch_tokens: (B, 196, embed_dim) - node features for graph
            cls_token: (B, embed_dim) - global visual representation
        """
        B, C, H, W = x.shape
        assert H == self.image_size and W == self.image_size, (
            f"Expected image size {self.image_size}x{self.image_size}, got {H}x{W}"
        )

        # ViT forward_features in timm returns token sequence (B, N_tokens, D)
        # Sequence layout: [CLS_token, patch_1, patch_2, ..., patch_196]
        tokens = self.vit.forward_features(x)

        # In timm ViT, tokens are (B, 197, D)
        cls_token = tokens[:, 0, :]  # (B, D)
        patch_tokens = tokens[:, 1:, :]  # (B, 196, D)

        return patch_tokens, cls_token

    def get_patch_coordinates(self, patch_idx: int) -> Tuple[int, int, int, int]:
        """
        Map a patch node index (0 to 195) to bounding box in original 224x224 image.
        Returns: (y1, y2, x1, x2) in pixel space.
        """
        row = patch_idx // self.grid_size
        col = patch_idx % self.grid_size
        y1 = row * self.patch_size
        y2 = y1 + self.patch_size
        x1 = col * self.patch_size
        x2 = x1 + self.patch_size
        return y1, y2, x1, x2
