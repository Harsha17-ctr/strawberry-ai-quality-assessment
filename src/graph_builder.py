"""
Graph Construction Module for Strawberry AI.

Converts 2D Vision Transformer spatial patch tokens into an attributed graph:
- Nodes: 196 image patches (from 14x14 grid) with 192-dimensional ViT embedding features.
- Edges:
  1. Spatial 8-connectivity grid adjacency (captures local tissue context & lesion spreading).
  2. Semantic feature cosine similarity (captures non-local dispersed disease symptoms).
  3. Self-loops (ensures node self-information is preserved in message passing).
"""

from typing import Tuple, List, Optional
import torch
import torch.nn.functional as F


class PatchGraphBuilder:
    """
    Constructs edge connectivity matrices for patch sequences.
    Grid: 14x14 = 196 patches.
    """

    def __init__(
        self,
        grid_size: int = 14,
        connect_by: str = "hybrid",
        similarity_threshold: float = 0.5,
        top_k_similarity: int = 4,
        include_self_loops: bool = True,
    ):
        self.grid_size = grid_size
        self.num_nodes = grid_size * grid_size  # 196
        self.connect_by = connect_by
        self.similarity_threshold = similarity_threshold
        self.top_k_similarity = top_k_similarity
        self.include_self_loops = include_self_loops

        # Precompute static spatial 8-neighborhood edges for the grid
        self._static_spatial_edge_index = self._build_spatial_8_edges()

    def _build_spatial_8_edges(self) -> torch.Tensor:
        """
        Build 8-connected grid neighborhood edges for (grid_size x grid_size) patches.
        Returns: edge_index of shape (2, num_spatial_edges).
        """
        sources: List[int] = []
        targets: List[int] = []

        # 8 directions in grid: N, S, E, W, NE, NW, SE, SW
        offsets = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                curr_idx = r * self.grid_size + c
                for dr, dc in offsets:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        neighbor_idx = nr * self.grid_size + nc
                        sources.append(curr_idx)
                        targets.append(neighbor_idx)

        # Self-loops if enabled
        if self.include_self_loops:
            for idx in range(self.num_nodes):
                sources.append(idx)
                targets.append(idx)

        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        return edge_index

    def build_similarity_edges(self, patch_features: torch.Tensor) -> torch.Tensor:
        """
        Compute dynamic semantic edges based on cosine similarity of ViT patch embeddings.
        Args:
            patch_features: (num_nodes, embed_dim)
        Returns:
            edge_index: (2, num_similarity_edges)
        """
        norm_features = F.normalize(patch_features, p=2, dim=-1)  # (196, D)
        sim_matrix = torch.mm(norm_features, norm_features.t())  # (196, 196)

        # Zero out diagonal to avoid duplicate self-loops
        sim_matrix.fill_diagonal_(-1.0)

        # Connect top-K most similar patches
        topk_vals, topk_indices = torch.topk(sim_matrix, k=self.top_k_similarity, dim=-1)

        sources = []
        targets = []
        for i in range(self.num_nodes):
            for rank in range(self.top_k_similarity):
                if topk_vals[i, rank] >= self.similarity_threshold:
                    sources.append(i)
                    targets.append(topk_indices[i, rank].item())

        if len(sources) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=patch_features.device)

        return torch.tensor([sources, targets], dtype=torch.long, device=patch_features.device)

    def build_batch_graph(
        self, patch_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build batched graph from a batch of patch embeddings.
        
        Args:
            patch_tokens: (B, 196, embed_dim)
            
        Returns:
            x_batch: (B * 196, embed_dim) - concatenated node features
            edge_index_batch: (2, Total_edges) - shifted edge indices for batch
            batch_index: (B * 196,) - graph membership indicator
        """
        B, N, D = patch_tokens.shape
        device = patch_tokens.device

        static_edges = self._static_spatial_edge_index.to(device)

        all_edges = []
        batch_indices = []

        for b in range(B):
            offset = b * N
            features_b = patch_tokens[b]

            if self.connect_by == "spatial":
                b_edges = static_edges + offset
            elif self.connect_by == "similarity":
                sim_edges = self.build_similarity_edges(features_b)
                if self.include_self_loops:
                    self_loops = torch.arange(N, device=device).repeat(2, 1)
                    sim_edges = torch.cat([sim_edges, self_loops], dim=1)
                b_edges = sim_edges + offset
            else:  # 'hybrid'
                sim_edges = self.build_similarity_edges(features_b)
                combined = torch.cat([static_edges, sim_edges], dim=1)
                # Remove duplicate edges
                combined = torch.unique(combined, dim=1)
                b_edges = combined + offset

            all_edges.append(b_edges)
            batch_indices.append(torch.full((N,), b, dtype=torch.long, device=device))

        edge_index_batch = torch.cat(all_edges, dim=1)
        x_batch = patch_tokens.contiguous().view(B * N, D)
        batch_index = torch.cat(batch_indices, dim=0)

        return x_batch, edge_index_batch, batch_index
