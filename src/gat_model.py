"""
Graph Attention Network (GAT) Architecture for Strawberry AI.

Operates on image patch graphs to learn spatial and semantic lesion relationships:
- Multi-head attention allows attending to diverse symptom characteristics
  (e.g., halo boundaries, sporulation centers, necroses).
- Graph pooling (global mean + max) aggregates patch representations into a whole-strawberry graph embedding.
- Exposes attention weights for edge-level and node-level explainability.
"""

from typing import Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class PurePyTorchGATLayer(nn.Module):
    """
    Self-contained pure-PyTorch Multi-Head Graph Attention Layer.
    Guarantees compatibility even in environments where compiled PyG operators are absent.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 4,
        dropout: float = 0.2,
        concat: bool = True,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope

        self.lin = nn.Linear(in_features, heads * out_features, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_features))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_features))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.
        Args:
            x: (N, in_features)
            edge_index: (2, E)
        """
        N = x.size(0)
        h = self.lin(x).view(N, self.heads, self.out_features)  # (N, H, F)

        src, dst = edge_index[0], edge_index[1]

        alpha_src = (h[src] * self.att_src).sum(dim=-1)  # (E, H)
        alpha_dst = (h[dst] * self.att_dst).sum(dim=-1)  # (E, H)
        alpha = self.leaky_relu(alpha_src + alpha_dst)  # (E, H)

        # Softmax over incoming edges using scatter_add or indexed exp sum
        alpha_exp = torch.exp(alpha - alpha.max(dim=0, keepdim=True)[0])  # numerical stability
        denom = torch.zeros(N, self.heads, device=x.device).scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.heads), alpha_exp)
        denom = denom[dst] + 1e-16
        alpha_norm = alpha_exp / denom  # (E, H)
        alpha_norm = self.dropout(alpha_norm)

        # Message passing: out_i = sum_{j in N(i)} alpha_ij * h_j
        weighted_h = h[src] * alpha_norm.unsqueeze(-1)  # (E, H, F)
        out = torch.zeros(N, self.heads, self.out_features, device=x.device)
        out.scatter_add_(0, dst.view(-1, 1, 1).expand(-1, self.heads, self.out_features), weighted_h)

        if self.concat:
            out = out.view(N, self.heads * self.out_features)
        else:
            out = out.mean(dim=1)

        att_tuple = (edge_index, alpha_norm.mean(dim=-1)) if return_attention_weights else None
        return out, att_tuple


class StrawberryGAT(nn.Module):
    """
    Two-layer Graph Attention Network for strawberry patch graphs.
    Includes residual connections, LayerNorm, and multi-head attention.
    """

    def __init__(
        self,
        in_channels: int = 192,
        hidden_channels: int = 128,
        out_channels: int = 64,
        heads: int = 4,
        dropout: float = 0.2,
        use_pyg: bool = True,
    ):
        super().__init__()
        self.use_pyg = use_pyg and HAS_PYG
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout_rate = dropout

        if self.use_pyg:
            self.gat1 = GATConv(
                in_channels=in_channels,
                out_channels=hidden_channels,
                heads=heads,
                concat=True,
                dropout=dropout,
            )
            self.gat2 = GATConv(
                in_channels=hidden_channels * heads,
                out_channels=out_channels,
                heads=heads,
                concat=False,
                dropout=dropout,
            )
        else:
            self.gat1 = PurePyTorchGATLayer(
                in_features=in_channels,
                out_features=hidden_channels,
                heads=heads,
                concat=True,
                dropout=dropout,
            )
            self.gat2 = PurePyTorchGATLayer(
                in_features=hidden_channels * heads,
                out_features=out_channels,
                heads=heads,
                concat=False,
                dropout=dropout,
            )

        self.norm1 = nn.LayerNorm(hidden_channels * heads)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.last_attention_weights = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass over batched patch graph.
        
        Args:
            x: (Total_nodes, in_channels)
            edge_index: (2, Total_edges)
            batch: (Total_nodes,) indicating sample index in batch
            return_attention: whether to capture attention coefficients for XAI
            
        Returns:
            graph_rep: (B, 2 * out_channels) pooled graph-level feature vector
            node_rep: (Total_nodes, out_channels) refined patch representations
        """
        # Layer 1
        if self.use_pyg:
            h1 = self.gat1(x, edge_index)
        else:
            h1, _ = self.gat1(x, edge_index)

        h1 = self.norm1(h1)
        h1 = F.elu(h1)
        h1 = self.dropout(h1)

        # Layer 2
        if self.use_pyg:
            if return_attention:
                h2, (edge_i, alpha) = self.gat1(x, edge_index, return_attention_weights=True)
                self.last_attention_weights = (edge_i, alpha)
            h2 = self.gat2(h1, edge_index)
        else:
            h2, att = self.gat2(h1, edge_index, return_attention_weights=return_attention)
            if att is not None:
                self.last_attention_weights = att

        h2 = self.norm2(h2)
        node_rep = h2  # (Total_nodes, out_channels)

        # Graph Readout / Pooling (Mean + Max)
        if self.use_pyg:
            mean_pool = global_mean_pool(node_rep, batch)
            max_pool = global_max_pool(node_rep, batch)
        else:
            B = int(batch.max().item() + 1) if batch.numel() > 0 else 1
            mean_pool = torch.zeros(B, node_rep.size(-1), device=node_rep.device)
            max_pool = torch.zeros(B, node_rep.size(-1), device=node_rep.device)
            for b in range(B):
                mask = (batch == b)
                if mask.any():
                    mean_pool[b] = node_rep[mask].mean(dim=0)
                    max_pool[b] = node_rep[mask].max(dim=0)[0]

        graph_rep = torch.cat([mean_pool, max_pool], dim=-1)  # (B, 2 * out_channels)
        return graph_rep, node_rep
