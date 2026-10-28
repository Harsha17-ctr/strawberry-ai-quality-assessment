"""
Unit tests for Graph Construction module (PatchGraphBuilder).
"""

import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph_builder import PatchGraphBuilder


def test_spatial_edges():
    """Verify spatial 8-connected grid edges for 14x14 patches."""
    builder = PatchGraphBuilder(grid_size=14, include_self_loops=True)
    edges = builder._static_spatial_edge_index
    assert edges.shape[0] == 2
    # Check that node indices are within [0, 195]
    assert edges.min() == 0
    assert edges.max() == 195
    # For a 14x14 grid, interior nodes have 8 neighbors, corners have 3, edges have 5, + 1 self loop each
    assert edges.shape[1] > 196 * 4


def test_batch_graph_construction():
    """Verify building batched graph tensor for batch size B=3."""
    builder = PatchGraphBuilder(grid_size=14, connect_by="hybrid")
    dummy_patches = torch.randn(3, 196, 192)

    x_batch, edge_index_batch, batch_index = builder.build_batch_graph(dummy_patches)

    assert x_batch.shape == (3 * 196, 192)
    assert edge_index_batch.shape[0] == 2
    assert batch_index.shape == (3 * 196,)
    assert batch_index.max() == 2
    assert edge_index_batch.max() < 3 * 196


if __name__ == "__main__":
    test_spatial_edges()
    test_batch_graph_construction()
    print("[OK] All graph builder unit tests passed successfully!")
