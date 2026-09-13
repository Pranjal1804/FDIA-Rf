"""
GCS — Graph Consistency Signature
  O_t = D^{-1} A X_t     (degree-normalised neighbourhood averaging)
  GCS_t = (1/|V|) * ||X_t - O_t||_F^2
"""
import numpy as np


def build_adjacency(n_bus, line_or, line_ex, include_self_loops=False):
    """Build an unweighted adjacency matrix for the grid."""
    A = np.zeros((n_bus, n_bus))
    for u, v in zip(line_or, line_ex):
        A[u, v] = 1.0
        A[v, u] = 1.0
    if include_self_loops:
        A += np.eye(n_bus)
    return A


def compute_gcs(X_nodes: np.ndarray, A: np.ndarray) -> float:
    """
    X_nodes : (n_bus, n_features) — per-bus feature matrix
    A       : (n_bus, n_bus)      — adjacency matrix
    Returns scalar GCS value.
    """
    n_bus = A.shape[0]
    deg = A.sum(axis=1).clip(min=1.0)           # degree vector, avoid /0
    D_inv = np.diag(1.0 / deg)
    O = D_inv @ A @ X_nodes                     # (n_bus, n_features)
    diff = X_nodes - O
    gcs = np.sum(diff ** 2) / n_bus
    return float(gcs)
