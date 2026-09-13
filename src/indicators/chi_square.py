"""
Standardised chi-square residual
  chi2(z) = (z - H x_hat)^T W (z - H x_hat)
  Then z-score standardise across the training distribution.
"""
import numpy as np


def compute_chi2(z: np.ndarray, H: np.ndarray, x_hat: np.ndarray,
                 W: np.ndarray | None = None) -> float:
    """
    z     : measurement vector (m,)
    H     : measurement matrix  (m, n)
    x_hat : estimated state     (n,)
    W     : weight matrix (m, m); defaults to identity
    """
    residual = z - H @ x_hat
    if W is None:
        W = np.eye(len(z))
    return float(residual @ W @ residual)


class Chi2Standardiser:
    """Fits mean/std on a set of chi2 values and maps new values to z-scores."""
    def __init__(self):
        self.mu = 0.0
        self.sigma = 1.0

    def fit(self, chi2_values: np.ndarray):
        self.mu = float(np.mean(chi2_values))
        self.sigma = float(np.std(chi2_values) + 1e-8)

    def transform(self, chi2_val: float) -> float:
        return (chi2_val - self.mu) / self.sigma
