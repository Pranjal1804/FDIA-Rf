import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.indicators.gcs import build_adjacency, compute_gcs
from src.indicators.chi_square import compute_chi2, Chi2Standardiser
from src.simulation.state_estimation import DCStateEstimator
from src.simulation.grid_env import GridSimulation


def test_gcs_low_on_smooth():
    """GCS should be low when node features are already smoothed by neighbours."""
    n_bus = 4
    line_or = [0, 1, 2]
    line_ex = [1, 2, 3]
    A = build_adjacency(n_bus, line_or, line_ex)
    # perfectly smooth: all nodes identical
    X = np.ones((n_bus, 3))
    gcs = compute_gcs(X, A)
    assert gcs < 1e-6, f"Expected near-zero GCS for uniform features, got {gcs}"


def test_gcs_rises_with_perturbation():
    n_bus = 4
    line_or = [0, 1, 2]
    line_ex = [1, 2, 3]
    A = build_adjacency(n_bus, line_or, line_ex)
    X_clean = np.ones((n_bus, 3)) * 10.0
    X_perturbed = X_clean.copy()
    X_perturbed[0] += 50.0  # perturb one node significantly
    gcs_clean = compute_gcs(X_clean, A)
    gcs_perturbed = compute_gcs(X_perturbed, A)
    assert gcs_perturbed > gcs_clean, \
        f"Expected GCS to rise after perturbation: {gcs_clean:.4f} -> {gcs_perturbed:.4f}"


def test_chi2_zero_on_perfect_estimate():
    sim = GridSimulation()
    se = DCStateEstimator(sim.n_bus, sim.line_or, sim.line_ex, sim.reactances)
    theta_true = np.random.randn(se.n_bus - 1) * 0.1
    z = se.H @ theta_true
    x_hat, _ = se.estimate(z)
    chi2 = compute_chi2(z, se.H, x_hat)
    assert chi2 < 1e-4, f"Chi2 should be ~0 for perfect estimate, got {chi2:.6f}"


def test_chi2_standardiser():
    vals = np.random.chisquare(df=14, size=500)
    std = Chi2Standardiser()
    std.fit(vals)
    z = std.transform(float(vals.mean()))
    assert abs(z) < 0.5, f"z-score of mean should be near 0, got {z:.4f}"


if __name__ == "__main__":
    test_gcs_low_on_smooth()
    test_gcs_rises_with_perturbation()
    test_chi2_zero_on_perfect_estimate()
    test_chi2_standardiser()
    print("All Phase-3 indicator tests passed.")
