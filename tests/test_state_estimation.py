import numpy as np
from src.simulation.state_estimation import DCStateEstimator
from src.simulation.fdia_injection import generate_fdia

def test_state_estimation():
    n_bus = 4
    line_or = [0, 0, 1, 2]
    line_ex = [1, 2, 3, 3]
    reactances = [0.1, 0.2, 0.1, 0.2]
    
    se = DCStateEstimator(n_bus, line_or, line_ex, reactances)
    assert se.H.shape == (8, 3)  # (4 lines + 4 buses, 4-1 states)
    
    # Test perfect measurements
    true_theta = np.array([0.1, -0.05, 0.02])
    z_true = se.H @ true_theta
    
    x_hat, z_hat = se.estimate(z_true)
    assert np.allclose(x_hat, true_theta, atol=1e-4)
    assert np.allclose(z_hat, z_true, atol=1e-4)

def test_fdia_injection():
    n_bus = 4
    line_or = [0, 0, 1, 2]
    line_ex = [1, 2, 3, 3]
    reactances = [0.1, 0.2, 0.1, 0.2]
    se = DCStateEstimator(n_bus, line_or, line_ex, reactances)
    
    a, c = generate_fdia(se.H, attack_strength=1.0)
    assert a.shape == (8,)
    assert c.shape == (3,)
    
    # Verify a = Hc
    assert np.allclose(a, se.H @ c)
