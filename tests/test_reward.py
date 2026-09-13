"""
Unit tests for the reward function.
(Subset of test_env.py kept standalone for the tests/test_reward.py slot.)
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rl.reward import (compute_reward, classification_reward,
                            DECIDE_NORMAL, DECIDE_ATTACK, ACQUIRE_GNN)

def test_positive_reward_correct():
    r = compute_reward(y_true=0, action=DECIDE_NORMAL, y_hat=0, lambda_c=0.0)
    assert r == 1.0

def test_negative_reward_incorrect():
    r = compute_reward(y_true=0, action=DECIDE_ATTACK, y_hat=1, lambda_c=0.0)
    assert r == -1.0

def test_cost_subtracted():
    r = compute_reward(y_true=1, action=ACQUIRE_GNN, y_hat=None,
                       lambda_c=1.0, cost_gnn=1.0)
    assert r == -1.0   # 0 - 1.0*1.0

def test_lambda_scaling():
    for lam in [0.1, 0.5, 2.0]:
        r = compute_reward(y_true=0, action=ACQUIRE_GNN, y_hat=None,
                           lambda_c=lam, cost_gnn=1.0)
        assert abs(r - (-lam)) < 1e-9, f"Expected {-lam}, got {r}"

if __name__ == "__main__":
    test_positive_reward_correct()
    test_negative_reward_incorrect()
    test_cost_subtracted()
    test_lambda_scaling()
    print("All reward tests passed.")
