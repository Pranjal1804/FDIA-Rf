"""Unit tests for FDIAEnv and the reward function."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rl.env import FDIAEnv
from src.rl.reward import (compute_reward, classification_reward,
                            DECIDE_NORMAL, DECIDE_ATTACK, ACQUIRE_GNN, REVIEW)


def _make_dataset(n=50, attack_frac=0.4):
    rng = np.random.default_rng(0)
    data = []
    for _ in range(n):
        label = int(rng.random() < attack_frac)
        data.append({
            "p_tier1":  float(rng.random()),
            "gcs":      float(rng.random() * 2),
            "chi2_std": float(rng.normal(0, 1)),
            "label":    label,
        })
    return data


def test_reward_correct_classify():
    r = compute_reward(y_true=1, action=DECIDE_ATTACK, y_hat=1, lambda_c=0.5)
    assert r == 1.0, f"Expected 1.0, got {r}"


def test_reward_wrong_classify():
    r = compute_reward(y_true=1, action=DECIDE_NORMAL, y_hat=0, lambda_c=0.5)
    assert r == -1.0, f"Expected -1.0, got {r}"


def test_reward_acquire_gnn_cost():
    r = compute_reward(y_true=1, action=ACQUIRE_GNN, y_hat=None,
                       lambda_c=0.5, cost_gnn=1.0)
    assert r == -0.5, f"Expected -0.5 (= 0 - 0.5*1.0), got {r}"


def test_env_reset():
    ds = _make_dataset()
    env = FDIAEnv(ds, lambda_c=0.5)
    obs, info = env.reset()
    assert obs.shape == (4,)
    assert obs[3] == 0.0, "b_acq should be 0 after reset"


def test_env_direct_decision_terminates():
    ds = _make_dataset()
    env = FDIAEnv(ds, lambda_c=0.5)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(DECIDE_ATTACK)
    assert terminated, "Direct decision should terminate the episode"
    assert reward in (-1.0, 1.0), f"Unexpected reward {reward}"


def test_env_two_step_episode():
    """AcquireGNN at step 0 -> Review at step 1 -> terminal."""
    ds = _make_dataset()
    env = FDIAEnv(ds, lambda_c=0.5)
    obs0, _ = env.reset()
    obs1, r0, term0, _, info0 = env.step(ACQUIRE_GNN)
    assert not term0, "AcquireGNN should NOT terminate"
    assert info0["hgat_called"], "hgat_called flag should be True"
    assert obs1[3] == 1.0, "b_acq should be 1 after AcquireGNN"
    obs2, r1, term1, _, info1 = env.step(REVIEW)
    assert term1, "Review should terminate episode"


def test_env_step_b_acq_flag():
    ds = _make_dataset()
    env = FDIAEnv(ds, lambda_c=0.3)
    env.reset()
    obs, _, term, _, _ = env.step(ACQUIRE_GNN)
    assert obs[3] == 1.0


if __name__ == "__main__":
    test_reward_correct_classify()
    test_reward_wrong_classify()
    test_reward_acquire_gnn_cost()
    test_env_reset()
    test_env_direct_decision_terminates()
    test_env_two_step_episode()
    test_env_step_b_acq_flag()
    print("All Phase-4 env/reward tests passed.")
