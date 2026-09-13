"""
Custom Gymnasium environment for the FDIA selective-verification MDP.

State   s_t = [P_Tier1, GCS_t, chi2_std_t, b_acq_t]   (4-dim continuous)
Actions {0=DecideNormal, 1=DecideAttack, 2=AcquireGNN, 3=Review}
Episode cap: 2 decision steps.

Episode flow:
  step 0: agent sees s_0 = [P_Tier1, GCS, chi2_std, 0]
           -> picks DecideNormal/DecideAttack  (terminal, reward = r_cls)
           -> or picks AcquireGNN             (b_acq becomes 1, cost applied)
  step 1 (only after AcquireGNN): agent sees s_1 = [..., 1]
           -> must pick DecideNormal/DecideAttack or Review
           -> Review uses the HGAT probability (passed into the env)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.rl.reward import (compute_reward, DECIDE_NORMAL, DECIDE_ATTACK,
                            ACQUIRE_GNN, REVIEW)


class FDIAEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dataset: list[dict], lambda_c: float = 0.5,
                 cost_gnn: float = 1.0, hgat_stub: bool = True):
        """
        dataset : list of dicts with keys 'p_tier1', 'gcs', 'chi2_std', 'label',
                  and optionally 'p_hgat'
        lambda_c: cost weight (tunable hyperparameter)
        cost_gnn: raw cost of invoking HGAT (subtracted when AcquireGNN chosen)
        hgat_stub: if True, HGAT is stubbed as a noisy version of P_Tier1
        """
        super().__init__()
        self.dataset = dataset
        self.lambda_c = lambda_c
        self.cost_gnn = cost_gnn
        self.hgat_stub = hgat_stub

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -10.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 100.0, 10.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._sample = None
        self._step_idx = 0
        self._b_acq = 0
        self._hgat_called = False

    # ── helpers ───────────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        s = self._sample
        return np.array([s["p_tier1"], s["gcs"], s["chi2_std"], float(self._b_acq)],
                        dtype=np.float32)

    def _get_hgat_prob(self) -> float:
        if "p_hgat" in self._sample and not self.hgat_stub:
            return float(self._sample["p_hgat"])
        # stub: add small Gaussian noise to P_Tier1, clipped to [0,1]
        noise = np.random.normal(0, 0.05)
        return float(np.clip(self._sample["p_tier1"] + noise, 0.0, 1.0))

    # ── gym API ───────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._sample = self.dataset[np.random.randint(len(self.dataset))]
        self._step_idx = 0
        self._b_acq = 0
        self._hgat_called = False
        return self._get_obs(), {}

    def step(self, action: int):
        s = self._sample
        y_true = int(s["label"])
        truncated = False
        info = {"hgat_called": False}

        # ── AcquireGNN (only valid at step 0) ────────────────────────────────
        if action == ACQUIRE_GNN:
            if self._step_idx > 0:
                # penalise invalid re-acquisition
                reward = -1.0
                return self._get_obs(), reward, True, truncated, info

            self._b_acq = 1
            self._hgat_called = True
            info["hgat_called"] = True
            reward = compute_reward(y_true, ACQUIRE_GNN, None,
                                    self.lambda_c, self.cost_gnn)
            self._step_idx += 1
            return self._get_obs(), reward, False, truncated, info  # not terminal

        # ── Review (only valid at step 1 after AcquireGNN) ───────────────────
        if action == REVIEW:
            p_hgat = self._get_hgat_prob()
            y_hat = int(p_hgat >= 0.5)
            reward = compute_reward(y_true, DECIDE_ATTACK if y_hat else DECIDE_NORMAL,
                                    y_hat, self.lambda_c, 0.0)
            self._step_idx += 1
            return self._get_obs(), reward, True, truncated, info

        # ── Direct decisions: DecideNormal / DecideAttack ─────────────────────
        if action in (DECIDE_NORMAL, DECIDE_ATTACK):
            y_hat = 1 if action == DECIDE_ATTACK else 0
            reward = compute_reward(y_true, action, y_hat, self.lambda_c, 0.0)
            self._step_idx += 1
            return self._get_obs(), reward, True, truncated, info

        # fallback
        return self._get_obs(), -1.0, True, truncated, info
