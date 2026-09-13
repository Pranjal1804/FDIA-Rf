"""
Reward function for the FDIA selective-verification RL agent.

r_t = r_cls(y_t, y_hat_t) - lambda_c * c(a_t)

r_cls:
  +1  if correct classification (Normal→Normal or Attack→Attack)
  -1  if incorrect classification
   0  if action was AcquireGNN (no direct decision yet)

c(a_t):
  cost_gnn  if action == AcquireGNN
  0         otherwise

Actions (int):
  0: DecideNormal
  1: DecideAttack
  2: AcquireGNN
  3: Review  (only valid at step 2, uses HGAT probability)
"""

# action index constants
DECIDE_NORMAL  = 0
DECIDE_ATTACK  = 1
ACQUIRE_GNN    = 2
REVIEW         = 3


def classification_reward(y_true: int, y_hat: int) -> float:
    """Returns +1 for correct binary classification, -1 for wrong."""
    return 1.0 if y_true == y_hat else -1.0


def action_cost(action: int, cost_gnn: float = 1.0) -> float:
    """Returns the cost of the chosen action (non-zero only for AcquireGNN)."""
    return cost_gnn if action == ACQUIRE_GNN else 0.0


def compute_reward(y_true: int, action: int, y_hat: int | None,
                   lambda_c: float, cost_gnn: float = 1.0) -> float:
    """
    Full reward function:  r = r_cls - lambda_c * c(a)
    y_hat is None when the action is AcquireGNN (deferred decision).
    """
    if action == ACQUIRE_GNN:
        r_cls = 0.0          # no classification yet
    else:
        assert y_hat is not None
        r_cls = classification_reward(y_true, y_hat)
    cost = action_cost(action, cost_gnn)
    return r_cls - lambda_c * cost
