import numpy as np

class NumpyA2C:
    """
    Pure NumPy implementation of the Advantage Actor-Critic (A2C) algorithm.
    Used as an immediate fallback since PyTorch downloads timed out continuously.
    """
    def __init__(self, state_dim, action_dim=4, hidden_dim=64, lr=5e-3):
        # Actor
        self.Wa1 = np.random.randn(state_dim, hidden_dim) * np.sqrt(2/state_dim)
        self.ba1 = np.zeros(hidden_dim)
        self.Wa2 = np.random.randn(hidden_dim, action_dim) * np.sqrt(2/hidden_dim)
        self.ba2 = np.zeros(action_dim)
        
        # Critic
        self.Wc1 = np.random.randn(state_dim, hidden_dim) * np.sqrt(2/state_dim)
        self.bc1 = np.zeros(hidden_dim)
        self.Wc2 = np.random.randn(hidden_dim, 1) * np.sqrt(2/hidden_dim)
        self.bc2 = np.zeros(1)
        
        self.lr = lr

    def _relu(self, x): return np.maximum(0, x)
    def _drelu(self, h): return (h > 0).astype(float)
    def _softmax(self, x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    def forward_actor(self, s):
        h = self._relu(s @ self.Wa1 + self.ba1)
        logits = h @ self.Wa2 + self.ba2
        probs = self._softmax(logits)
        return probs, h

    def forward_critic(self, s):
        h = self._relu(s @ self.Wc1 + self.bc1)
        v = h @ self.Wc2 + self.bc2
        return v, h

    def select_action(self, s):
        probs, _ = self.forward_actor(s)
        return np.random.choice(len(probs), p=probs)

    def update(self, states, actions, returns):
        S = np.array(states)
        A = np.array(actions)
        G = np.array(returns).reshape(-1, 1)

        probs, ha = self.forward_actor(S)
        V, hc = self.forward_critic(S)

        # Advantage A_t = G_t - V(s_t)
        adv = G - V

        # Critic Gradients (MSE)
        d_V = -2 * adv / len(S)
        dWc2 = hc.T @ d_V; dbc2 = d_V.sum(0)
        dhc = d_V @ self.Wc2.T * self._drelu(hc)
        dWc1 = S.T @ dhc; dbc1 = dhc.sum(0)

        # Actor Gradients (Policy Gradient: dlog(pi) * A)
        d_logits = probs.copy()
        d_logits[np.arange(len(S)), A] -= 1
        d_logits = d_logits * adv / len(S)
        dWa2 = ha.T @ d_logits; dba2 = d_logits.sum(0)
        dha = d_logits @ self.Wa2.T * self._drelu(ha)
        dWa1 = S.T @ dha; dba1 = dha.sum(0)

        # Apply gradients
        for w, dw in zip([self.Wa1, self.ba1, self.Wa2, self.ba2, self.Wc1, self.bc1, self.Wc2, self.bc2],
                         [dWa1, dba1, dWa2, dba2, dWc1, dbc1, dWc2, dbc2]):
            dw_cl = np.clip(dw, -1.0, 1.0)
            w -= self.lr * dw_cl
