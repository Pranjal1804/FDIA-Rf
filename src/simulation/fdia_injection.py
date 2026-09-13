import numpy as np

def generate_fdia(H, attack_strength=0.1, n_targets=None):
    """
    Generates an attack vector a = H @ c.
    c is the hidden state deviation vector.
    """
    n_states = H.shape[1]
    c = np.random.randn(n_states) * attack_strength
    
    if n_targets is not None and n_targets < n_states:
        mask = np.zeros(n_states)
        targets = np.random.choice(n_states, n_targets, replace=False)
        mask[targets] = 1.0
        c *= mask
        
    a = H @ c
    return a, c
