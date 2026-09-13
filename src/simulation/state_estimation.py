import numpy as np

class DCStateEstimator:
    def __init__(self, n_bus, line_or, line_ex, reactances):
        """
        Builds the measurement matrix H for DC state estimation.
        x = theta (length n_bus), with theta[0] = 0 (slack).
        z = [P_flow; P_inj]
        """
        self.n_bus = n_bus
        self.n_line = len(line_or)
        
        # If reactances are zeros (e.g. ideal switch), replace with small number
        susceptances = 1.0 / np.maximum(reactances, 1e-4)
        
        H_flow = np.zeros((self.n_line, n_bus))
        for l in range(self.n_line):
            i, j = int(line_or[l]), int(line_ex[l])
            b = susceptances[l]
            H_flow[l, i] = b
            H_flow[l, j] = -b
            
        H_inj = np.zeros((n_bus, n_bus))
        for i in range(n_bus):
            for l in range(self.n_line):
                if line_or[l] == i:
                    H_inj[i] += H_flow[l]
                elif line_ex[l] == i:
                    H_inj[i] -= H_flow[l]
                    
        self.H_full = np.concatenate([H_flow, H_inj], axis=0)
        self.H = self.H_full[:, 1:] # remove slack bus column
        
        # Precompute WLS matrix G = (H^T H)^-1 H^T
        HTH = self.H.T @ self.H
        # Add small regularization to avoid singular matrix if topology is disconnected
        HTH += np.eye(HTH.shape[0]) * 1e-6
        self.G = np.linalg.inv(HTH) @ self.H.T

    def estimate(self, z):
        """
        z: measurements [P_flow; P_inj]
        Returns estimated state x_hat and estimated measurements z_hat.
        """
        x_hat = self.G @ z
        z_hat = self.H @ x_hat
        return x_hat, z_hat
