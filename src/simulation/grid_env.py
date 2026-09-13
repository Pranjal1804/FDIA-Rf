import numpy as np

class GridSimulation:
    def __init__(self, case_name="rte_case14_realistic"):
        self.n_bus = 14
        # IEEE 14-bus lines (origin, extremum, reactance)
        lines = [
            (0, 1, 0.05), (0, 4, 0.22), (1, 2, 0.19), (1, 3, 0.17),
            (1, 4, 0.17), (2, 3, 0.17), (3, 4, 0.04), (3, 6, 0.20),
            (3, 8, 0.55), (4, 5, 0.25), (5, 10, 0.19), (5, 11, 0.25),
            (5, 12, 0.13), (6, 7, 0.17), (6, 8, 0.11), (8, 9, 0.08),
            (8, 13, 0.34), (9, 10, 0.19), (11, 12, 0.19), (12, 13, 0.34)
        ]
        self.n_line = len(lines)
        self.line_or = np.array([u for u,v,x in lines])
        self.line_ex = np.array([v for u,v,x in lines])
        self.reactances = np.array([x for u,v,x in lines])
        
        self.base_loads = np.random.uniform(10, 50, self.n_bus)
        self.base_loads[0] = -np.sum(self.base_loads[1:]) 

    def step(self):
        loads = self.base_loads * np.random.normal(1.0, 0.05, self.n_bus)
        loads[0] = -np.sum(loads[1:]) 
        
        sus = 1.0 / self.reactances
        H_flow = np.zeros((self.n_line, self.n_bus))
        for l in range(self.n_line):
            i, j = self.line_or[l], self.line_ex[l]
            H_flow[l, i] = sus[l]
            H_flow[l, j] = -sus[l]
            
        H_inj = np.zeros((self.n_bus, self.n_bus))
        for i in range(self.n_bus):
            for l in range(self.n_line):
                if self.line_or[l] == i:
                    H_inj[i] += H_flow[l]
                elif self.line_ex[l] == i:
                    H_inj[i] -= H_flow[l]
                    
        # DC Power Flow solver
        # Add tiny damping for numerical stability
        H_dc = H_inj[1:, 1:] + np.eye(self.n_bus - 1)*1e-6
        P_inj_reduced = loads[1:]
        theta_reduced = np.linalg.solve(H_dc, P_inj_reduced)
        
        theta = np.zeros(self.n_bus)
        theta[1:] = theta_reduced
        
        p_flow = H_flow @ theta
        p_inj = loads
        
        # simulate unobserved nonlinearity
        p_flow += np.random.randn(self.n_line) * 0.1 
        
        z_true = np.concatenate([p_flow, p_inj])
        return z_true
