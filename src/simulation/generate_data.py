import numpy as np
import json
import os
from src.simulation.grid_env import GridSimulation
from src.simulation.state_estimation import DCStateEstimator
from src.simulation.fdia_injection import generate_fdia
from src.config import load_config

def main():
    config = load_config()
    noise_level = config['simulation']['noise_level']
    attack_fraction = config['simulation']['attack_fraction']
    attack_strength = config['simulation']['attack_strength']
    
    sim = GridSimulation()
    se = DCStateEstimator(sim.n_bus, sim.line_or, sim.line_ex, sim.reactances)
    
    n_samples = 1000
    
    data = [] 
    
    for i in range(n_samples):
        z_true = sim.step(time_idx=i)
        
        # add measurement noise (Gaussian proportional to magnitude)
        z = z_true + np.random.randn(len(z_true)) * noise_level * np.abs(z_true + 1e-4)
        
        attacked = False
        attacked_buses = []
        if np.random.rand() < attack_fraction:
            # Generate unobservable DC FDIA, explicitly targeting 2-3 random buses
            a, c = generate_fdia(se.H, attack_strength=attack_strength, n_targets=np.random.randint(2, 4))
            z += a
            attacked = True
            attacked_buses = c.nonzero()[0].tolist()
            
        # compute GCS ingredients (used later) - saving true z as base for simplicity
        data.append({
            "z": z.tolist(),
            "label": int(attacked),
            "attacked_buses": attacked_buses
        })
        
    os.makedirs("data/simulated", exist_ok=True)
    with open("data/simulated/case14_data.json", "w") as f:
        json.dump(data, f)
        
    n_attack = sum(d["label"] for d in data)
    print(f"Generated {n_samples} samples. {n_attack} attacked.")
    
if __name__ == "__main__":
    main()
