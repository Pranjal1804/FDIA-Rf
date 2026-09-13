import json
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.rl.env import FDIAEnv
from src.rl.numpy_a2c import NumpyA2C

def load_and_augment_dataset():
    with open("data/simulated/case14_data.json") as f:
        dataset = json.load(f)

    # In a full pipeline, we would run Tier 1 models to extract P_Tier1, GCS and Chi2 here.
    # To demonstrate RL capability decoupled from inference passes, we use structured synthetic features mapping to the labels.
    for d in dataset:
        target = float(d["label"])
        # Give P_tier1 some predictive power with noise
        d["p_tier1"] = np.clip(target + np.random.normal(0, 0.2), 0.0, 1.0)
        # GCS is loosely correlated
        d["gcs"] = np.abs(np.random.normal(target, 0.5)) * 10
        d["chi2_std"] = np.random.normal(target * 2, 1.0)
    return dataset

def main():
    print("Preparing dataset and features...", flush=True)
    dataset = load_and_augment_dataset()

    lambdas = [0.1, 0.5, 2.0]
    
    for lam in lambdas:
        print(f"\n--- Training A2C Agent at lambda_c = {lam} ---", flush=True)
        env = FDIAEnv(dataset, lambda_c=lam)
        agent = NumpyA2C(state_dim=4, action_dim=4, hidden_dim=64, lr=1e-3)
        
        episodes = 5000
        rewards = []
        hgat_calls = 0
        
        for ep in range(episodes):
            s, _ = env.reset()
            ep_states, ep_actions, ep_rewards = [], [], []
            done = False
            while not done:
                a = agent.select_action(s)
                s_next, r, done, _, info = env.step(a)
                ep_states.append(s)
                ep_actions.append(a)
                ep_rewards.append(r)
                s = s_next
                if info.get("hgat_called"):
                    hgat_calls += 1
            
            # Simple Monte-Carlo returns
            returns = []
            G = 0
            for r in reversed(ep_rewards):
                G = r + G
                returns.insert(0, G)
                
            agent.update(ep_states, ep_actions, returns)
            rewards.append(sum(ep_rewards))
            
            if (ep + 1) % 1000 == 0:
                print(f"  Ep {ep+1:4d} | Mean Reward (last 500): {np.mean(rewards[-500:]):.4f}")
        
        hgat_rate = hgat_calls / episodes
        print(f"Final HGAT Verification Invocation Rate: {hgat_rate:.1%}", flush=True)

if __name__ == "__main__":
    main()
