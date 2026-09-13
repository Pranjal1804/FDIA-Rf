# Safe Reinforcement Learning for Smart Grid Control Under False Data Injection Attacks (FDIA)
**Final Year Project Prototype — Real-World Implementation**

<p align="center">
  <i>A Two-Tier Selective Verification Framework combining lightweight machine learning, structural graph heuristics, and A2C reinforcement learning to detect cyber-physical attacks on critical grid infrastructure.</i>
</p>

---

## 1. Project Abstract & The Problem
State Estimators in modern Supervisory Control and Data Acquisition (SCADA) systems are vulnerable to **False Data Injection Attacks (FDIA)**. In an FDIA, an attacker mathematically crafts a compromised measurement vector (`a = Hc`) that completely evades traditional physical safeguards (like the Chi-Square residual test), allowing them to manipulate grid markets or artificially induce power outages.

While sophisticated deep-learning models like Graph Neural Networks (GNNs) can detect these attacks by looking at the deep topology of the grid, **running a heavy GNN every 5 seconds on every single grid measurement cycle is computationally prohibitive** for edge industrial controllers.

**The Solution:** This project builds a *Two-Tier Selective Verification Framework*. Every cycle is screened by an ultra-fast "Tier-1" ensemble of lightweight machine learning models. A **Reinforcement Learning (RL)** agent then analyzes the Tier-1 confidence and actively decides whether to classify the grid state immediately, or—if the sample is highly ambiguous—spend computational resources to invoke the heavy "Tier-2" Hypergraph Attention Network for a deep structural verdict.

---

## 2. Real-World Dataset Generation (Phase 1)
Instead of relying on idealized datasets, this prototype strictly models realistic physical grid conditions.
Using the **IEEE 14-bus test system**, we simulate genuine grid dynamics:
- **Diurnal Load Curves:** Grid loads cycle realistically over a 24-hour period (peak loads modeled via superimposed sinusoidal functions over base generation).
- **Physical DC Power Flow:** We solve the grid admittance matrices (`GridSimulation.step()`) strictly according to Kirchhoff's physical laws.
- **FDIA Injection targeting:** The attacker optimally masks their deviations onto 2-3 specific buses, generating a totally unobservable residual attack vector. 

---

## 3. Tier-1: High-Speed Screening & Indicators (Phases 2 & 3)
The goal of Tier-1 is to extract fast, statistical anomalies without running heavy convolutions across the grid.

### 3.1 The Fast Machine Learning Cluster
When a load cycle arrives, it is instantly evaluated by three models:
1. **Isolation Forest:** Catches statistical outliers (fast decision trees bounding normal data).
2. **PCA Autoencoder (VAE Proxy):** We project the 14-bus state into a highly compressed latent space (d=4). FDIA attacks break historical correlation boundaries, resulting in massive reconstruction errors.
3. **Multi-Layer Perceptron (MLP):** A supervised neural network providing non-linear classification.

*All 3 models vote to form a single, aggregated anomaly confidence score: `P_Tier1`.*

### 3.2 The Physical State Indicators
Because attackers mathematically bypass the Chi-Square test, we use physics-informed graph metrics:
- **Graph Consistency Signature (GCS):** Using the grid's adjacency matrix, we monitor the smoothness of the power flow across connected neighboring buses. FDIA introduces harsh mathematical discontinuities between usually smooth neighboring sensors, spiking the GCS.

---

## 4. The Decision Engine: Reinforcement Learning (Phase 4)
This is where the true intelligence of the system resides. 
We model the grid as a Markov Decision Process (MDP) and employ an **Advantage Actor-Critic (A2C)** agent.

The agent receives the `[P_Tier1, GCS_Score, Chi2]` state and must balance two things: detection accuracy versus GNN computation cost.
Its reward function is defined as:
`Reward = Accuracy_Reward - (λ_c * computational_cost)`

- If `λ_c` is small, the agent learns to safely query Tier-2 (HGAT) very often.
- If `λ_c` is high, the agent learns to only use Tier-2 for the absolute most suspicious edge-cases, handling 95% of the grid data instantly at Tier-1.

### 5. Tier-2: Structural Hypergraph Verification (Phase 5)
When the RL Agent is unsure, it triggers **Tier-2**. This is a mathematically complex **Spectral Hypergraph Attention Network (HGAT)**.
Unlike standard GNNs which only connect nodes directly, a Hypergraph connects *zones* of the grid (e.g. all high-voltage industrial generation buses forming one hyperedge). The network performs message passing over these sets, making it incredibly robust at spotting stealthy topological manipulations. 

---

## 6. How to Run the Prototype Demonstration

This project has been implemented fully end-to-end to run identically on any machine without proprietary software (Pure NumPy/CPU fallback enabled).

### 6.1 Executing the Core Simulation

```bash
# 1. Activate the environment (Ensure the dependencies from requirements.txt are installed)
source ~/myenv/bin/activate

# 2. Navigate to the project
cd ~/Desktop/fdia

# 3. Generate the Real Diurnal Smart Grid Dataset
PYTHONPATH=. python src/simulation/generate_data.py

# 4. Train Tier-1 Models
PYTHONPATH=. python src/tier1/train_tier1_numpy.py

# 5. Run the Full Integration Pipeline (Trains HGAT & RL Agent + Computes Results)
# This will also automatically update the demo dashboard with the REAL inferences!
PYTHONPATH=. python src/full_pipeline.py
```

### 6.2 Viewing the Live Web Dashboard
After running the commands above, the live interactive dashboard is fully synchronized with the real ML predictions.

1. Open your File Explorer. 
2. Go to `Desktop/fdia/dashboard/`.
3. Double click on `index.html` to open it in your web browser. 

The dashboard provides a highly visual, fully responsive interface showing:
- Real-time animated **IEEE 14-Bus topological diagrams** with identified FDIA targets reflecting in red.
- **Tier-1 Score Gauges** showing exactly how the IF, Autoencoder, and MLP voted.
- **The RL A2C Decision Pipeline**, allowing you to dynamically toggle the verification penalty (`λ_c`) and watch the RL agent shift its strategy live.
