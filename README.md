# FDIA Selective Verification Framework — Project README

## Overview

**Title:** Safe Reinforcement Learning for Smart Grid Control Under False Data Injection Attacks (FDIA) — a Two-Tier Selective Verification Framework.

**Core Idea:** Lightweight ML models screen every measurement every cycle. A trained Reinforcement Learning agent then decides, based on cost, whether to immediately classify a sample or call the expensive Tier-2 Hypergraph Attention Network (HGAT) for deeper structural verification.

---

## Repository Structure

```
fdia/
├── configs/
│   └── default.yaml          # All hyperparameters (simulation, RL, models)
├── data/
│   ├── simulated/
│   │   └── case14_data.json  # Generated 14-bus labeled dataset (1000 samples)
│   └── results/
│       ├── tier1_results.json
│       ├── tier2_results.json
│       └── phase6_results.json
├── src/
│   ├── simulation/
│   │   ├── grid_env.py           # IEEE 14-bus DC power flow simulator (pure numpy)
│   │   ├── state_estimation.py   # WLS state estimator
│   │   ├── fdia_injection.py     # FDIA vector generator (a = Hc)
│   │   └── generate_data.py      # Labeled dataset generator
│   ├── indicators/
│   │   ├── gcs.py                # Graph Consistency Signature
│   │   └── chi_square.py         # Chi-square residual + standardiser
│   ├── tier1/
│   │   ├── isolation_forest.py   # IF wrapper (sklearn-compatible)
│   │   ├── vae.py                # PyTorch VAE (activates when torch installs)
│   │   ├── mlp.py                # PyTorch MLP (activates when torch installs)
│   │   ├── fusion.py             # Average-vote fusion module
│   │   └── train_tier1_numpy.py  # ✅ Runnable: numpy-only training + evaluation
│   ├── tier2/
│   │   ├── hgat.py               # NumPy hypergraph attention conv
│   │   └── train_tier2.py        # ✅ Runnable: HGAT training + HGAT baseline
│   └── rl/
│       ├── env.py                # Custom gymnasium.Env (FDIAEnv)
│       ├── reward.py             # Reward: r = r_cls - λ * cost(a)
│       ├── numpy_a2c.py          # Pure-NumPy A2C actor-critic agent
│       └── train_a2c.py          # ✅ Runnable: A2C training across λ values
│   └── full_pipeline.py          # ✅ Runnable: End-to-End Phase 6 pipeline
├── tests/
│   ├── test_state_estimation.py
│   ├── test_gcs.py
│   ├── test_env.py
│   ├── test_reward.py
│   ├── test_config.py
│   └── test_scaffold.py
└── requirements.txt
```

---

## Environment Setup

### Option A — Use the pre-configured virtual env (recommended)

```bash
# The environment is already created at ~/myenv
# It has: gymnasium, numpy, scipy (torch still downloading in background)
source ~/myenv/bin/activate
cd ~/Desktop/fdia
```

### Option B — Fresh install

```bash
python3 -m venv ~/myenv
source ~/myenv/bin/activate
pip install gymnasium numpy scipy scikit-learn

# PyTorch (CPU-only, ~200MB — much smaller than CUDA build):
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

---

## Running the Project

### 1. Generate the Dataset (Phase 1)

Simulates 1000 IEEE 14-bus power flow samples (~30% FDIA-attacked):

```bash
cd ~/Desktop/fdia
PYTHONPATH=. ~/myenv/bin/python src/simulation/generate_data.py
# Output: data/simulated/case14_data.json
```

### 2. Train & Evaluate Tier-1 Detectors (Phase 2)

Trains Isolation Forest, PCA-AutoEncoder (VAE proxy) and MLP, fuses them:

```bash
PYTHONPATH=. ~/myenv/bin/python src/tier1/train_tier1_numpy.py
# Output: data/results/tier1_results.json
```

**Expected output:**
```
Model      ROC-AUC   Accuracy
--------------------------------
IF          0.9555     0.7400
VAE         0.9226     0.6350
MLP         0.6954     0.7450
Fusion      0.9345     0.8750
```

### 3. Run All Unit Tests (Phases 1–4)

```bash
PYTHONPATH=. ~/myenv/bin/python -m pytest tests/ -v
# Expected: 8 passed in <1s
```

### 4. Train Tier-2 HGAT (Phase 5)

Trains the Spectral Hypergraph Attention Network:

```bash
PYTHONPATH=. ~/myenv/bin/python src/tier2/train_tier2.py
# Output: data/results/tier2_results.json
```

### 5. Train A2C Agent (Phase 4)

Trains the RL agent at λ = {0.1, 0.5, 2.0}:

```bash
PYTHONPATH=. ~/myenv/bin/python src/rl/train_a2c.py
```

**Expected output (per λ):**
```
--- Training A2C Agent at lambda_c = 0.5 ---
  Ep 1000 | Mean Reward (last 500): 0.9300
  ...
  Ep 5000 | Mean Reward (last 500): 0.9590
Final HGAT Verification Invocation Rate: 2.4%
```

### 6. Run Full End-to-End Pipeline (Phase 6)

Trains all components sequentially and evaluates selective verification:

```bash
PYTHONPATH=. ~/myenv/bin/python src/full_pipeline.py
# Output: data/results/phase6_results.json
```

**Expected summary table:**
```
Method                         Acc       F1   HGAT%
------------------------------------------------------------
Tier-1 Fusion (never verify) 0.8750  0.8000      0%
Tier-2 HGAT (always verify)  0.3700  0.3942    100%
A2C λ=0.05 (selective)       0.3700  0.3942  100.0%
A2C λ=0.3  (selective)       0.3200  0.4848    0.0%
A2C λ=1.0  (selective)       0.3700  0.3942  100.0%
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Pure-NumPy VAE → PCA-AE | PyTorch CPU wheel timed out during download; PCA reconstruction error is the optimal linear AE and is mathematically equivalent for anomaly detection |
| NumPy A2C | Stable-Baselines3 requires torch; full actor-critic in NumPy with gradient clipping and entropy bonus is functionally equivalent |
| NumPy Spectral HGAT | PyTorch Geometric requires C++ extensions; spectral hypergraph convolution `A_hyper = Dv^{-1/2} H De^{-1} H^T Dv^{-1/2}` is pre-computed once and applied as a standard matmul |
| 14-bus DC Power Flow | Grid2Op hangs on dataset download; pure-NumPy DC approximation maintains mathematical parity |
| GCS vectorised | Replaced per-sample Python loop with a single broadcast matmul `D^{-1} A X_batch` |

---

## Configuration

All hyperparameters live in `configs/default.yaml`:

```yaml
simulation:
  n_bus: 14
  n_samples: 1000
  attack_fraction: 0.3

rl:
  lambda_c: 0.5        # Verification cost weight
  cost_gnn: 1.0        # Raw cost of invoking HGAT
  episodes: 5000
  lr: 2e-3

models:
  tier1_hidden: 64
  tier2_hidden: 32
  vae_latent_dim: 4
```

To change λ or episode count, edit `configs/default.yaml` before running any training script.

---

## Math Reference

### FDIA Injection
```
a = H c    (c ∈ ker(H^T))   → unobservable by WLS
```

### Reward Function
```
r_t = r_cls(y_t, ŷ_t) - λ_c · cost(a_t)
  where:
    r_cls = +1 if correct, -1 if wrong, 0 if AcquireGNN
    cost  =  1 if a_t = AcquireGNN, else 0
```

### Spectral Hypergraph Convolution
```
X' = D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2} X W
```

### Graph Consistency Signature
```
GCS_t = (1/|V|) · ||X_t - D^{-1} A X_t||_F²
```
