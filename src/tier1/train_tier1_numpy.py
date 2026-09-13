"""
Pure-numpy Tier-1 training + evaluation.
Avoids torch/sklearn dependencies while producing real ROC-AUC/accuracy numbers.

Models:
  IF   -> sklearn IsolationForest if available, else Mahalanobis distance
  VAE  -> PCA-reconstruction error (linear AE, mathematically == optimal linear VAE)
  MLP  -> 2-layer fully-vectorised numpy network, trained end-to-end
  Fuse -> (P_IF + P_VAE + P_MLP) / 3
"""
import json, os
import numpy as np

# ─── helpers ─────────────────────────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def roc_auc_score_np(y_true, y_score):
    order = np.argsort(-y_score)
    yt = y_true[order]
    npos = yt.sum(); nneg = len(yt) - npos
    if npos == 0 or nneg == 0:
        return 0.5
    tp = fp = 0; tprs = []; fprs = []
    for v in yt:
        tp += int(v); fp += 1 - int(v)
        tprs.append(tp / npos); fprs.append(fp / nneg)
    return float(np.trapezoid(tprs, fprs) if hasattr(np, 'trapezoid') else np.trapz(tprs, fprs))

def acc(y_true, p, t=0.5):
    return float(np.mean((p > t).astype(int) == y_true))

# ─── Isolation Forest ─────────────────────────────────────────────────────────
def fit_predict_if(X_norm, X_test, contamination=0.3):
    try:
        from sklearn.ensemble import IsolationForest
        m = IsolationForest(contamination=contamination, random_state=42)
        m.fit(X_norm)
        raw = m.decision_function(X_test)
        return (raw.max() - raw) / (raw.max() - raw.min() + 1e-8)
    except ImportError:
        mu = X_norm.mean(0)
        cov = np.cov(X_norm.T) + np.eye(X_norm.shape[1]) * 1e-3
        ic = np.linalg.pinv(cov)
        d = np.array([(x-mu)@ic@(x-mu) for x in X_test])
        return d / (d.max() + 1e-8)

# ─── VAE proxy: PCA reconstruction error ──────────────────────────────────────
class PCAAutoEncoder:
    """Linear autoencoder trained on normal data. Reconstruction error = anomaly score."""
    def __init__(self, n_components=4):
        self.k = n_components
        self.mu = None; self.V = None
        self.max_err = 1.0

    def fit(self, X):
        self.mu = X.mean(0)
        Xc = X - self.mu
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.V = Vt[:self.k].T                   # (d, k)
        recon_tr = (Xc @ self.V) @ self.V.T + self.mu
        errs = np.mean((X - recon_tr)**2, axis=1)
        self.max_err = float(np.percentile(errs, 95))

    def predict_proba_batch(self, X):
        Xc = X - self.mu
        recon = (Xc @ self.V) @ self.V.T + self.mu
        errs = np.mean((X - recon)**2, axis=1)
        return np.clip(errs / (self.max_err + 1e-8), 0, 1)

# ─── MLP — fully vectorised ───────────────────────────────────────────────────
class NumpyMLP:
    def __init__(self, d_in, d_h=32, lr=5e-3):
        self.W1 = np.random.randn(d_in, d_h) * np.sqrt(2/d_in)
        self.b1 = np.zeros(d_h)
        self.W2 = np.random.randn(d_h, 1) * np.sqrt(2/d_h)
        self.b2 = np.zeros(1)
        self.lr = lr

    def _fwd(self, X):
        h = np.maximum(0, X @ self.W1 + self.b1)
        return sigmoid(h @ self.W2 + self.b2), h

    def fit(self, X, y, epochs=50, bs=64):
        y = y.reshape(-1, 1).astype(float)
        n = len(X)
        for _ in range(epochs):
            idx = np.random.permutation(n)
            for s in range(0, n, bs):
                b = idx[s:s+bs]
                xb, yb = X[b], y[b]
                out, h = self._fwd(xb)
                do = (out - yb) / len(b)
                dW2 = h.T @ do;  db2 = do.sum(0)
                dh  = do @ self.W2.T * (h > 0)
                dW1 = xb.T @ dh; db1 = dh.sum(0)
                self.W1 -= self.lr*dW1; self.b1 -= self.lr*db1
                self.W2 -= self.lr*dW2; self.b2 -= self.lr*db2

    def predict_proba_batch(self, X):
        return self._fwd(X)[0].flatten()

# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(42)
    print("Loading data...")
    with open("data/simulated/case14_data.json") as f:
        data = json.load(f)

    X = np.array([d["z"] for d in data], dtype=np.float64)
    y = np.array([d["label"] for d in data], dtype=int)

    # standardise
    mu = X.mean(0); std = X.std(0) + 1e-8
    X = (X - mu) / std

    n = len(X)
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    split = int(0.8 * n)
    tr, te = idx[:split], idx[split:]
    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]
    Xnorm = Xtr[ytr == 0]

    d_in = X.shape[1]
    print(f"d_in={d_in}  Train={len(Xtr)} ({ytr.sum()} atk)  Test={len(Xte)} ({yte.sum()} atk)")

    print("\nTraining IF ...")
    p_if = fit_predict_if(Xnorm, Xte)

    print("Training VAE (PCA-AE, k=4) ...")
    vae = PCAAutoEncoder(n_components=4)
    vae.fit(Xnorm)
    p_vae = vae.predict_proba_batch(Xte)

    print("Training MLP (50 epochs) ...")
    mlp = NumpyMLP(d_in, d_h=64, lr=5e-3)
    mlp.fit(Xtr, ytr, epochs=50)
    p_mlp = mlp.predict_proba_batch(Xte)

    p_fuse = (p_if + p_vae + p_mlp) / 3.0

    print(f"\n{'Model':<10} {'ROC-AUC':>9} {'Accuracy':>10}")
    print("-" * 32)
    results = {}
    for name, p in [("IF", p_if), ("VAE", p_vae), ("MLP", p_mlp), ("Fusion", p_fuse)]:
        au = roc_auc_score_np(yte, p)
        ac = acc(yte, p)
        results[name] = {"auc": round(au,4), "acc": round(ac,4)}
        print(f"{name:<10} {au:>9.4f} {ac:>10.4f}")

    # Sanity-check: verify fusion arithmetic exactly
    i0 = 0
    pi, pv, pm = float(p_if[i0]), float(p_vae[i0]), float(p_mlp[i0])
    pf_check = (pi + pv + pm) / 3.0
    print(f"\n--- Fusion arithmetic sanity check (sample 0, label={yte[i0]}) ---")
    print(f"  IF={pi:.4f}  VAE={pv:.4f}  MLP={pm:.4f}")
    print(f"  P_Tier1 = ({pi:.4f}+{pv:.4f}+{pm:.4f})/3 = {pf_check:.4f}  (stored={float(p_fuse[i0]):.4f})")
    assert abs(pf_check - float(p_fuse[i0])) < 1e-9, "Fusion arithmetic mismatch!"
    print("  ✓ Fusion arithmetic verified.")

    os.makedirs("data/results", exist_ok=True)
    with open("data/results/tier1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved → data/results/tier1_results.json")

if __name__ == "__main__":
    main()
