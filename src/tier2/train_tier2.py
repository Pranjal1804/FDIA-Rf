import json
import numpy as np
from src.tier2.hgat import build_incidence_matrix, HGATModelNumpy

def roc_auc_score_np(y_true, y_score):
    order = np.argsort(-y_score)
    yt = y_true[order]
    npos = yt.sum(); nneg = len(yt) - npos
    if npos == 0 or nneg == 0: return 0.5
    tp = fp = 0; tprs = []; fprs = []
    for v in yt:
        tp += int(v); fp += 1 - int(v)
        tprs.append(tp / npos); fprs.append(fp / nneg)
    return float(np.trapz(tprs, fprs))

def main():
    print("Loading data...")
    with open("data/simulated/case14_data.json") as f:
        data = json.load(f)

    X_nodes = []
    y = []
    for d in data:
        z = np.array(d["z"])
        # Use nodal active power injections as HGAT features
        p_inj = z[-14:] 
        X_nodes.append(p_inj.reshape(14, 1))
        y.append(d["label"])
        
    X = np.array(X_nodes, dtype=np.float32)
    y = np.array(y, dtype=int)

    mu = X.mean(0); std = X.std(0) + 1e-6
    X = (X - mu) / std

    lines = [(0, 1), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (3, 4), (3, 6),
             (3, 8), (4, 5), (5, 10), (5, 11), (5, 12), (6, 7), (6, 8), (8, 9),
             (8, 13), (9, 10), (11, 12), (12, 13)]
    line_or = [u for u,v in lines]
    line_ex = [v for u,v in lines]
    
    H = build_incidence_matrix(14, line_or, line_ex)

    n = len(X)
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    tr, te = idx[:int(0.8*n)], idx[int(0.8*n):]
    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]

    print("Training Tier-2 HGAT model (100 epochs)...")
    model = HGATModelNumpy(14, in_features=1, hidden_dim=32, lr=1e-2)
    model.fit(Xtr, ytr, H, epochs=100)

    p_hgat = model.predict_proba_batch(Xte, H)
    
    auc = roc_auc_score_np(yte, p_hgat)
    pred = (p_hgat > 0.5).astype(int)
    acc = np.mean(pred == yte)
    
    tp = np.sum((pred == 1) & (yte == 1))
    fp = np.sum((pred == 1) & (yte == 0))
    fn = np.sum((pred == 0) & (yte == 1))
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)

    print("\n--- Tier-2 HGAT ('Always-Verify' Baseline) ---")
    print(f"ROC-AUC : {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")

    import os
    os.makedirs("data/results", exist_ok=True)
    with open("data/results/tier2_results.json", "w") as f:
        json.dump({
            "auc": float(auc),
            "acc": float(acc),
            "f1": float(f1)
        }, f, indent=2)

if __name__ == "__main__":
    main()
