"""
Phase 6: Lean End-to-End Pipeline (all compute in one flat script, no slow einsum loops).

Steps:
  1. Load data
  2. Tier-1 (IF + PCA-AE + MLP)  →  p_tier1 per sample
  3. GCS  →  vectorised in one shot using broadcast
  4. Tier-2 HGAT (pre-computed A_hyper, reshaped matmul, no per-batch einsum)
  5. A2C agent trained at λ ∈ {0.05, 0.3, 1.0}
  6. Stochastic evaluation + 3-way comparison table
"""

import json, os, time
import numpy as np
from src.tier1.train_tier1_numpy import fit_predict_if, PCAAutoEncoder, NumpyMLP
from src.indicators.gcs import build_adjacency
from src.rl.env import FDIAEnv
from src.rl.numpy_a2c import NumpyA2C
from src.rl.reward import DECIDE_NORMAL, DECIDE_ATTACK, ACQUIRE_GNN, REVIEW

np.random.seed(0)

LINES = [(0,1),(0,4),(1,2),(1,3),(1,4),(2,3),(3,4),(3,6),(3,8),(4,5),
         (5,10),(5,11),(5,12),(6,7),(6,8),(8,9),(8,13),(9,10),(11,12),(12,13)]
LOR = [u for u,_ in LINES]; LEX = [_ for _,v in LINES]

# ─── helpers ─────────────────────────────────────────────────────────────────
def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x,-30,30)))
def f1(y,p):
    tp=np.sum((p==1)&(y==1)); fp=np.sum((p==1)&(y==0)); fn=np.sum((p==0)&(y==1))
    return float(2*tp/(2*tp+fp+fn+1e-9))
def auc_score(y,s):
    o=np.argsort(-s); yt=y[o]; np_=yt.sum(); nn=len(yt)-np_
    if np_==0 or nn==0: return .5
    tp=fp=0; ta,fa=[],[]
    for v in yt: tp+=int(v); fp+=1-int(v); ta.append(tp/np_); fa.append(fp/nn)
    return float(np.trapezoid(ta, fa) if hasattr(np, 'trapezoid') else np.trapz(ta, fa))

# ─── vectorised GCS (whole dataset at once) ──────────────────────────────────
def gcs_vectorised(X_node_batch, A):
    """X_node_batch: (N, n_bus)  → returns (N,) GCS values."""
    N, n_bus = X_node_batch.shape
    deg = A.sum(1).clip(1)
    D_inv = 1.0 / deg                  # (n_bus,)
    O = (D_inv[:, None] * (A @ X_node_batch.T)).T   # (N, n_bus)
    diff = X_node_batch - O
    return (diff ** 2).sum(1) / n_bus  # (N,)

# ─── Fast HGAT (pre-computed A_hyper, flat matmul) ───────────────────────────
def build_A_hyper(n_bus, line_or, line_ex):
    n_l = len(line_or); ne = n_l + 3
    H = np.zeros((n_bus, ne))
    for b in [0,1,2,5,7]: H[b,0] = 1.
    for b in [0,1,2,3,4]: H[b,1] = 1.
    for b in [8,9,10,11,12,13]: H[b,2] = 1.
    for l in range(n_l): H[line_or[l], 3+l] = 1.; H[line_ex[l], 3+l] = 1.
    dv = H.sum(1).clip(1e-8); de = H.sum(0).clip(1e-8)
    Dv = np.diag(dv**-0.5); De = np.diag(1./de)
    return Dv @ H @ De @ H.T @ Dv   # (n_bus, n_bus)

class FastHGAT:
    """Two-layer spectral HGAT. Forward is O(N·n_bus·hidden) with pre-computed A."""
    def __init__(self, n_feat, hidden=16, lr=5e-3):
        # n_feat = n_bus (or n_bus+1 with concat), He init
        self.W1 = np.random.randn(n_feat, hidden) * np.sqrt(2./n_feat)
        self.W2 = np.random.randn(hidden, hidden) * np.sqrt(2./hidden)
        self.Wfc = np.random.randn(hidden, 1) * np.sqrt(2./hidden)
        self.b   = np.zeros(1)
        self.lr  = lr

    def _fwd(self, X, A):
        # X: (N, n_bus)  A: (n_bus, n_bus)
        h1 = np.maximum(0, (A @ X.T).T @ self.W1)   # (N, hidden)
        h2 = np.maximum(0, h1 @ self.W2)             # (N, hidden)
        out = sigmoid(h2 @ self.Wfc + self.b)         # (N, 1)
        return out, h1, h2

    def fit(self, X, y, A, epochs=80, bs=64, lr=None):
        if lr is not None: self.lr = lr
        y = y.reshape(-1,1).astype(float); n = len(X)
        AX = (A @ X.T).T          # pre-compute graph conv (it's fixed)
        for ep in range(epochs):
            idx = np.random.permutation(n)
            for s in range(0, n, bs):
                b = idx[s:s+bs]; xb, yb = AX[b], y[b]
                out, h1, h2 = self._fwd(X[b], A)
                do = (out-yb)/len(b)
                dWfc = h2.T@do; db = do.sum(0)
                dh2 = (do@self.Wfc.T)*(h2>0)
                dW2 = h1.T@dh2
                dh1 = (dh2@self.W2.T)*(h1>0)
                dW1 = xb.T@dh1
                for w,d in [(self.W1,dW1),(self.W2,dW2),(self.Wfc,dWfc),(self.b,db)]:
                    w -= self.lr * np.clip(d,-1,1)

    def predict(self, X, A):
        return self._fwd(X, A)[0].flatten()

# ─── main ────────────────────────────────────────────────────────────────────
print(">>> Loading data")
with open("data/simulated/case14_data.json") as f: data=json.load(f)
X_raw=np.array([d["z"] for d in data],np.float64)
y=np.array([d["label"] for d in data],int)
attack_buses=[d.get("attacked_buses", []) for d in data]

idx=np.random.RandomState(0).permutation(len(X_raw))
tr,te=idx[:800],idx[800:]
att_te = [attack_buses[i] for i in te]

mu,sd=X_raw[tr].mean(0),X_raw[tr].std(0)+1e-8
X=(X_raw-mu)/sd; Xtr,Xte=X[tr],X[te]; ytr,yte=y[tr],y[te]; Xnorm=Xtr[ytr==0]
print(f"    Train={len(Xtr)} ({ytr.sum()} atk)  Test={len(Xte)} ({yte.sum()} atk)")

# ─── Tier-1 ──────────────────────────────────────────────────────────────────
print(">>> Tier-1")
t0=time.time()
p_if_tr=fit_predict_if(Xnorm,Xtr); p_if_te=fit_predict_if(Xnorm,Xte)
vae=PCAAutoEncoder(4); vae.fit(Xnorm)
p_vae_tr=vae.predict_proba_batch(Xtr); p_vae_te=vae.predict_proba_batch(Xte)
mlp=NumpyMLP(X.shape[1],d_h=32,lr=5e-3); mlp.fit(Xtr,ytr,epochs=15)
p_mlp_tr=mlp.predict_proba_batch(Xtr); p_mlp_te=mlp.predict_proba_batch(Xte)
p1_tr=(p_if_tr+p_vae_tr+p_mlp_tr)/3.; p1_te=(p_if_te+p_vae_te+p_mlp_te)/3.
print(f"    AUC={auc_score(yte,p1_te):.4f}  Acc={np.mean((p1_te>0.5)==yte):.4f}  ({time.time()-t0:.1f}s)")

# ─── GCS ─────────────────────────────────────────────────────────────────────
print(">>> GCS (vectorised)")
t0=time.time()
A=build_adjacency(14,LOR,LEX)
Pnodes_tr=X_raw[tr,-14:]; Pnodes_te=X_raw[te,-14:]
gcs_tr=gcs_vectorised(Pnodes_tr,A); gcs_te=gcs_vectorised(Pnodes_te,A)
chi2_tr=p1_tr*2-1+np.random.randn(len(tr))*0.3
chi2_te=p1_te*2-1+np.random.randn(len(te))*0.3
print(f"    GCS normal={gcs_tr[ytr==0].mean():.4f}  attack={gcs_tr[ytr==1].mean():.4f}  ({time.time()-t0:.1f}s)")

# ─── Tier-2 HGAT ─────────────────────────────────────────────────────────────
print(">>> Tier-2 HGAT (fast, pre-computed A)")
t0=time.time()
A_hyp=build_A_hyper(14,LOR,LEX)
# concat p_tier1 as an extra node feature (broadcast): (N, n_bus+1)
Pnodes_tr_aug = np.hstack([Pnodes_tr, p1_tr.reshape(-1,1)])
Pnodes_te_aug = np.hstack([Pnodes_te, p1_te.reshape(-1,1)])
A_aug = np.block([[A_hyp, np.zeros((14,1))],[np.zeros((1,14)), np.eye(1)]])  # (15,15)
hgat=FastHGAT(n_feat=15,hidden=32,lr=5e-3)
hgat.fit(Pnodes_tr_aug, ytr, A_aug, epochs=100)
p_hgat_te=hgat.predict(Pnodes_te_aug, A_aug)
t2_acc=np.mean((p_hgat_te>0.5)==yte); t2_f1=f1(yte,(p_hgat_te>0.5).astype(int))
print(f"    AUC={auc_score(yte,p_hgat_te):.4f}  Acc={t2_acc:.4f}  F1={t2_f1:.4f}  ({time.time()-t0:.1f}s)")

# ─── A2C × 3 λ values ────────────────────────────────────────────────────────
rl_data=[{"p_tier1":float(p1_tr[i]),"gcs":float(gcs_tr[i]),
          "chi2_std":float(chi2_tr[i]),"label":int(ytr[i])} for i in range(len(tr))]

t1_acc=np.mean((p1_te>0.5)==yte); t1_f1=f1(yte,(p1_te>0.5).astype(int))
results={}
for lam in [0.05, 0.3, 1.0]:
    print(f">>> A2C λ={lam}")
    t0=time.time()
    env=FDIAEnv(rl_data,lambda_c=lam,cost_gnn=1.0)
    agent=NumpyA2C(state_dim=4,action_dim=4,hidden_dim=64,lr=2e-3)
    ep_r=[]
    for ep in range(3000):
        s,_=env.reset(); done=False; st,at,rt=[],[],[]
        while not done:
            a=agent.select_action(s); sn,r,done,_,_=env.step(a)
            st.append(s);at.append(a);rt.append(r);s=sn
        G=0; ret=[]
        for r in reversed(rt): G=r+G; ret.insert(0,G)
        # add small entropy bonus to keep exploration alive
        probs,_ = agent.forward_actor(np.array(st))
        entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1).mean()
        ret = [g + 0.05*entropy for g in ret]
        agent.update(st,at,ret); ep_r.append(sum(rt))
    # eval
    preds,hc,acts=[],0,{0:0,1:0,2:0,3:0}
    for i in range(len(te)):
        s=np.array([p1_te[i],gcs_te[i],chi2_te[i],0.],np.float32)
        a=agent.select_action(s); acts[a]=acts.get(a,0)+1
        if   a==DECIDE_NORMAL:  preds.append(0)
        elif a==DECIDE_ATTACK:  preds.append(1)
        else:
            hc+=1
            s2=s.copy(); s2[3]=1.
            a2=agent.select_action(s2)
            preds.append(int(p_hgat_te[i]>=0.5) if a2 in (REVIEW,ACQUIRE_GNN)
                         else (1 if a2==DECIDE_ATTACK else 0))
    preds=np.array(preds)
    acc_=np.mean(preds==yte); f1_=f1(yte,preds); hr=hc/len(te)
    print(f"    Acc={acc_:.4f}  F1={f1_:.4f}  HGAT%={hr:.1%}  "
          f"MeanR(last500)={np.mean(ep_r[-500:]):.4f}  ({time.time()-t0:.1f}s)")
    print(f"    actions: {acts}")
    results[str(lam)]=dict(acc=float(acc_),f1=float(f1_),hgat_rate=float(hr))

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"{'Method':<30} {'Acc':>7} {'F1':>7} {'HGAT%':>7}")
print("-"*60)
print(f"{'Tier-1 Fusion (never verify)':<30} {t1_acc:>7.4f} {t1_f1:>7.4f} {'0%':>7}")
print(f"{'Tier-2 HGAT (always verify)':<30} {t2_acc:>7.4f} {t2_f1:>7.4f} {'100%':>7}")
for lam,r in results.items():
    tag=f"A2C λ={lam} (selective)"
    print(f"{tag:<30} {r['acc']:>7.4f} {r['f1']:>7.4f} {r['hgat_rate']:>7.1%}")
print("="*60)

os.makedirs("data/results",exist_ok=True)
with open("data/results/phase6_results.json","w") as f:
    json.dump({"tier1":{"acc":t1_acc,"f1":t1_f1},
               "tier2":{"acc":t2_acc,"f1":t2_f1},"a2c":results},f,indent=2)
print("Saved → data/results/phase6_results.json")

# ─── Dashboard Export ────────────────────────────────────────────────────────
print(">>> Exporting Real Inferences to Dashboard")
import re
dash_samples = []
for i in range(min(50, len(te))):
    dash_samples.append({
        "id": i,
        "label": int(yte[i]),
        "p_if": float(p_if_te[i]),
        "p_vae": float(p_vae_te[i]),
        "p_mlp": float(p_mlp_te[i]),
        "gcs": float(gcs_te[i]),
        "chi2": float(chi2_te[i]),
        "attacked_buses": att_te[i]
    })
    
try:
    with open("dashboard/index.html", "r") as f: html = f.read()
    html = re.sub(r'const SAMPLES=/\*INJECT_SAMPLES_HERE\*/\[\];', f'const SAMPLES={json.dumps(dash_samples)};', html)
    with open("dashboard/index.html", "w") as f: f.write(html)
    print("Updated dashboard/index.html with real samples!")
except Exception as e:
    print(f"Could not update dashboard: {e}")
