import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import os

from src.tier1.isolation_forest import IFDetector
from src.tier1.vae import VAEDetector
from src.tier1.mlp import MLPDetector
from src.tier1.fusion import Tier1Fusion

def main():
    print("Loading data...")
    with open("data/simulated/case14_data.json", "r") as f:
        data = json.load(f)
        
    X = np.array([d["z"] for d in data])
    y = np.array([d["label"] for d in data])
    
    # 80-20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Extract only normal data for IF and VAE training
    X_train_normal = X_train[y_train == 0]
    
    input_dim = X.shape[1]
    
    # Init models
    model_if = IFDetector(contamination=0.3) 
    model_vae = VAEDetector(input_dim)
    model_mlp = MLPDetector(input_dim)
    
    print("Training IF...")
    model_if.fit(X_train_normal)
    print("Training VAE...")
    model_vae.fit(X_train_normal)
    print("Training MLP...")
    model_mlp.fit(X_train, y_train)
    
    fusion = Tier1Fusion(model_if, model_vae, model_mlp)
    
    # Evaluation
    print("Evaluating...")
    p_ifs, p_vaes, p_mlps, p_fusions = [], [], [], []
    for z in X_test:
        p_f, p_i, p_v, p_m = fusion.predict_proba(z)
        p_ifs.append(p_i)
        p_vaes.append(p_v)
        p_mlps.append(p_m)
        p_fusions.append(p_f)
        
    for name, p_preds in [("IF", p_ifs), ("VAE", p_vaes), ("MLP", p_mlps), ("Fusion", p_fusions)]:
        auc = roc_auc_score(y_test, p_preds)
        acc = accuracy_score(y_test, np.array(p_preds) > 0.5)
        print(f"{name} -> ROC-AUC: {auc:.4f}, Accuracy: {acc:.4f}")
        
    # Example logic dump
    print("\n--- Example Sanity Check ---")
    idx = 0
    p_f, p_i, p_v, p_m = fusion.predict_proba(X_test[idx])
    print(f"Sample {idx} (True Label: {y_test[idx]}): IF {p_i:.2f}, VAE {p_v:.2f}, MLP {p_m:.2f} -> Fusion {p_f:.2f}")

if __name__ == "__main__":
    main()
