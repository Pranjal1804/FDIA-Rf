import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MLPDetector:
    def __init__(self, input_dim, hidden_dim=32):
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def fit(self, X, y, epochs=30, lr=1e-3):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        X_t = torch.FloatTensor(X)
        y_t = torch.FloatTensor(y).unsqueeze(1)
        
        self.model.train()
        for ep in range(epochs):
            optimizer.zero_grad()
            out = self.model(X_t)
            loss = criterion(out, y_t)
            loss.backward()
            optimizer.step()

    def predict_proba(self, z):
        self.model.eval()
        z_t = torch.FloatTensor(z).unsqueeze(0)
        with torch.no_grad():
            p = self.model(z_t).item()
        return float(p)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
