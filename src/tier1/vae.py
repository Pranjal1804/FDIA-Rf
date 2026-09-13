import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class VAEDetector:
    def __init__(self, input_dim, hidden_dim=16, latent_dim=4):
        self.model = VAE(input_dim, hidden_dim, latent_dim)
        self.max_recon_err = 1.0

    def fit(self, X, epochs=20, lr=1e-3):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        X_t = torch.FloatTensor(X)
        self.model.train()
        for ep in range(epochs):
            optimizer.zero_grad()
            recon, mu, logvar = self.model(X_t)
            recon_loss = nn.MSELoss(reduction='sum')(recon, X_t) / X_t.shape[0]
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / X_t.shape[0]
            loss = recon_loss + kld_loss * 0.1
            loss.backward()
            optimizer.step()
            
        self.model.eval()
        with torch.no_grad():
            recon, _, _ = self.model(X_t)
            errors = torch.mean((X_t - recon)**2, dim=1)
            self.max_recon_err = torch.quantile(errors, 0.95).item()
            
    def predict_proba(self, z):
        self.model.eval()
        z_t = torch.FloatTensor(z).unsqueeze(0)
        with torch.no_grad():
            recon, _, _ = self.model(z_t)
            err = torch.mean((z_t - recon)**2).item()
            
        p = err / (self.max_recon_err + 1e-6)
        return float(np.clip(p, 0.0, 1.0))

    def save(self, path):
        torch.save({'state_dict': self.model.state_dict(), 'err': self.max_recon_err}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.max_recon_err = checkpoint['err']
