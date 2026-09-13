from sklearn.ensemble import IsolationForest
import numpy as np
import pickle

class IFDetector:
    def __init__(self, contamination=0.3):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scale_min = -1.0
        self.scale_max = 1.0

    def fit(self, X):
        self.model.fit(X)
        scores = self.model.decision_function(X) # lower is more anomalous
        self.scale_min = np.min(scores)
        self.scale_max = np.max(scores)

    def predict_proba(self, z):
        z = np.array(z).reshape(1, -1)
        score = self.model.decision_function(z)[0]
        # Invert such that lower score -> higher probability of attack
        p = (self.scale_max - score) / (self.scale_max - self.scale_min + 1e-6)
        return float(np.clip(p, 0.0, 1.0))

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            self.model = obj.model
            self.scale_min = obj.scale_min
            self.scale_max = obj.scale_max
