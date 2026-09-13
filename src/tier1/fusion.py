class Tier1Fusion:
    def __init__(self, detector_if, detector_vae, detector_mlp):
        self.detector_if = detector_if
        self.detector_vae = detector_vae
        self.detector_mlp = detector_mlp

    def predict_proba(self, z):
        p_if = self.detector_if.predict_proba(z)
        p_vae = self.detector_vae.predict_proba(z)
        p_mlp = self.detector_mlp.predict_proba(z)
        p_fusion = (p_if + p_vae + p_mlp) / 3.0
        return p_fusion, p_if, p_vae, p_mlp
