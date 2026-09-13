from src.config import load_config

def test_load_config():
    config = load_config("configs/default.yaml")
    assert "experiment_name" in config
    assert config["experiment_name"] == "fdia_selective_verification"
    assert "simulation" in config
    assert "rl" in config
