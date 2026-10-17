import json
from src.medimage.utils import save_json, set_seed

def test_save_json(tmp_path):
    path = tmp_path / "nested" / "metrics.json"
    save_json({"accuracy": 0.5}, path)
    assert json.loads(path.read_text()) == {"accuracy": 0.5}

def test_seed_does_not_raise():
    set_seed(42)
