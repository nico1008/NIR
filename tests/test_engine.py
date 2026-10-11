import csv
import hashlib
import json
from unittest.mock import Mock

import numpy as np
import pytest

from nir import engine


def test_prediction_csv_uses_selected_threshold_and_records_errors(tmp_path):
    rows = [{"path": "normal.png", "label": 0}, {"path": "pneumonia.png", "label": 1}]
    scores = np.array([[0.8, 0.2], [0.6, 0.4]])
    path = tmp_path / "predictions.csv"
    engine.write_predictions(path, rows, scores, -0.1)
    with path.open() as file:
        records = list(csv.DictReader(file))
    assert [int(r["prediction"]) for r in records] == [0, 0]
    assert [int(r["error"]) for r in records] == [0, 1]


@pytest.mark.parametrize("scores", [np.ones((1, 2)), np.full((2, 2), np.nan)])
def test_prediction_writer_rejects_contract_violations(tmp_path, scores):
    path = tmp_path / "predictions.csv"
    with pytest.raises(ValueError):
        engine.write_predictions(path, [{"path": "a"}, {"path": "b"}], scores, 0)
    assert not path.exists()


def test_completed_experiment_is_never_overwritten(tmp_path):
    (tmp_path / "test_metrics.json").write_text("{}")
    with pytest.raises(RuntimeError, match="already exists"):
        engine.train(tmp_path)
    with pytest.raises(RuntimeError, match="already evaluated"):
        engine.evaluate(tmp_path)


def test_restore_rejects_changed_manifest(tmp_path, monkeypatch):
    (tmp_path / "manifest.csv").write_text("modified")
    monkeypatch.setattr(engine.torch, "load", lambda *a, **k: {"manifest_sha256": "old"})
    with pytest.raises(RuntimeError, match="Manifest changed"):
        engine.restored(tmp_path)


def test_restore_rejects_threshold_for_another_checkpoint(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("original")
    checkpoint = {
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "epoch": 6,
        "state": {},
    }
    (tmp_path / "threshold.json").write_text(json.dumps({"threshold": 0.1, "checkpoint_epoch": 3}))
    monkeypatch.setattr(engine.torch, "load", lambda *a, **k: checkpoint)
    monkeypatch.setattr(engine, "Model", Mock())
    with pytest.raises(RuntimeError, match="different checkpoint"):
        engine.restored(tmp_path)


def test_training_selection_restore_and_one_shot_evaluation(tmp_path, monkeypatch):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    class SmallModel(nn.Module):
        def __init__(self, pretrained=True):
            super().__init__()
            self.features = nn.Linear(2, 2, bias=False)
            self.capsules = nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.features.weight.copy_(torch.eye(2))
                self.capsules.weight.copy_(torch.eye(2))
            self.set_stage(0)

        def set_stage(self, stage):
            self.features.requires_grad_(bool(stage))

        def forward(self, images):
            return torch.sigmoid(self.capsules(self.features(images)))

    rows = [{"path": "normal.png", "label": 0}, {"path": "pneumonia.png", "label": 1}]
    dataset = TensorDataset(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 1]))
    (tmp_path / "manifest.csv").write_text("synthetic training manifest")
    monkeypatch.setattr(engine, "Model", SmallModel)
    monkeypatch.setattr(engine, "load_rows", lambda *a, **k: rows)
    monkeypatch.setattr(engine, "loader", lambda *a, **k: DataLoader(dataset, batch_size=2))
    monkeypatch.setattr(engine.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        engine, "CONFIG", {**engine.CONFIG, "epochs": 4, "warmup_epochs": 1, "patience": 2}
    )
    engine.train(tmp_path, workers=0)
    selection = json.loads((tmp_path / "threshold.json").read_text())
    history = json.loads((tmp_path / "history.json").read_text())
    assert selection["checkpoint_epoch"] == 1
    assert len(history) == 3
    engine.evaluate(tmp_path, workers=0)
    metrics = json.loads((tmp_path / "test_metrics.json").read_text())
    assert metrics["confusion_matrix"] == [[1, 0], [0, 1]]
    assert metrics["roc_auc"] == 1
    assert metrics["threshold"] == selection["threshold"]
    with pytest.raises(RuntimeError, match="already evaluated"):
        engine.evaluate(tmp_path, workers=0)
