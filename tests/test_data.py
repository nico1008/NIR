import numpy as np
import pytest
import torch
from PIL import Image

from nir.data import audit, image_tensor, load_rows, loader


def save_image(path, pixels):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)


def test_audit_excludes_duplicates_and_keeps_groups_together(tmp_path):
    data = tmp_path / "data"
    rng = np.random.default_rng(42)
    for index in range(30):
        for view in range(2):
            save_image(
                data / "NORMAL" / f"IM-{index:04d}-{view}.png",
                rng.integers(0, 256, (16, 16), dtype=np.uint8),
            )
            condition = "bacteria" if view == 0 else "virus"
            save_image(
                data / "PNEUMONIA" / f"person{index}_{condition}_{view}.png",
                rng.integers(0, 256, (16, 16), dtype=np.uint8),
            )
    duplicate = data / "NORMAL" / "IM-9999-copy.png"
    duplicate.write_bytes((data / "NORMAL" / "IM-0000-0.png").read_bytes())
    out = tmp_path / "run"
    audit(data, out)
    rows = load_rows(out)
    assert len(rows) == 120
    assert duplicate.exists()
    assert len({row["sha256"] for row in rows}) == len(rows)
    split_groups = {
        split: {r["group"] for r in rows if r["split"] == split}
        for split in ["train", "val", "test"]
    }
    assert all(split_groups.values())
    assert split_groups["train"].isdisjoint(split_groups["val"])
    assert split_groups["train"].isdisjoint(split_groups["test"])
    assert split_groups["val"].isdisjoint(split_groups["test"])
    for index in range(30):
        matches = [r for r in rows if r["group"] == f"PERSON{index}"]
        assert len(matches) == 2
        assert len({r["split"] for r in matches}) == 1


def test_conflicting_duplicate_labels_fail_before_manifest_is_written(tmp_path):
    pixels = np.arange(256, dtype=np.uint8).reshape(16, 16)
    save_image(tmp_path / "data/NORMAL/IM-0001.png", pixels)
    save_image(tmp_path / "data/PNEUMONIA/person1_bacteria.png", pixels)
    with pytest.raises(ValueError, match="conflicting labels"):
        audit(tmp_path / "data", tmp_path / "run")
    assert not (tmp_path / "run/manifest.csv").exists()


def test_preprocessing_and_zero_worker_loader(tmp_path):
    path = tmp_path / "sample.png"
    save_image(path, np.full((100, 200), 127, dtype=np.uint8))
    tensor = image_tensor(path)
    assert tensor.shape == (1, 224, 224)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()
    assert -1024 <= tensor.min() <= tensor.max() <= 1024
    batch, label = next(iter(loader([{"path": str(path), "label": 1}], workers=0)))
    assert torch.equal(batch[0], tensor)
    assert label.tolist() == [1]
