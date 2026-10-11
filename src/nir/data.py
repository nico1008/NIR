"""Image auditing, group splits and the shared preprocessing pipeline."""

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

from .config import CLASSES, CONFIG


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def audit(data, out):
    rows, seen, duplicates = [], {}, []
    for path in sorted(data.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if path.parent.name not in CLASSES:
            raise ValueError(f"Unexpected label: {path}")
        with Image.open(path) as im:
            gray = im.convert("L")
            digest = hashlib.sha256(str(gray.size).encode() + gray.tobytes()).hexdigest()
        label = CLASSES.index(path.parent.name)
        if digest in seen:
            if seen[digest]["label"] != label:
                raise ValueError(f"Duplicate with conflicting labels: {path}")
            duplicates.append(dict(path=str(path), retained=seen[digest]["path"]))
            continue
        match = (
            re.search(r"person\d+", path.stem, re.I)
            if label
            else re.search(r"(?:NORMAL2-)?IM-\d+", path.stem, re.I)
        )
        if not match:
            raise ValueError(f"Cannot infer grouping identifier: {path}")
        row = dict(
            path=str(path.resolve()), label=label, group=match.group(0).upper(), sha256=digest
        )
        seen[digest] = row
        rows.append(row)
    if not rows:
        raise ValueError("No images found")
    labels = np.array([r["label"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    # 20 fixed stratified group folds: 14 train, 3 validation, 3 test.
    folds = np.empty(len(rows), dtype=int)
    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)
    for fold, (_, indexes) in enumerate(splitter.split(np.zeros(len(rows)), labels, groups)):
        folds[indexes] = fold
    for row, fold in zip(rows, folds, strict=True):
        row["split"] = "train" if fold < 14 else "val" if fold < 17 else "test"
    split_groups = {
        s: {r["group"] for r in rows if r["split"] == s} for s in ("train", "val", "test")
    }
    assert not (
        split_groups["train"] & split_groups["val"]
        or split_groups["train"] & split_groups["test"]
        or split_groups["val"] & split_groups["test"]
    )
    out.mkdir(parents=True, exist_ok=True)
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    counts = {
        s: {
            c: sum(r["split"] == s and r["label"] == i for r in rows) for i, c in enumerate(CLASSES)
        }
        for s in split_groups
    }
    save_json(
        out / "audit.json",
        dict(
            retained=len(rows),
            excluded_duplicates=len(duplicates),
            counts=counts,
            grouping_limitation="Filename identifiers are proxies, not verified patient IDs. person numbers are grouped across bacteria/virus conservatively.",
            dataset_provenance="Local files lack patient metadata and verified provenance; external generalization is not established.",
        ),
    )
    save_json(out / "duplicates.json", duplicates)
    print(json.dumps(counts), flush=True)


def load_rows(out, split=None):
    with (out / "manifest.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["label"] = int(row["label"])
    return [r for r in rows if split is None or r["split"] == split]


def image_tensor(path):
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32)
    img = xrv.datasets.normalize(rgb, 255).mean(axis=2)[None, ...]
    img = xrv.datasets.XRayCenterCrop()(img)
    img = xrv.datasets.XRayResizer(224)(img)
    return torch.from_numpy(np.asarray(img, dtype=np.float32).copy())


class Images(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return image_tensor(row["path"]), row["label"]


def loader(rows, shuffle=False, workers=4):
    return DataLoader(
        Images(rows),
        batch_size=CONFIG["batch_size"],
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
