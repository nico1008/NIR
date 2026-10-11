"""Train once, select on validation and evaluate the held-out test set once."""

import csv
import hashlib
import json
import random
import time

import numpy as np
import torch
import torchxrayvision as xrv
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn

from .config import CONFIG
from .data import load_rows, loader, save_json
from .model import Model, margin_loss


def seed_all():
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.inference_mode()
def infer(model, data_loader, device):
    model.eval()
    labels, outputs = [], []
    for x, y in data_loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            scores = model(x.to(device, non_blocking=True))
        labels.extend(y.numpy().tolist())
        outputs.extend(scores.cpu().numpy().tolist())
    return np.array(labels), np.array(outputs)


def train(out, workers=4):
    if (out / "test_metrics.json").exists() or (out / "best.pt").exists():
        raise RuntimeError("Experiment already exists; refusing to overwrite it")
    seed_all()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_rows(out, "train")
    train_loader, val_loader = (
        loader(rows, True, workers),
        loader(load_rows(out, "val"), workers=workers),
    )
    model = Model().to(device)
    class_counts = np.bincount([r["label"] for r in rows], minlength=2)
    weights = torch.tensor(len(rows) / (2 * class_counts), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.capsules.parameters(),
        lr=CONFIG["head_warmup_lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best, stale, history = -1.0, 0, []
    save_json(
        out / "config.json",
        {**CONFIG, "device": str(device), "torch": torch.__version__, "xrv": xrv.__version__},
    )
    start = time.perf_counter()
    for epoch in range(CONFIG["epochs"]):
        if epoch == CONFIG["warmup_epochs"]:
            model.set_stage(1)
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.capsules.parameters(), "lr": CONFIG["head_lr"]},
                    {
                        "params": [p for p in model.features.parameters() if p.requires_grad],
                        "lr": CONFIG["backbone_lr"],
                    },
                ],
                weight_decay=CONFIG["weight_decay"],
            )
            stale = 0
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                scores = model(x)
                loss = margin_loss(scores, y, weights)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(y)
        y, scores = infer(model, val_loader, device)
        auc = roc_auc_score(y, scores[:, 1] - scores[:, 0])
        improved = auc > best + 1e-6
        if improved:
            best, stale = auc, 0
            torch.save(
                dict(
                    state=model.state_dict(),
                    epoch=epoch + 1,
                    val_auc=auc,
                    config=CONFIG,
                    manifest_sha256=hashlib.sha256((out / "manifest.csv").read_bytes()).hexdigest(),
                ),
                out / "best.pt",
            )
        else:
            stale += 1
        record = dict(
            epoch=epoch + 1,
            train_loss=total / len(rows),
            val_auc=float(auc),
            elapsed_seconds=time.perf_counter() - start,
        )
        history.append(record)
        save_json(out / "history.json", history)
        print(json.dumps(record), flush=True)
        if epoch >= CONFIG["warmup_epochs"] and stale >= CONFIG["patience"]:
            break
    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state"])
    y, scores = infer(model, val_loader, device)
    raw = scores[:, 1] - scores[:, 0]
    fpr, tpr, thresholds = roc_curve(y, raw)
    candidates = np.flatnonzero(np.isfinite(thresholds))
    index = candidates[np.argmax((tpr - fpr)[candidates])]
    save_json(
        out / "threshold.json",
        dict(
            threshold=float(thresholds[index]),
            method="Youden index on validation",
            checkpoint_epoch=checkpoint["epoch"],
        ),
    )
    write_predictions(out / "val_predictions.csv", load_rows(out, "val"), scores, thresholds[index])


def write_predictions(path, rows, scores, threshold):
    scores = np.asarray(scores)
    if scores.shape != (len(rows), 2):
        raise ValueError("Each input row must have exactly two class scores")
    if not np.isfinite(scores).all() or not np.isfinite(threshold):
        raise ValueError("Scores and threshold must be finite")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "path",
                "label",
                "normal_score",
                "pneumonia_score",
                "decision_score",
                "prediction",
                "error",
            ]
        )
        for row, score in zip(rows, scores, strict=True):
            raw = float(score[1] - score[0])
            pred = int(raw >= threshold)
            w.writerow(
                [
                    row["path"],
                    row.get("label", ""),
                    *score,
                    raw,
                    pred,
                    "" if "label" not in row else int(pred != row["label"]),
                ]
            )


def restored(out):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=True)
    if (
        hashlib.sha256((out / "manifest.csv").read_bytes()).hexdigest()
        != checkpoint["manifest_sha256"]
    ):
        raise RuntimeError("Manifest changed after training")
    model = Model(pretrained=False).to(device)
    model.load_state_dict(checkpoint["state"])
    selection = json.loads((out / "threshold.json").read_text())
    if selection["checkpoint_epoch"] != checkpoint["epoch"]:
        raise RuntimeError("Threshold was selected for a different checkpoint")
    threshold = selection["threshold"]
    if not np.isfinite(threshold):
        raise ValueError("Classification threshold must be finite")
    return model, device, threshold


def evaluate(out, workers=4):
    if (out / "test_metrics.json").exists():
        raise RuntimeError("Test already evaluated; read existing metrics instead")
    model, device, threshold = restored(out)
    rows = load_rows(out, "test")
    y, scores = infer(model, loader(rows, workers=workers), device)
    raw = scores[:, 1] - scores[:, 0]
    pred = raw >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    metrics = dict(
        roc_auc=float(roc_auc_score(y, raw)),
        pr_auc=float(average_precision_score(y, raw)),
        sensitivity=float(tp / (tp + fn)),
        specificity=float(tn / (tn + fp)),
        f1=float(f1_score(y, pred)),
        accuracy=float((tp + tn) / len(y)),
        confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]],
        count=len(y),
        threshold=threshold,
    )
    write_predictions(out / "test_predictions.csv", rows, scores, threshold)
    save_json(out / "test_metrics.json", metrics)
    print(json.dumps(metrics), flush=True)


def predict(out, folder, workers=4):
    model, device, threshold = restored(out)
    rows = [
        dict(path=str(p))
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if not rows:
        raise ValueError("No supported images in folder")
    data = [{**r, "label": 0} for r in rows]
    _, scores = infer(model, loader(data, workers=workers), device)
    write_predictions(out / "folder_predictions.csv", rows, scores, threshold)
