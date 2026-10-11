"""Command-line entry point. All experiment paths are explicit and portable."""

import argparse
import sys
from pathlib import Path

import torch

from .data import audit
from .engine import evaluate, predict, train


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return number


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(
        prog="nir",
        description="DenseNet121 + capsule chest X-ray experiment",
    )
    parser.add_argument("command", choices=["audit", "train", "evaluate", "predict"])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("runs/default"))
    parser.add_argument("--folder", type=Path)
    parser.add_argument(
        "--workers", type=nonnegative, default=4, help="DataLoader processes; use 0 for debugging"
    )
    args = parser.parse_args(argv)
    if args.command == "audit":
        if (args.out / "best.pt").exists():
            raise RuntimeError("Cannot change the manifest of a trained experiment")
        audit(args.data, args.out)
    elif args.command == "train":
        train(args.out, workers=args.workers)
    elif args.command == "evaluate":
        evaluate(args.out, workers=args.workers)
    else:
        if args.folder is None:
            parser.error("predict requires --folder")
        predict(args.out, args.folder, workers=args.workers)
