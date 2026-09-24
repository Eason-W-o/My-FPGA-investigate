"""Train and evaluate the original FSRCNN-s x2 reference on current T91/Set5 data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from member_a.original_fsrcnn_s import train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "original_fsrcnn_s")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    result = train_and_evaluate(
        ROOT / ".data" / "t91",
        ROOT / ".data" / "set5",
        args.output_dir,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
