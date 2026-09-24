"""Train and evaluate the standard FSRCNN(56,12,4) x2 reference on T91/Set5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from member_a.standard_fsrcnn import train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "standard_fsrcnn")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    print(json.dumps(train_and_evaluate(ROOT / ".data" / "t91", ROOT / ".data" / "set5", args.output_dir, device, args.epochs, args.batch_size), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

