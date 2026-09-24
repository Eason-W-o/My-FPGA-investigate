"""Reproducible software implementation of the paper's standard FSRCNN x2.

The original FSRCNN paper denotes its standard model as FSRCNN(56,12,4).
This module keeps that topology and its 9x9 transposed-convolution upsampler.
It is intentionally a software reference, not an FPGA deployment promise.
"""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import EvalDataset, TrainDataset
from .metrics import psnr_y, ssim_y
from .training import seed_everything


@dataclass(frozen=True)
class StandardFSRCNNConfig:
    """The standard FSRCNN(d=56, s=12, m=4) configuration in the paper."""

    scale: int = 2
    d: int = 56
    s: int = 12
    m: int = 4

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class StandardFSRCNN(nn.Module):
    """Paper-standard FSRCNN with PReLU and a 9x9 x2 deconvolution layer."""

    def __init__(self, config: StandardFSRCNNConfig | None = None) -> None:
        super().__init__()
        self.config = config or StandardFSRCNNConfig()
        if self.config.scale != 2:
            raise ValueError("This reproduction is fixed to x2.")
        cfg = self.config
        self.feature = nn.Sequential(nn.Conv2d(1, cfg.d, 5, padding=2), nn.PReLU(cfg.d))
        self.shrink = nn.Sequential(nn.Conv2d(cfg.d, cfg.s, 1), nn.PReLU(cfg.s))
        mapping: list[nn.Module] = []
        for _ in range(cfg.m):
            mapping.extend((nn.Conv2d(cfg.s, cfg.s, 3, padding=1), nn.PReLU(cfg.s)))
        self.mapping = nn.Sequential(*mapping)
        self.expand = nn.Sequential(nn.Conv2d(cfg.s, cfg.d, 1), nn.PReLU(cfg.d))
        self.deconvolution = nn.ConvTranspose2d(cfg.d, 1, kernel_size=9, stride=2, padding=4, output_padding=1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(module.weight, mean=0.0, std=0.001)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, lr_y: torch.Tensor) -> torch.Tensor:
        return self.deconvolution(self.expand(self.mapping(self.shrink(self.feature(lr_y)))))


def macs_per_lr_pixel() -> int:
    """Convolution MAC count per LR pixel, excluding bias and PReLU operations."""
    cfg = StandardFSRCNNConfig()
    return 1 * cfg.d * 5 * 5 + cfg.d * cfg.s + cfg.m * cfg.s * cfg.s * 3 * 3 + cfg.s * cfg.d + cfg.d * 9 * 9


@torch.no_grad()
def evaluate(model: StandardFSRCNN, dataset: EvalDataset, device: torch.device) -> tuple[list[dict[str, float | str]], dict[str, float]]:
    model.eval()
    rows: list[dict[str, float | str]] = []
    for index in range(len(dataset)):
        name, lr, hr = dataset[index]
        lr, hr = lr.unsqueeze(0).to(device), hr.unsqueeze(0).to(device)
        prediction = model(lr).clamp(0.0, 1.0)
        bicubic = functional.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False).clamp(0.0, 1.0)
        rows.append({
            "image": name,
            "bicubic_psnr_db": psnr_y(hr, bicubic),
            "bicubic_ssim": ssim_y(hr, bicubic),
            "standard_fsrcnn_psnr_db": psnr_y(hr, prediction),
            "standard_fsrcnn_ssim": ssim_y(hr, prediction),
        })
    summary = {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in ("bicubic_psnr_db", "bicubic_ssim", "standard_fsrcnn_psnr_db", "standard_fsrcnn_ssim")
    }
    return rows, summary


def train_and_evaluate(
    train_dir: Path,
    eval_dir: Path,
    output_dir: Path,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 16,
    seed: int = 123,
) -> dict[str, float | int | str]:
    """Train standard FSRCNN on T91 and evaluate it on Set5."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    model = StandardFSRCNN().to(device)
    train_set, eval_set = TrainDataset(train_dir), EvalDataset(eval_dir)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    optimizer = torch.optim.Adam(
        [
            {"params": [parameter for name, parameter in model.named_parameters() if not name.startswith("deconvolution.")]},
            {"params": model.deconvolution.parameters(), "lr": 1.0e-4},
        ],
        lr=1.0e-3,
    )
    criterion = nn.MSELoss()
    best_psnr, best_state = -float("inf"), deepcopy(model.state_dict())
    log: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        if epoch == 31:
            for group in optimizer.param_groups:
                group["lr"] *= 0.1
        model.train()
        total_loss, samples = 0.0, 0
        for lr, hr in tqdm(loader, desc=f"FSRCNN {epoch}/{epochs}", leave=False):
            lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(lr), hr)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * lr.shape[0]
            samples += lr.shape[0]
        _, evaluation = evaluate(model, eval_set, device)
        row = {
            "epoch": epoch,
            "mse_loss": total_loss / max(samples, 1),
            "set5_psnr_db": evaluation["standard_fsrcnn_psnr_db"],
            "set5_ssim": evaluation["standard_fsrcnn_ssim"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        log.append(row)
        print(f"FSRCNN epoch {epoch:02d}/{epochs}: MSE={row['mse_loss']:.6f}, Set5 PSNR={row['set5_psnr_db']:.4f}")
        if evaluation["standard_fsrcnn_psnr_db"] > best_psnr:
            best_psnr, best_state = evaluation["standard_fsrcnn_psnr_db"], deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    rows, summary = evaluate(model, eval_set, device)
    torch.save(
        {"state_dict": model.state_dict(), "config": model.config.to_dict(), "seed": seed, "epochs": epochs},
        output_dir / "standard_fsrcnn_x2_t91.pt",
    )
    with (output_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(log[0]))
        writer.writeheader()
        writer.writerows(log)
    with (output_dir / "set5_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows + [{"image": "AVERAGE", **summary}])
    result: dict[str, float | int | str] = {
        "model": "Standard FSRCNN(56,12,4) x2, 9x9 transposed convolution",
        "dataset_train": str(train_dir),
        "dataset_eval": str(eval_dir),
        "epochs": epochs,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "macs_per_lr_pixel": macs_per_lr_pixel(),
        "gmac_per_960x540_frame": macs_per_lr_pixel() * 960 * 540 / 1.0e9,
        "gmac_per_second_30fps": macs_per_lr_pixel() * 960 * 540 * 30 / 1.0e9,
        **summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result

