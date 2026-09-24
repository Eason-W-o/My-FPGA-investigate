"""Build a concise, same-benchmark comparison of Bicubic, current model, and FSRCNN-s."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from member_a.model import FSRCNNSubpixel, mac_breakdown

CURRENT = ROOT / "artifacts" / "evaluation" / "summary.json"
ORIGINAL = ROOT / "artifacts" / "original_fsrcnn_s" / "summary.json"
STANDARD = ROOT / "artifacts" / "standard_fsrcnn" / "summary.json"
OUTPUT = ROOT / "artifacts" / "model_comparison.csv"
REPORT = ROOT / "artifacts" / "model_comparison.md"


def main() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
    standard = json.loads(STANDARD.read_text(encoding="utf-8"))
    lightweight = FSRCNNSubpixel()
    lightweight_parameters = sum(parameter.numel() for parameter in lightweight.parameters())
    lightweight_gmac = float(mac_breakdown()["gmac_per_frame"])
    rows = [
        {
            "model": "Bicubic",
            "architecture": "traditional interpolation",
            "parameters": 0,
            "gmac_per_frame_960x540": 0.0,
            "psnr_db": current["set5"]["bicubic_psnr_db"],
            "ssim": current["set5"]["bicubic_ssim"],
            "deployment_role": "baseline",
        },
        {
            "model": "Original FSRCNN-s",
            "architecture": "FSRCNN-s(32,5,1) + 9x9 deconvolution",
            "parameters": original["parameters"],
            "gmac_per_frame_960x540": original["gmac_per_960x540_frame"],
            "psnr_db": original["original_fsrcnn_s_psnr_db"],
            "ssim": original["original_fsrcnn_s_ssim"],
            "deployment_role": "software quality reference",
        },
        {
            "model": "Standard FSRCNN",
            "architecture": "FSRCNN(56,12,4) + 9x9 deconvolution",
            "parameters": standard["parameters"],
            "gmac_per_frame_960x540": standard["gmac_per_960x540_frame"],
            "psnr_db": standard["standard_fsrcnn_psnr_db"],
            "ssim": standard["standard_fsrcnn_ssim"],
            "deployment_role": "higher-cost software reference",
        },
        {
            "model": "Team lightweight model INT8",
            "architecture": "d=16,s=8,m=1 + 5x5 PixelShuffle",
            "parameters": lightweight_parameters,
            "gmac_per_frame_960x540": lightweight_gmac,
            "psnr_db": current["set5"]["quant_psnr_db"],
            "ssim": current["set5"]["quant_ssim"],
            "deployment_role": "FPGA implementation target",
        },
    ]
    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = "# 模型对照结论\n\n"
    report += "所有指标均在同一 Set5 图像、Y 通道、Bicubic ×2 降采样口径下计算。\n\n"
    for row in rows:
        report += f"- **{row['model']}**：PSNR {row['psnr_db']:.4f} dB，SSIM {row['ssim']:.5f}，算力 {row['gmac_per_frame_960x540']:.4f} GMAC/帧。\n"
    report += "\nFSRCNN-s 与标准 FSRCNN 都是软件画质参考；团队 INT8 轻量模型才是 FPGA 实现目标，三者不可混称。\n"
    REPORT.write_text(report, encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
