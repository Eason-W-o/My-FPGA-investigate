"""Create a 540p-to-1080p super-resolution video demonstration.

The script uses the team's INT8 fake-quantized lightweight model and the
software-only original FSRCNN-s reference on the same 540p Y input.  It emits
individual result videos, a labelled 2x2 comparison video, and per-frame
PSNR/SSIM measured against the source 1080p Y frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from member_a.metrics import psnr_y, ssim_y
from member_a.model import FSRCNNSubpixel
from member_a.original_fsrcnn_s import OriginalFSRCNNS
from member_a.quantization import quantized_forward_float


HR_SIZE = (1920, 1080)
LR_SIZE = (960, 540)


def tensor_from_y(y: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(y.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)


def y_from_tensor(tensor: torch.Tensor) -> np.ndarray:
    return np.clip(np.rint(tensor.squeeze().detach().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)


def load_models(device: torch.device) -> tuple[FSRCNNSubpixel, OriginalFSRCNNS, dict[str, float]]:
    light_checkpoint = torch.load(ROOT / "artifacts" / "model" / "fsrcnn_d16_s8_m1_c16_x2_fp32.pth", map_location=device, weights_only=False)
    light = FSRCNNSubpixel().to(device).eval()
    light.load_state_dict(light_checkpoint["state_dict"])
    original_checkpoint = torch.load(ROOT / "artifacts" / "original_fsrcnn_s" / "original_fsrcnn_s_x2_t91.pt", map_location=device, weights_only=False)
    original = OriginalFSRCNNS().to(device).eval()
    original.load_state_dict(original_checkpoint["state_dict"])
    scales = json.loads((ROOT / "artifacts" / "evaluation" / "summary.json").read_text(encoding="utf-8"))["activation_scales"]
    return light, original, {key: float(value) for key, value in scales.items()}


def normalize_hr(frame: np.ndarray) -> Image.Image:
    image = Image.fromarray(frame[:, :, :3]).convert("RGB")
    if image.size != HR_SIZE:
        image = image.resize(HR_SIZE, Image.Resampling.BICUBIC)
    return image


def combine_y_with_bicubic_chroma(lr_ycbcr: Image.Image, reconstructed_y: np.ndarray) -> np.ndarray:
    cbcr = lr_ycbcr.resize(HR_SIZE, Image.Resampling.BICUBIC)
    output = Image.merge("YCbCr", (Image.fromarray(reconstructed_y), cbcr.getchannel("Cb"), cbcr.getchannel("Cr"))).convert("RGB")
    return np.asarray(output)


def labelled_tile(frame: np.ndarray, text: str) -> Image.Image:
    image = Image.fromarray(frame).resize(LR_SIZE, Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, 960, 40), fill=(0, 0, 0))
    draw.text((18, 12), text, fill=(255, 255, 255), font=font)
    return image


def comparison_frame(source: np.ndarray, bicubic: np.ndarray, original: np.ndarray, light: np.ndarray) -> np.ndarray:
    canvas = Image.new("RGB", HR_SIZE)
    for index, (frame, name) in enumerate(((source, "Original 1080P"), (bicubic, "Bicubic x2"), (original, "Original FSRCNN-s x2"), (light, "Team lightweight INT8 x2"))):
        canvas.paste(labelled_tile(frame, name), ((index % 2) * 960, (index // 2) * 540))
    return np.asarray(canvas)


def metric_pair(reference_y: np.ndarray, candidate_y: np.ndarray, device: torch.device) -> tuple[float, float]:
    """Compute full-HD metrics on the same accelerator as model inference."""
    reference = tensor_from_y(reference_y, device)
    candidate = tensor_from_y(candidate_y, device)
    return psnr_y(reference, candidate), ssim_y(reference, candidate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a 540p->1080p three-model video comparison.")
    parser.add_argument("--input", type=Path, required=True, help="A native 1080p video is recommended.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to artifacts/video_demo/<input file name>.")
    parser.add_argument("--max-frames", type=int, default=150, help="0 means process all frames.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = ROOT / "artifacts" / "video_demo" / args.input.stem
    if args.max_frames < 0:
        raise ValueError("max-frames must be >= 0.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    light, original, scales = load_models(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reader = imageio.get_reader(args.input)
    metadata = reader.get_meta_data()
    fps = float(metadata.get("fps") or 30.0)
    writers = {
        "lr_540p": imageio.get_writer(args.output_dir / "input_540p.mp4", fps=fps, codec="libx264", quality=8, macro_block_size=1),
        "bicubic": imageio.get_writer(args.output_dir / "bicubic_1080p.mp4", fps=fps, codec="libx264", quality=8, macro_block_size=1),
        "original": imageio.get_writer(args.output_dir / "original_fsrcnn_s_1080p.mp4", fps=fps, codec="libx264", quality=8, macro_block_size=1),
        "light": imageio.get_writer(args.output_dir / "team_light_int8_1080p.mp4", fps=fps, codec="libx264", quality=8, macro_block_size=1),
        "comparison": imageio.get_writer(args.output_dir / "comparison_2x2_1080p.mp4", fps=fps, codec="libx264", quality=8, macro_block_size=1),
    }
    rows: list[dict[str, float | int]] = []
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for frame_index, frame in enumerate(reader):
                if args.max_frames and frame_index >= args.max_frames:
                    break
                hr_rgb = normalize_hr(np.asarray(frame))
                hr_ycbcr = hr_rgb.convert("YCbCr")
                hr_y = np.asarray(hr_ycbcr.getchannel("Y"), dtype=np.uint8)
                lr_ycbcr = hr_ycbcr.resize(LR_SIZE, Image.Resampling.BICUBIC)
                lr_y = np.asarray(lr_ycbcr.getchannel("Y"), dtype=np.uint8)
                bicubic_y = np.asarray(Image.fromarray(lr_y).resize(HR_SIZE, Image.Resampling.BICUBIC), dtype=np.uint8)
                lr_tensor = tensor_from_y(lr_y, device)
                original_y = y_from_tensor(original(lr_tensor).clamp(0.0, 1.0))
                light_y = y_from_tensor(quantized_forward_float(light, lr_tensor, scales))
                source_rgb = np.asarray(hr_rgb)
                lr_rgb = np.asarray(lr_ycbcr.convert("RGB"))
                bicubic_rgb = combine_y_with_bicubic_chroma(lr_ycbcr, bicubic_y)
                original_rgb = combine_y_with_bicubic_chroma(lr_ycbcr, original_y)
                light_rgb = combine_y_with_bicubic_chroma(lr_ycbcr, light_y)
                writers["lr_540p"].append_data(lr_rgb)
                writers["bicubic"].append_data(bicubic_rgb)
                writers["original"].append_data(original_rgb)
                writers["light"].append_data(light_rgb)
                writers["comparison"].append_data(comparison_frame(source_rgb, bicubic_rgb, original_rgb, light_rgb))
                bicubic_psnr, bicubic_ssim = metric_pair(hr_y, bicubic_y, device)
                original_psnr, original_ssim = metric_pair(hr_y, original_y, device)
                light_psnr, light_ssim = metric_pair(hr_y, light_y, device)
                rows.append({
                    "frame": frame_index,
                    "bicubic_psnr_db": bicubic_psnr,
                    "bicubic_ssim": bicubic_ssim,
                    "original_fsrcnn_s_psnr_db": original_psnr,
                    "original_fsrcnn_s_ssim": original_ssim,
                    "team_light_int8_psnr_db": light_psnr,
                    "team_light_int8_ssim": light_ssim,
                })
                if frame_index % 10 == 0:
                    print(f"frame {frame_index}: light INT8 PSNR={light_psnr:.3f} dB")
    finally:
        reader.close()
        for writer in writers.values():
            writer.close()
    if not rows:
        raise RuntimeError("No video frames were read.")
    fields = list(rows[0])
    average: dict[str, float | int | str] = {"frame": "AVERAGE"}
    average.update({field: float(np.mean([float(row[field]) for row in rows])) for field in fields[1:]})
    with (args.output_dir / "frame_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows + [average])
    elapsed = time.perf_counter() - started
    summary = {
        "input": str(args.input),
        "source_policy": "resized_to_1920x1080_if_needed",
        "input_to_model": "960x540 Y, Bicubic downsampling",
        "output": "1920x1080 Y with Bicubic Cb/Cr",
        "frames": len(rows),
        "video_fps": fps,
        "processing_fps": len(rows) / elapsed,
        "device": str(device),
        "metrics_average": average,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
