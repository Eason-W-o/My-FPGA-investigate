"""Make a clearly labelled synthetic 1080p motion clip from the existing Set5 images.

This is only a video-pipeline smoke-test source.  Use a real native 1080p video
with ``video_demo.py`` for competition results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]


def fit_cover(image: Image.Image, size: tuple[int, int], phase: float) -> Image.Image:
    """Apply a small deterministic pan/zoom while covering the 16:9 frame."""
    width, height = size
    zoom = 1.08 + 0.12 * phase
    scaled = image.resize((int(width * zoom), int(height * zoom)), Image.Resampling.BICUBIC)
    max_x, max_y = scaled.width - width, scaled.height - height
    left = int(max_x * phase)
    top = int(max_y * (1.0 - phase))
    return scaled.crop((left, top, left + width, top + height))


def label(frame: Image.Image, text: str) -> np.ndarray:
    draw = ImageDraw.Draw(frame)
    font = ImageFont.load_default()
    draw.rectangle((24, 24, 710, 76), fill=(0, 0, 0))
    draw.text((40, 40), text, fill=(255, 255, 255), font=font)
    return np.asarray(frame)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seconds-per-image", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "video_demo" / "synthetic_set5_source_1080p.mp4")
    args = parser.parse_args()
    if args.fps < 1 or args.seconds_per_image <= 0:
        raise ValueError("fps and seconds-per-image must be positive.")
    images = sorted((ROOT / ".data" / "set5").glob("*.png"))
    if not images:
        raise FileNotFoundError("Set5 images are unavailable.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames_per_image = max(1, round(args.fps * args.seconds_per_image))
    writer = imageio.get_writer(args.output, fps=args.fps, codec="libx264", quality=8, macro_block_size=1)
    try:
        for source in images:
            with Image.open(source) as original:
                image = original.convert("RGB")
            # First enlarge the still, then produce a motion crop. This is deliberately synthetic.
            base = image.resize((2304, 1296), Image.Resampling.BICUBIC)
            for index in range(frames_per_image):
                phase = index / max(frames_per_image - 1, 1)
                frame = fit_cover(base, (1920, 1080), phase)
                writer.append_data(label(frame, f"Synthetic Set5 motion source - {source.stem}"))
    finally:
        writer.close()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
