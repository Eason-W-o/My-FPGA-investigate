"""Generate reproducible bandwidth tables for the 540p-to-1080p video path."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


VIDEO_MODES = (
    ("540p30", 960, 540, 30),
    ("720p60", 1280, 720, 60),
    ("1080p30", 1920, 1080, 30),
    ("1080p60", 1920, 1080, 60),
    ("1440p30", 2560, 1440, 30),
    ("4K30", 3840, 2160, 30),
    ("4K60", 3840, 2160, 60),
    ("8K30", 7680, 4320, 30),
    ("8K60", 7680, 4320, 60),
)

PIXEL_FORMATS = (
    ("Y8", 1.0),
    ("YCbCr420_8bit", 1.5),
    ("YCbCr422_8bit", 2.0),
    ("RGB888", 3.0),
)

MODEL_LAYERS = (
    ("feature_5x5_1to16", 207_360_000),
    ("shrink_1x1_16to8", 66_355_200),
    ("mapping_3x3_8to8", 298_598_400),
    ("expand_1x1_8to16", 66_355_200),
    ("subpixel_5x5_16to4", 829_440_000),
)


def active_payload_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for mode, width, height, fps in VIDEO_MODES:
        pixels_per_second = width * height * fps
        for pixel_format, bytes_per_pixel in PIXEL_FORMATS:
            bytes_per_frame = width * height * bytes_per_pixel
            bytes_per_second = pixels_per_second * bytes_per_pixel
            rows.append(
                {
                    "mode": mode,
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "pixel_format": pixel_format,
                    "bytes_per_pixel": bytes_per_pixel,
                    "active_mpixel_per_s": pixels_per_second / 1e6,
                    "frame_mb": bytes_per_frame / 1e6,
                    "active_payload_mb_per_s": bytes_per_second / 1e6,
                    "active_payload_gbit_per_s": bytes_per_second * 8 / 1e9,
                }
            )
    return rows


def project_summary() -> dict[str, object]:
    in_pixels = 960 * 540
    out_pixels = 1920 * 1080
    fps = 30
    total_macs = sum(macs for _, macs in MODEL_LAYERS)

    frame_path = {}
    for pixel_format, bytes_per_pixel in PIXEL_FORMATS:
        input_bw = in_pixels * fps * bytes_per_pixel
        output_bw = out_pixels * fps * bytes_per_pixel
        frame_path[pixel_format] = {
            "input_active_mb_per_s": input_bw / 1e6,
            "output_active_mb_per_s": output_bw / 1e6,
            "stream_sum_mb_per_s": (input_bw + output_bw) / 1e6,
            "ddr_write_read_roundtrip_mb_per_s": 2 * (input_bw + output_bw) / 1e6,
            "ddr_with_1080p60_repeat_read_mb_per_s": (
                2 * input_bw + output_bw + 2 * output_bw
            )
            / 1e6,
        }

    feature_bytes = {
        "feature_16ch_int16": in_pixels * 16 * 2,
        "shrink_8ch_int16": in_pixels * 8 * 2,
        "mapping_8ch_int16": in_pixels * 8 * 2,
        "expand_16ch_int16": in_pixels * 16 * 2,
        "subpixel_4ch_uint8": in_pixels * 4,
    }
    feature_total = sum(feature_bytes.values())

    line_buffers = {
        "feature_5x5_y8_4_lines": 4 * 960,
        "mapping_3x3_8ch_int16_2_lines": 2 * 960 * 8 * 2,
        "subpixel_5x5_16ch_int16_4_lines": 4 * 960 * 16 * 2,
    }

    return {
        "project": "960x540 Y8 to 1920x1080 Y8 at 30 fps",
        "frame_path": frame_path,
        "model": {
            "mac_per_frame": total_macs,
            "gmac_per_s_at_30fps": total_macs * fps / 1e9,
            "mac_per_lr_pixel": total_macs / in_pixels,
            "ideal_mac_units_by_clock_mhz": {
                str(clock): total_macs * fps / (clock * 1e6)
                for clock in (100, 150, 180, 200)
            },
            "layers": [
                {
                    "name": name,
                    "mac_per_frame": macs,
                    "share_percent": 100 * macs / total_macs,
                    "gmac_per_s_at_30fps": macs * fps / 1e9,
                }
                for name, macs in MODEL_LAYERS
            ],
        },
        "intermediate_feature_maps": {
            "bytes_per_frame_if_each_map_written_once": feature_total,
            "mb_per_frame_if_each_map_written_once": feature_total / 1e6,
            "ddr_roundtrip_gb_per_s_at_30fps": 2 * feature_total * fps / 1e9,
            "maps": feature_bytes,
        },
        "minimum_line_buffers": {
            "total_bytes": sum(line_buffers.values()),
            "total_kib": sum(line_buffers.values()) / 1024,
            "buffers": line_buffers,
            "note": "Excludes border handling, FIFOs, banking, alignment, and timing buffers.",
        },
        "ddr3_x32_theoretical_examples_gb_per_s": {
            "800_MTps": 3.2,
            "1066_MTps": 4.264,
            "1600_MTps": 6.4,
        },
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/video_bandwidth"),
    )
    args = parser.parse_args()

    rows = active_payload_rows()
    summary = project_summary()
    write_csv(args.output_dir / "active_payload_bandwidth.csv", rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "project_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Generated: {args.output_dir / 'active_payload_bandwidth.csv'}")
    print(f"Generated: {args.output_dir / 'project_summary.json'}")
    print(f"Required model throughput: {summary['model']['gmac_per_s_at_30fps']:.6f} GMAC/s")


if __name__ == "__main__":
    main()
