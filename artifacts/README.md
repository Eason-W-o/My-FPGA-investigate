# Artifact Index

This directory keeps only final, inspectable, or hardware-useful artifacts. One-off installation logs, terminal logs, and smoke-test-only folders are intentionally excluded.

| Path | Contents | Purpose |
| --- | --- | --- |
| `evaluation/` | Lightweight-model and Bicubic Set5 metrics | Lightweight quality baseline |
| `model/` | FP32 lightweight checkpoint, training log, model contract | Software inference and quantization source |
| `original_fsrcnn_s/` | Original FSRCNN-s checkpoint, 50-epoch log, Set5 metrics | Paper-reproduction evidence |
| `standard_fsrcnn/` | Standard FSRCNN(56,12,4) checkpoint, 50-epoch log, Set5 metrics | Higher-cost paper-reproduction evidence |
| `model_comparison.csv` / `.md` | Same-protocol comparison of three methods | PPT/report citation |
| `video_demo/` | 540P-to-1080P software video demo | Demo and pipeline validation |
| `quant/` | INT8 weights, INT32 biases, Q1.15 PReLU, Q31 requantization data | Verilog/RTL initialization |
| `test_vectors/` | zero, impulse, ramp, random layer-by-layer small vectors | RTL unit-simulation comparison |
| `full_reference/` | FP32/fake-quant full-frame outputs | Software checks |
| `full_integer_golden/` | Full-integer full-frame 540P-to-1080P Golden | Final byte-level hardware validation |

## Video Demo

`video_demo/synthetic_set5/` is the retained 15-frame synthetic example and contains:

- `input_540p.mp4`: simulated model input;
- `bicubic_1080p.mp4`: interpolation baseline;
- `original_fsrcnn_s_1080p.mp4`: original FSRCNN-s software output;
- `team_light_int8_1080p.mp4`: lightweight INT8 model output;
- `comparison_2x2_1080p.mp4`: four-way comparison;
- `frame_metrics.csv`, `summary.json`: per-frame/average metrics and runtime information.

It validates the pipeline only and is not a substitute for a formal evaluation on a native 1080P video or an offline HR-ground-truth dataset.
