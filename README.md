# vadkit

Code and evidence for the paper:

> E. Uzun and K. Kayabol, "Light, Modular and Online: Object Attributes and Bayesian Cell Statistics for Video Anomaly Detection," under review at *IEEE Access*.

`vadkit` is a lightweight video anomaly detector for fixed cameras. It learns from normal footage of a scene, scores frames causally as they arrive (delay $\le 4$ frames from a 9-frame causal median), and requires no test labels.

- **Object branch (training-free, 16.9 M parameters):** YOLO11n + motion-contour boxes, YOLO11n-pose keypoints, DIS optical flow, and MobileCLIP2-S0 crop embeddings. Each object is represented by velocity, pose, and appearance, and scored with normal–inverse-Wishart (NIW) mixture densities selected on held-out normal video.
- **Grid branch (optional, +1.0 M parameters):** A small scene autoencoder (`I`) and two optical-flow background models (`Fs`, `Fd`) scored in fixed $16\times 16$ cells with statistics shrunk towards the scene, following the hyperspectral background models of Arısoy and Kayabol.
- **Fusion:** Per-channel calibration against normal reference scores (`-log p`), summed across channels and smoothed causally.

## Results

Frame-level micro AUC (%). *Standard* uses per-video min–max normalisation and Gaussian smoothing; *online* is strictly causal with no test-video statistics.

| Configuration | Params | Ped1 std / online | Ped2 std / online | Avenue std / online | ShanghaiTech std / online |
|---|---:|---:|---:|---:|---:|
| Object branch | 16.9 M | 74.6 / 70.9 | 89.5 / 93.9 | 87.9 / 82.8 | 78.4 / 79.5 |
| Object + grid | 17.9 M | 79.4 / 76.4 | 96.7 / 96.5 | 89.8 / 87.4 | 76.9 / 70.2 |

On ShanghaiTech, a post-hoc check shows that the grid drop comes from the per-scene residual channel `I` (55.8% online alone): keeping only the training-free flow-grid channels (`Fs + Fd`) with the object branch gives 78.5% online on ShanghaiTech (+8.3 pts) while keeping Ped1, Ped2, and Avenue at 76.4%, 96.1%, and 87.6%.

**End-to-end streaming speed (batch size 1, RTX 3090, TensorRT detectors):**
- First measurement session (object branch, Avenue): **25.7 FPS** (19.6 FPS in PyTorch).
- Combined-streaming session (object branch + native streaming grid branch in parallel threads): **56.7 FPS** object-only and **55.7 FPS** object + grid on Avenue (17.9 ms median, 28.1 ms p99); **40.7 FPS** object-only and **34.3 FPS** object + grid on Ped2.

## Installation

Requires Python 3.11+ and PyTorch.

```bash
git clone https://github.com/uzunenes/vadkit.git
cd vadkit
python -m venv .venv && . .venv/bin/activate
pip install torch torchvision
pip install -r requirements.txt
```

First run downloads YOLO11n, YOLO11n-pose (Ultralytics, AGPL-3.0) and MobileCLIP2-S0 (Hugging Face). Check their licenses before commercial use.

## Quick start on your own camera

Place normal training frames under `<root>/training/frames/<video>/<t>.jpg` (`t = 0 ... n-1`), copy `vadkit/configs/r2/base.yaml`, and set `gap:` (1 at 10 fps, 2 at 25 fps).

```bash
# 1. Extract object features and fit NIW densities (no network training)
python -m vadkit extract --config mycam.yaml --frames <root> --splits training --out feat/mycam
python -m vadkit fit     --config mycam.yaml --features feat/mycam --out state/mycam
```

Score a video file, frame directory, or live camera stream one frame at a time:

```python
from vadkit.grid import StreamingGrid
from vadkit.io.frames import VideoSource
from vadkit.pipeline import FittedState, StreamingVAD

state = FittedState.load("state/mycam")
grid = StreamingGrid.load("state/mycam_grid")   # optional; omit for object-only
vad = StreamingVAD(state, grid=grid)

for t, frame in VideoSource("test.mp4"):
    r = vad.step(frame)
    if r.score is not None:                     # None during the 4-frame causal median warm-up
        print(t, f"{r.score:.3f}", r.channels)
```

## Reproducing the paper

### 1. Tables and macros from the evidence files (no GPU or datasets required)

```bash
python3 paper/ieee_access/fig_src/make_tables_v3.py
python3 paper/ieee_access/fig_src/make_hakem_v3.py
python3 paper/ieee_access/fig_src/make_text_v3.py
python3 paper/ieee_access/fig_src/make_birlesik_v4.py
python3 paper/ieee_access/fig_src/fig_size_accuracy.py
python3 paper/ieee_access/fig_src/fig_data_efficiency.py
git diff --exit-code paper/ieee_access/gen
```

Every number in the paper is generated from `paper/ieee_access/kanit/` and traced in `paper/ieee_access/gen/*_provenance.txt`.

### 2. Full evaluation from datasets

```bash
# Object branch
python -m vadkit extract --config vadkit/configs/r2/base.yaml --frames frames/ped2 --dataset ped2 --out feat/base/ped2
python -m vadkit eval    --config vadkit/configs/r2/base.yaml --features feat/base/ped2 --gt cache/ped2/gt.json --out res/base/ped2_obj.json

# Combined streaming evaluation and end-to-end latency benchmark
python tools/grid_check.py  --dataset avenue --cache cache/avenue --ckpt runs/avenue_2d_noadv/best.pt --frames frames/avenue --gt cache/avenue/gt.json --out out/grid
python tools/stream_eval.py --dataset avenue --config vadkit/configs/r2/base.yaml --state out/state/avenue --grid-state out/grid/state_avenue --frames frames/avenue --gt cache/avenue/gt.json --out out/stream/avenue
python tools/bench_e2e.py   --dataset avenue --config vadkit/configs/r2/base_trt.yaml --state out/state/avenue --grid-state out/grid/state_avenue --frames frames/avenue --mode object+grid --parallel-flow --out out/bench/avenue.json
```

Run tests with:

```bash
pytest -q tests
```

## Repository structure

- `vadkit/` — core package: perception (`detector`, `pose`, `flow`, `embed`), attributes, NIW density, calibration, fusion, causal median, `StreamingGrid`, offline & streaming pipelines.
- `aean/`, `train.py`, `evaluate.py` — grid autoencoder training and offline cell evaluation.
- `tools/` — streaming grid verification (`grid_check.py`), end-to-end benchmark (`bench_e2e.py`), combined stream evaluation (`stream_eval.py`), and post-hoc flow-grid diagnosis (`flow_grid_diag.py`).
- `paper/ieee_access/{fig_src,gen,kanit}/` — table/figure generators, generated LaTeX macros/tables, and raw experimental evidence.
- `docs/tez_notlari/` — pre-registration records (`*_on_kayit_*.md`).

## Citation & Contact

```bibtex
@misc{uzun2026vadkit,
  author = {Uzun, Enes and Kayabol, Koray},
  title  = {Light, Modular and Online: Object Attributes and Bayesian Cell Statistics for Video Anomaly Detection},
  note   = {Under review at IEEE Access},
  year   = {2026},
  url    = {https://github.com/uzunenes/vadkit}
}
```

For questions, contact Enes Uzun (`uzun.ens@gmail.com`). Released under the MIT License (`LICENSE`).
