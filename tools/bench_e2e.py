"""End-to-end per-frame latency of the streaming detector, object branch alone or object + grid, at batch size one
(pre-registration: docs/tez_notlari/birlesik_hiz_on_kayit_2026-11-08.md).

    python tools/bench_e2e.py --config vadkit/configs/r2/base_trt.yaml --frames /workspace/frames/avenue \
        --dataset avenue --state STATE [--grid-state GSTATE --grid-run CK/runs/avenue_2d_noadv] --out OUT.json

Per frame, from reading the JPEG to the fused online score: cv2.imread, StreamingVAD.step (object perception, the grid
branch on the same frame when given, calibration, sum, causal median); the score is a Python float, so the GPU work
of the frame has finished when the clock stops. Not instrumented (no per-stage synchronisation). Test frames in time
order across videos, reset at each video start; `warm` frames, then reps x n timed frames.

Before the run the GPU is sampled for --idle-secs (nvidia-smi every 100 ms, plus the compute processes and
/proc/loadavg); the run starts only when every sample is <= 5 % utilisation and no compute process is listed, after
up to --idle-tries attempts 60 s apart; every attempt is recorded, and a run that never saw an idle GPU is still run
and flagged (idle_ok false). The GPU is also sampled during the run (informational).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from vadkit.grid.stream import GridBranch, GridState
from vadkit.io.frames import frame_paths, read
from vadkit.pipeline.offline import FittedState, gap_of, load_config
from vadkit.pipeline.stream import StreamingVAD

SMI = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"]


def smi(args: list[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def loadavg() -> list[float]:
    return [float(x) for x in Path("/proc/loadavg").read_text().split()[:3]]


def idle_check(secs: float) -> dict:
    """Utilisation / memory samples every 100 ms for secs, the compute processes and the load average."""
    t_end, samples = time.time() + secs, []
    while time.time() < t_end:
        line = smi(SMI).splitlines()
        if line:
            u, m = (float(x) for x in line[0].split(","))
            samples.append([u, m])
        time.sleep(0.1)
    apps = smi(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
    apps = [a for a in apps.splitlines() if a.strip() and not a.startswith(f"{os.getpid()},")]
    util = [s[0] for s in samples]
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "samples": samples, "max_util": max(util, default=None),
            "compute_apps": apps, "loadavg": loadavg(),
            "idle": bool(samples) and max(util) <= 5 and not apps}


class Sampler(threading.Thread):
    """GPU utilisation during the run (every 200 ms), informational."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples, self.stop = [], threading.Event()

    def run(self):
        while not self.stop.is_set():
            line = smi(SMI).splitlines()
            if line:
                self.samples.append([float(x) for x in line[0].split(",")])
            self.stop.wait(0.2)


def env(script: Path) -> dict:
    cpu = next((l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                if l.startswith("model name")), platform.processor())
    import ultralytics
    return {"host": platform.node(), "cpu": cpu, "affinity_cpus": len(os.sched_getaffinity(0)),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "driver": smi(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
            "torch": torch.__version__, "cuda": torch.version.cuda, "opencv": cv2.__version__,
            "ultralytics": ultralytics.__version__, "python": platform.python_version(),
            "torch_threads": torch.get_num_threads(), "cv2_threads": cv2.getNumThreads(),
            "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--frames", required=True, help="<frames>/testing/frames/<video>/<t>.jpg")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--state", required=True, help="fitted object-branch state (python -m vadkit fit)")
    ap.add_argument("--grid-state", help="fitted grid state (tools/grid_check.py / fit_grid); object + grid mode")
    ap.add_argument("--grid-run", help="autoencoder run of the grid state (model.pt, args.json)")
    ap.add_argument("--parallel-flow", action="store_true", help="object-branch DIS in a worker thread")
    ap.add_argument("--parallel-grid", action="store_true", help="grid branch in a worker thread (own DIS thread)")
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--idle-secs", type=float, default=5.0)
    ap.add_argument("--idle-tries", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = load_config(a.config)
    cfg = {**cfg, "parallel_flow": a.parallel_flow, "parallel_grid": a.parallel_grid}
    gap = gap_of(cfg, a.dataset)
    state = FittedState.load(a.state, a.device)
    grid = None
    if a.grid_state:
        grid = GridBranch(GridState.load(a.grid_state), a.grid_run, a.device, parallel_flow=a.parallel_grid)
    sv = StreamingVAD(state, cfg, gap, next(iter(state.groups)) if state.scenes else None, a.device, grid=grid)
    seq = [(t, p) for paths in frame_paths(a.frames, "testing").values() for t, p in enumerate(paths)]
    total = a.warm + a.reps * a.n
    if len(seq) < total:
        raise SystemExit(f"{len(seq)} test frames, the bench needs {total}")

    attempts = []
    for i in range(a.idle_tries):
        attempts.append(idle_check(a.idle_secs))
        if attempts[-1]["idle"] or i == a.idle_tries - 1:
            break
        time.sleep(60)
    sampler = Sampler()
    sampler.start()
    ms, scores = [], []
    wall0 = time.perf_counter()
    for t, p in seq[:total]:
        if t == 0:
            sv.reset()
        s0 = time.perf_counter()
        r = sv.step(read(p, 10 ** 6))                 # original frame; step() resizes for the object branch
        ms.append((time.perf_counter() - s0) * 1e3)
        scores.append(r.score)
    wall = time.perf_counter() - wall0
    sampler.stop.set()
    sampler.join()
    load_after = loadavg()

    stat = lambda x: {"median_ms": float(np.median(x)), "p95_ms": float(np.percentile(x, 95)),
                      "p99_ms": float(np.percentile(x, 99)), "max_ms": float(np.max(x))}
    timed = [ms[a.warm + i * a.n: a.warm + (i + 1) * a.n] for i in range(a.reps)]
    flat = [x for rep in timed for x in rep]
    rep_med = [float(np.median(r)) for r in timed]
    res = {"mode": "object+grid" if grid else "object", "config": cfg.get("name"), "config_file": a.config,
           "dataset": a.dataset, "gap": gap, "device": a.device, "parallel_flow": a.parallel_flow,
           "parallel_grid": a.parallel_grid, "warmup_frames": a.warm, "frames_per_rep": a.n,
           "reps": [{"total": stat(r)} for r in timed], "all": stat(flat),
           "fps_median": 1000.0 / float(np.median(rep_med)), "fps_reps": [1000.0 / m for m in rep_med],
           "timed_frames": {"total_ms": flat},
           "idle_attempts": attempts, "idle_ok": attempts[-1]["idle"],
           "during_run_util": {"n": len(sampler.samples),
                               "median": float(np.median([s[0] for s in sampler.samples])) if sampler.samples else None,
                               "max": float(max(s[0] for s in sampler.samples)) if sampler.samples else None},
           "loadavg_before": attempts[-1]["loadavg"], "loadavg_after": load_after,
           "wall_s_all_frames": wall, "frame_hw_original": list(read(seq[0][1], 10 ** 6).shape[:2]),
           "grid_state": a.grid_state, "env": env(Path(__file__))}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("mode", "dataset", "fps_median", "all", "idle_ok")}))


if __name__ == "__main__":
    main()
