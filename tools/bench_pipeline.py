"""Per-frame latency of the full final pipeline, frame by frame as on a live camera (batch 1).

    python tools/bench_pipeline.py --data /workspace/cache/ped2 --run runs/ped2_2d_noadv \
        --reference-dir checkpoints/scores/runs/ped2_2d_noadv --out latency.json

Stages timed per frame: DIS optical flow (preset medium, 1 thread) + cell flow features, AE 2d reconstruction with
the evaluation patch stride + FRX-Bayes map, flow-speed GMM-UBM map, flow-histogram RX map, frame scores and Fisher
fusion. Models are fitted on a subset of the normal training frames first (not timed); only latency is measured here.
"""
import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aean.channels import CellRX, CellUBM, flow_video, frame_scores_from_maps, speed  # noqa: E402
from aean.data import load_split, to_unit  # noqa: E402
from aean.models import build  # noqa: E402
from aean.scoring import BackgroundStats, reconstruct, frame_score  # noqa: E402
from fuse_runs import normaliser  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--gap", type=int, default=1)
    ap.add_argument("--cell", type=int, default=8)
    ap.add_argument("--stride", type=int, default=8, help="AE patch stride, matching evaluate.py")
    ap.add_argument("--pool", type=int, default=48)
    ap.add_argument("--reference-dir", type=Path, required=True,
                    help="saved run containing cross-fitted normal references in eval_*_pool48_mc4/mc5")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    # GPU runs synchronise CUDA before every timestamp (added 6 Nov 2026, hakem_duzeltmeleri_on_kayit G1); the CPU path
    # is unchanged. Optical flow (OpenCV DIS) runs on the CPU in both cases.
    sync = torch.cuda.synchronize if args.device.startswith("cuda") else (lambda: None)
    if min(args.frames, args.repeats, args.gap, args.threads, args.cell, args.pool, args.stride) < 1:
        ap.error("frames, repeats, gap, threads, cell, pool and stride must be positive")
    torch.set_num_threads(args.threads)
    import cv2
    cv2.setNumThreads(1)
    dev = args.device
    cfg = json.loads((Path(args.run) / "args.json").read_text())
    if cfg["variant"] != "2d" or args.pool != 48:
        ap.error("this runner supports the final 2d model and pool48 calibration only")
    ae, _, _ = build(cfg["variant"], cfg["T"])
    ae.load_state_dict(torch.load(Path(args.run) / "model.pt", map_location=dev, weights_only=True)["ae"])
    ae = ae.to(dev).eval()
    P = cfg["patch"]
    if args.stride > P:
        ap.error("stride must not exceed the patch size")
    train = load_split(Path(args.data), "train")
    test = load_split(Path(args.data), "test")
    vid = next(iter(test.values()))
    H, W = vid.shape[1:]
    if args.gap + args.frames + 5 > len(vid):
        ap.error("requested frames plus warmup exceed the first test video length")
    ds = Path(args.data).name
    with np.load(args.reference_dir / f"eval_{ds}_pool48_mc4/scores.npz", allow_pickle=False) as s:
        calibrate_ae = normaliser(s["ref/frx_bayes"], "fisher")
    with np.load(args.reference_dir / f"eval_{ds}_pool48_mc5/scores.npz", allow_pickle=False) as s:
        calibrate_ubm = normaliser(s["ref/flow_ubm_m"], "fisher")
        calibrate_rx = normaliser(s["ref/flow_rx_m"], "fisher")

    # fit (not timed): residual stats, flow channels on a subset of normal frames
    stats = BackgroundStats(1, H, W, P, dev)
    tr = [v[::10] for v in train.values()]
    with torch.no_grad():
        for v in tr:
            vt = to_unit(torch.from_numpy(v).to(dev))
            for t in range(len(vt)):
                x = vt[t:t + 1]
                stats.update(x - reconstruct(ae, x, "2d", P, args.stride))
    stats.finalize()
    feats = [flow_video(v[:60], gap=args.gap, cell=args.cell, preset="medium") for v in train.values()]
    ubm = CellUBM(dev).fit([speed(f) for f in feats])
    C, gy, gx = feats[0].shape[1:]
    rx = CellRX(C, gy, gx, dev).fit(feats)

    repetitions = []
    with torch.no_grad():
        for repeat in range(args.repeats):
            times = {"flow": [], "ae_frx": [], "flow_models": [], "total": []}
            history = deque(maxlen=9)
            for t in range(args.gap, args.gap + args.frames + 5):
                sync()
                t0 = time.perf_counter()
                pair = vid[t - args.gap:t + 1:args.gap]
                f = flow_video(pair, gap=1, cell=args.cell, preset="medium")[-1:]
                sync()
                t1 = time.perf_counter()
                x = to_unit(torch.from_numpy(vid[t:t + 1]).to(dev))
                d = x - reconstruct(ae, x, "2d", P, args.stride)
                m = stats.bayes_delta(d).reshape(H, W)
                s_ae = frame_score(m, args.pool)
                sync()
                t2 = time.perf_counter()
                s_u = frame_scores_from_maps(ubm.maps(speed(f)), H, W, args.pool)
                s_r = frame_scores_from_maps(rx.maps(f), H, W, args.pool)
                fused = float(calibrate_ae(np.asarray(s_ae)) + calibrate_ubm(s_u)[0] + calibrate_rx(s_r)[0])
                if not history:
                    history.extend([fused] * 8)
                history.append(fused)
                _ = np.median(history)
                sync()
                t3 = time.perf_counter()
                if t >= args.gap + 5:
                    times["flow"].append(t1 - t0)
                    times["ae_frx"].append(t2 - t1)
                    times["flow_models"].append(t3 - t2)
                    times["total"].append(t3 - t0)
            row = {k: round(1000 * float(np.median(v)), 2) for k, v in times.items()}
            row.update({"repeat": repeat, "fps": round(1 / float(np.median(times["total"])), 2),
                        "p95_total_ms": round(1000 * float(np.quantile(times["total"], 0.95)), 2),
                        "frames": len(times["total"]), "samples_ms": {
                            k: [1000 * x for x in v] for k, v in times.items()}})
            repetitions.append(row)
    res = {"device": dev, "threads": args.threads, "opencv_threads": 1, "frame": f"{H}x{W}",
           "repetitions": repetitions, "ae_parameters": sum(p.numel() for p in ae.parameters()),
           "reference_dir": str(args.reference_dir),
           "includes": "flow, AE, statistical maps, normal-reference tail calibration, fusion, causal median",
           "excludes": "training, background fitting, file decoding, network transport",
           "background_fit": "subsampled normal train frames; runtime experiment, not an accuracy re-evaluation"}
    res.update({"created_at": datetime.now(timezone.utc).isoformat(), "arguments": vars(args) | {
                    "reference_dir": str(args.reference_dir), "out": str(args.out) if args.out else None},
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "cpu": next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                            if line.startswith("model name")),
                "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                "opencv": cv2.__version__, "precision": "float32 model; flow features float16",
                "warmup_frames_per_repeat": 5, "test_video": next(iter(test))})
    text = json.dumps(res, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
