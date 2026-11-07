"""python -m vadkit {extract,fit,score,eval,bench}; configs are YAML files or built-in names (vadkit/configs).

    python -m vadkit extract --config light_default --frames F/avenue --out FEAT/avenue [--train-every 5] [--shard 0/2]
    python -m vadkit fit     --config released_features --features FEAT/ped2 --out STATE [--scenes]
    python -m vadkit score   --config released_features --features FEAT/ped2 --state STATE --out scores.npz
    python -m vadkit eval    --config released_features --features FEAT/ped2 --gt GT [--state STATE] [--scenes] \
        --extra-channels RUNS/ped2_2d_noadv/eval_ped2_pool48_mc4/scores.npz:frx_bayes=I \
                         RUNS/ped2_2d_noadv/eval_ped2_pool48_mc5/scores.npz:flow_ubm_m=Fs,flow_rx_m=Fd
    python -m vadkit bench   --config light_default --frames F/ped2 --out bench.json [--state STATE]

--extra-channels: scores.npz files with '<key>/<video>' and 'ref/<key>' (grid/precomputed.py); '{sc}' in a path is
the ShanghaiTech scene (a template with no file for any scene raises, one missing some scenes warns). The grid channels
use the config's grid_calibrator (default: its calibrator, which the object branch keeps), or --grid-calibrator NAME or
a JSON spec; a group calibrator (fisher_hb, HB-POT: '{"name": "fisher_hb", "args": {"mode": "t1"}}') also needs
--extra-ref-ids, npz templates with 'ids/<key>' per reference value (tools/grid_ref_ids.py). score writes the
tools/objscore2.py format ('<attr>/<video>', 'ref/<attr>') of the state's channels (attributes fitted in every group,
as eval and the stream use them); with a per-scene state, --out must hold '{sc}' and one file per scene is written
(objscore2 --prefix, without the attributes some scene lacks, which tools/objfuse_sht.py drops anyway). The gap comes
from --dataset (default: the name of the frames directory) when the config gives one per dataset; a fitted state
records the gap of its features. --device (default: cuda when available) also scores a loaded state. extract defaults
--train-every to the config's and batches flow pairs; score / eval with --state read the testing split only.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import os

import cv2
import numpy as np
import torch

from vadkit.grid.precomputed import PrecomputedChannels
from vadkit.io.frames import frame_paths, read
from vadkit.perception.detector import default_device
from vadkit.pipeline.offline import FittedState, ObjectBranch, Perception, channel_order, gap_of, load_config
from vadkit.pipeline.stream import StreamingVAD


def cmd_extract(a, cfg: dict) -> None:
    """Write the feat2-format cache of every split."""
    branch = ObjectBranch(cfg, device=a.device)
    gap = gap_of(cfg, a.dataset or Path(a.frames).name)
    for split in a.splits:             # train_every None: the config's
        for p in branch.extract(a.frames, split, a.out, gap, a.train_every, a.shard):
            print(f"extract {split}/{p.stem}", flush=True)


def cmd_fit(a, cfg: dict) -> None:
    """Fit on the training videos and save the state."""
    branch = ObjectBranch(cfg, a.scenes, a.device)
    branch.fit(branch.load(a.features, ("training",))[0]).save(a.out)
    print(f"state -> {a.out}")


def cmd_score(a, cfg: dict) -> None:
    """Raw test frame scores and references of state.channels in the tools/objscore2.py format, one file per group."""
    branch = ObjectBranch(cfg, device=a.device)
    state = FittedState.load(a.state, default_device(a.device))
    if state.scenes and "{sc}" not in a.out:
        raise SystemExit(f"--out {a.out!r} must contain '{{sc}}' for a per-scene state")
    test = branch.load(a.features, ("testing",))[0]
    for g, attrs in state.groups.items():
        vids = [v for v in sorted(test) if state.group_of(v) == g]
        if not vids:
            continue
        raw = branch.score({v: test[v] for v in vids}, state)              # state.channels
        out = Path(a.out.format(sc=g)) if state.scenes else Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"ref/{n}": attrs[n].ref for n in raw}
        arrays.update({f"{n}/{v}": raw[n][v] for n in raw for v in vids})
        np.savez_compressed(out, **arrays)
        print(f"scores -> {out}")


def grid_calibrator(cfg: dict, arg: str | None = None) -> str | dict:
    """Calibrator spec of the grid channels: arg (a name or a JSON spec), else the config's grid_calibrator, else its
    calibrator (the object branch always uses cfg['calibrator'], default fisher)."""
    if arg:
        return json.loads(arg) if arg.lstrip().startswith("{") else arg
    return cfg.get("grid_calibrator", cfg.get("calibrator", "fisher"))


def cmd_eval(a, cfg: dict) -> dict:
    """Causal micro / macro and offline frame AUC of every channel and of the fused set."""
    gt = json.loads(Path(a.gt).read_text())
    vids = sorted(gt)
    branch = ObjectBranch(cfg, a.scenes, a.device)
    test = branch.load(a.features, ("testing",))[0]
    for v in vids:
        assert int(test[v]["n_frames"]) == len(gt[v]), v
    state = (FittedState.load(a.state, default_device(a.device)) if a.state
             else branch.fit(branch.load(a.features, ("training",))[0]))
    z_grid = PrecomputedChannels(a.extra_channels, grid_calibrator(cfg, a.grid_calibrator),
                                 a.extra_ref_ids).calibrated(vids, state.scenes)
    z_obj = branch.calibrate(branch.score({v: test[v] for v in vids}, state), state)
    res = branch.evaluate(channel_order(z_grid, z_obj, state.scenes), gt, a.warmup)
    for k, r in res.items():
        print(f"{k:24s} causal {r['causal']:.4f}  macro {r['causal_macro']:.4f}  offline {r['offline']:.4f}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    return res


def bench(cfg: dict, frames_root: str | Path, gap: int, state: FittedState | None = None, device: str | None = None,
          warm: int = 20, reps: int = 3, n: int = 100) -> dict:
    """Per-stage ms (median, p95) on the test frames in time order at batch 1: warm-up frames, then reps x n."""
    width = int(cfg.get("width", 640))
    perc = Perception(cfg, gap, device)
    perc.clock = {}
    stream = None
    if state is not None:                  # scoring cost does not depend on the scene
        stream = StreamingVAD(state, cfg, gap, next(iter(state.groups)) if state.scenes else None)
    seq = [(t, p) for paths in frame_paths(frames_root, "testing").values() for t, p in enumerate(paths)]
    if len(seq) < warm + reps * n:
        raise ValueError(f"{len(seq)} test frames, bench needs {warm + reps * n}")
    rows, nobj = [], []
    for t, p in seq[:warm + reps * n]:
        if t == 0:
            perc.reset()
            if stream is not None:
                stream.reset()
        s0 = time.perf_counter()
        bgr = read(p, width)
        s1 = time.perf_counter()
        objs = perc(bgr)
        s2 = time.perf_counter()
        if stream is not None:
            stream.step_objects(t, objs, bgr.shape[:2])
        s3 = time.perf_counter()
        rows.append({"decode": (s1 - s0) * 1e3, **perc.clock, **({"score": (s3 - s2) * 1e3} if stream else {}),
                     "total": (s3 - s0) * 1e3})
        nobj.append(len(objs))
    stat = lambda x: {"median_ms": float(np.median(x)), "p95_ms": float(np.percentile(x, 95)),
                      "p99_ms": float(np.percentile(x, 99))}
    timed = [rows[warm + i * n: warm + (i + 1) * n] for i in range(reps)]
    nt = nobj[warm: warm + reps * n]
    return {"config": cfg.get("name"), "parallel_flow": bool(cfg.get("parallel_flow", False)),
            "parallel_pose": bool(cfg.get("parallel_pose", False)),
            "device": device or ("cuda" if torch.cuda.is_available() else "cpu"),
            "gap": gap, "frame_hw": list(read(seq[0][1], width).shape[:2]), "warmup_frames": warm, "frames_per_rep": n,
            "reps": [{k: stat([r[k] for r in rep]) for k in rows[0]} for rep in timed],
            "objects_per_frame": {"mean": float(np.mean(nt)), "max": int(np.max(nt))},
            "timed_frames": {"total_ms": [r["total"] for rep in timed for r in rep], "objects": nt},
            "torch_threads": torch.get_num_threads(), "cv2_threads": cv2.getNumThreads()}


def cmd_bench(a, cfg: dict) -> None:
    """Latency JSON of bench()."""
    if a.parallel_flow:
        cfg = {**cfg, "parallel_flow": True}
    if a.parallel_pose:
        cfg = {**cfg, "parallel_pose": True}
    res = bench(cfg, a.frames, gap_of(cfg, a.dataset or Path(a.frames).name),
                FittedState.load(a.state, default_device(a.device)) if a.state else None, a.device, a.warm, a.reps, a.n)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


COMMANDS = {"extract": cmd_extract, "fit": cmd_fit, "score": cmd_score, "eval": cmd_eval, "bench": cmd_bench}


def main(argv: list[str] | None = None):
    """Parse argv and run the sub-command; returns its result (eval: the result dict)."""
    ap = argparse.ArgumentParser(prog="python -m vadkit")
    sp = ap.add_subparsers(dest="cmd", required=True)
    ps = {c: sp.add_parser(c) for c in COMMANDS}
    for p in ps.values():
        p.add_argument("--config", required=True, help="YAML file or built-in name (released_features, light_default)")
        p.add_argument("--device", default=None, help="default: cuda when available")
    for c in ("fit", "score", "eval"):
        ps[c].add_argument("--features", required=True, help="feat2-format dir with training/ and testing/")
    for c in ("extract", "bench"):
        ps[c].add_argument("--frames", required=True, help="<frames>/<split>/frames/<video>/<t>.jpg")
        ps[c].add_argument("--dataset", help="gap key of the config (default: name of the frames dir)")
    for c in ("fit", "eval"):
        ps[c].add_argument("--scenes", action="store_true", help="ShanghaiTech: models and calibration per scene")
    ps["extract"].add_argument("--out", required=True)
    ps["extract"].add_argument("--splits", nargs="+", default=["training", "testing"])
    ps["extract"].add_argument("--train-every", type=int, help="training subsample (default: the config's)")
    ps["extract"].add_argument("--shard", default="0/1", help="i/n: videos i::n of each split")
    ps["fit"].add_argument("--out", required=True, help="state directory")
    ps["score"].add_argument("--state", required=True)
    ps["score"].add_argument("--out", required=True)
    ps["eval"].add_argument("--gt", required=True)
    ps["eval"].add_argument("--state", help="fitted state (default: fit now)")
    ps["eval"].add_argument("--extra-channels", nargs="*", default=[], metavar="SCORES.npz[:key=name,...]")
    ps["eval"].add_argument("--extra-ref-ids", nargs="*", default=None, metavar="IDS.npz",
                            help="'ids/<key>' per reference value, one file or one per --extra-channels file")
    ps["eval"].add_argument("--grid-calibrator", default=None,
                            help="grid channels only: a calibrator name or JSON spec (default: config grid_calibrator)")
    ps["eval"].add_argument("--warmup", type=int, default=None, help="default: the config's (4)")
    ps["eval"].add_argument("--out", help="result JSON")
    ps["bench"].add_argument("--out", required=True)
    ps["bench"].add_argument("--state", help="also time the scoring stage")
    ps["bench"].add_argument("--warm", type=int, default=20)
    ps["bench"].add_argument("--reps", type=int, default=3)
    ps["bench"].add_argument("--n", type=int, default=100)
    ps["bench"].add_argument("--parallel-flow", action="store_true",
                             help="flow in a worker thread, overlapping detector / pose / encoder (same outputs)")
    ps["bench"].add_argument("--parallel-pose", action="store_true",
                             help="pose model in a worker thread, overlapping the detector (same outputs)")
    a = ap.parse_args(argv)
    if os.environ.get("VADKIT_CV2_THREADS"):       # parallel shards: one or two OpenCV threads each
        cv2.setNumThreads(int(os.environ["VADKIT_CV2_THREADS"]))
    return COMMANDS[a.cmd](a, load_config(a.config))
