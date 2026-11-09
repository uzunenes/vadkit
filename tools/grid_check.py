"""Fit the native streaming grid branch (vadkit/grid/stream.py) on a grey cache and compare it with the archived
evaluate.py scores of the paper (pool48_mc4: I, pool48_mc5: Fs, Fd), then check streaming == batch on test videos.

    python tools/grid_check.py --cache /workspace/cache/ped2 --dataset ped2 \
        --run CK/runs/ped2_2d_noadv --archive CK/scores/runs/ped2_2d_noadv --state OUT/grid_ped2 --out OUT/check.json

Reports, per channel: max |raw difference| to the archive over the test frames t >= gap, the causal AUC of the refitted
and of the archived raw scores (Fisher calibration on each one's own reference, causal median 9, warmup 4), and
max |stream - batch| over the streamed videos. Nothing is tuned on the result.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from aean.data import load_gt, load_split
from vadkit.calibration.fisher import FisherCalibrator
from vadkit.eval.protocol import evaluate
from vadkit.grid.stream import GridBranch, GridConfig, GridState, batch_scores, fit_grid
from vadkit.temporal.median import causal_median

KEYS = {"I": ("mc4", "frx_bayes"), "Fs": ("mc5", "flow_ubm_m"), "Fd": ("mc5", "flow_rx_m")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--archive", required=True, help="scores/runs/<ds>_2d_noadv")
    ap.add_argument("--state", required=True, help="output (or existing) grid state directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stream-videos", type=int, default=2, help="test videos streamed frame by frame")
    a = ap.parse_args()
    cache, ds = Path(a.cache), a.dataset
    cfg = GridConfig.for_dataset(ds)
    res = {"dataset": ds, "device": a.device, "torch": torch.__version__, "host": platform.node()}
    if (Path(a.state) / "grid.json").exists():
        state = GridState.load(a.state)
        res["fit_s"] = None
    else:
        t0 = time.perf_counter()
        state = fit_grid(load_split(cache, "train"), a.run, cfg, a.device, a.workers)
        res["fit_s"] = round(time.perf_counter() - t0, 1)
        state.save(a.state)
    test, gt = load_split(cache, "test"), load_gt(cache)
    arch = {m: np.load(Path(a.archive) / f"eval_{ds}_pool48_{m}" / "scores.npz") for m in ("mc4", "mc5")}
    new = {}
    for v, video in test.items():
        new[v] = batch_scores(state, video, a.run, a.device)
    res["channels"] = {}
    vids = sorted(test)
    for ch, (m, key) in KEYS.items():
        z = arch[m]
        diffs = [np.nanmax(np.abs(new[v][ch][cfg.gap:] - z[f"{key}/{v}"][cfg.gap:])) for v in vids]
        ref_a, ref_n = z[f"ref/{key}"], state.ref[ch]
        cal_a, cal_n = FisherCalibrator().fit(ref_a), FisherCalibrator().fit(ref_n)
        sa = {v: cal_a(z[f"{key}/{v}"]) for v in vids}
        sn = {v: cal_n(np.nan_to_num(new[v][ch], nan=float(np.nanmin(new[v][ch])))) for v in vids}
        ea = evaluate(sa, gt, 4, lambda x: causal_median(x, 9))
        en = evaluate(sn, gt, 4, lambda x: causal_median(x, 9))
        res["channels"][ch] = {"max_abs_raw_diff_test": float(max(diffs)), "ref_len": [len(ref_a), len(ref_n)],
                               "ref_max_abs_diff": float(np.max(np.abs(np.sort(ref_a) - np.sort(ref_n))))
                               if len(ref_a) == len(ref_n) else None,
                               "causal_auc_archived": ea["causal"], "causal_auc_refit": en["causal"]}
    # fused I + Fs + Fd
    fuse = lambda d: {v: sum(d[c][v] for c in KEYS) for v in vids}
    za = {c: {v: FisherCalibrator().fit(arch[m][f"ref/{k}"])(arch[m][f"{k}/{v}"]) for v in vids}
          for c, (m, k) in KEYS.items()}
    zn = {c: {v: FisherCalibrator().fit(state.ref[c])(np.nan_to_num(new[v][c], nan=float(np.nanmin(new[v][c]))))
              for v in vids} for c in KEYS}
    res["grid_fused_causal_auc"] = {"archived": evaluate(fuse(za), gt, 4, lambda x: causal_median(x, 9))["causal"],
                                    "refit": evaluate(fuse(zn), gt, 4, lambda x: causal_median(x, 9))["causal"]}
    # streaming == batch
    br = GridBranch(state, a.run, a.device)
    worst = {c: 0.0 for c in KEYS}
    for v in vids[:a.stream_videos]:
        br.reset()
        rows = [br.step_gray(g) for g in test[v]]
        for c in KEYS:
            s = np.array([r.get(c, np.nan) for r in rows])
            worst[c] = max(worst[c], float(np.nanmax(np.abs(s[cfg.gap:] - new[v][c][cfg.gap:]))))
    res["stream_vs_batch_max_abs"] = worst
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
