"""Deployment measurements M1 / M2 of docs/tez_notlari/kurulum_olcumleri_on_kayit_2026-11-05.md (object branch).

  far      label-free alarm threshold: tau_alpha = (1 - alpha) quantile of the cross-fitted normal training reference,
           fused and smoothed exactly like the test scores, one threshold per camera (dataset, or ShT scene); realised
           false-alarm rate on test normal frames and detection rate on anomalous frames, video bootstrap CI.
           Also the analytic Gamma(k, 1) threshold (independent channels), for the object branch and grid + object.
  dataeff  normal-data efficiency: fit on max(2, ceil(f * N)) training videos (per scene on ShT), videos drawn with
           default_rng(1000 + seed); causal micro / macro AUC of the fused object branch on the full test set; fit time.

Run from the repo root, e.g.
  PYTHONPATH=. python vad_bench/vast/runs_r2/deploy_measures.py far --config C --features F --gt G [--scenes] \
      [--extra-channels ...] --out J
  PYTHONPATH=. python vad_bench/vast/runs_r2/deploy_measures.py dataeff --config C --features F --gt G [--scenes] \
      --frac 0.25 --seeds 0 1 2 3 4 --out J
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import gamma

from vadkit.density.niw import scene_of
from vadkit.grid.precomputed import PrecomputedChannels
from vadkit.pipeline.offline import ObjectBranch, channel_order, load_config, offline_smoother

ALPHAS = (0.1, 0.05, 0.01, 0.001)
FPS = {"ped1": 10, "ped2": 10, "avenue": 25, "shanghaitech": 24}


def fused_reference(state, train: dict, group: str) -> list[np.ndarray]:
    """Per training video of the group, the fused calibrated cross-fitted reference in time order (kept frames).

    fit_attribute concatenates the reference of folds (names[0::2] fit, names[1::2] scored) then the reverse, each
    video with its frames t % train_every == 0 (drop_startup is off in the measured configs)."""
    if bool(state.config.get("drop_startup", False)):
        raise ValueError("fused_reference assumes drop_startup: false")
    every = int(state.config.get("train_every", 1))
    names = sorted(v for v in train if (scene_of(v) if state.scenes else "all") == group)
    order = names[1::2] + names[0::2]
    lengths = [math.ceil(int(train[v]["n_frames"]) / every) for v in order]
    tot = None
    for a in state.channels:
        st = state.groups[group][a]
        assert len(st.ref) == sum(lengths), (group, a, len(st.ref), sum(lengths))
        z = st.calibrator(st.ref)
        tot = z if tot is None else tot + z
    return np.split(tot, np.cumsum(lengths)[:-1])


def smooth_all(seqs, smooth, drop: int) -> list[np.ndarray]:
    return [smooth(s[drop:]) for s in seqs]


def rates(test_s: dict, gt: dict, warmup: int, smooth, tau_of) -> dict:
    """Per video (count of normal frames over threshold, normal frames, anomalous over, anomalous)."""
    out = {}
    for v, s in test_s.items():
        y = np.asarray(gt[v])[warmup:]
        sc = smooth(s[warmup:])
        hit = sc > tau_of(v)
        out[v] = (int((hit & (y == 0)).sum()), int((y == 0).sum()), int((hit & (y == 1)).sum()), int((y == 1).sum()))
    return out


def summarise(per: dict, alpha: float, reps: int = 2000, seed: int = 0) -> dict:
    vids = sorted(per)
    A = np.array([per[v] for v in vids], float)
    far = A[:, 0].sum() / max(A[:, 1].sum(), 1)
    tpr = A[:, 2].sum() / max(A[:, 3].sum(), 1)
    rng = np.random.default_rng(seed)
    bf = []
    for _ in range(reps):
        B = A[rng.integers(0, len(vids), len(vids))]
        if B[:, 1].sum():
            bf.append(B[:, 0].sum() / B[:, 1].sum())
    ci = [float(np.percentile(bf, 2.5)), float(np.percentile(bf, 97.5))]
    return {"alpha": alpha, "far": float(far), "far_ci95": ci, "far_over_alpha": float(far / alpha),
            "calibrated_0.5_2": bool(0.5 <= far / alpha <= 2), "tpr": float(tpr),
            "normal_frames": int(A[:, 1].sum()), "anomalous_frames": int(A[:, 3].sum())}


def cmd_far(a, cfg):
    gt = json.loads(Path(a.gt).read_text())
    vids = sorted(gt)
    branch = ObjectBranch(cfg, a.scenes, "cpu")
    train, test = branch.load(a.features, ("training", "testing"))
    state = branch.fit(train)
    warm = int(cfg.get("warmup", 4))
    every = int(cfg.get("train_every", 1))
    smooth = offline_smoother(cfg.get("temporal", "causal_median"))
    ident = lambda s: np.asarray(s, float)
    z_obj = branch.calibrate(branch.score({v: test[v] for v in vids}, state), state)
    name, f_obj = branch.fuse(channel_order({}, z_obj, state.scenes))
    res = {"config": cfg.get("name"), "fused": name, "train_every": every, "warmup": warm,
           "ref_warmup_dropped": math.ceil(warm / every), "groups": {}, "empirical": {}, "analytic": {}}
    taus = {}
    for g in state.groups:
        ref = fused_reference(state, train, g)
        taus[g] = {}
        for tag, sm in (("smoothed", smooth), ("unsmoothed", ident)):
            r = np.concatenate(smooth_all(ref, sm, math.ceil(warm / every)))
            taus[g][tag] = {al: float(np.quantile(r, 1 - al)) for al in ALPHAS}
        res["groups"][g] = {"train_videos": len(ref), "ref_frames": int(sum(len(x) for x in ref)), "tau": taus[g]}
    grp = (lambda v: scene_of(v)) if state.scenes else (lambda v: "all")
    tested = [v for v in vids if grp(v) in taus]
    for tag, sm in (("smoothed", smooth), ("unsmoothed", ident)):
        res["empirical"][tag] = [summarise(rates({v: f_obj[v] for v in tested}, gt, warm, sm,
                                                 lambda v, al=al: taus[grp(v)][tag][al]), al) for al in ALPHAS]
    k = len(name.split("+"))
    res["analytic"]["obj"] = {"k": k, "rows": [summarise(rates({v: f_obj[v] for v in vids}, gt, warm, smooth,
                                                                 lambda v, al=al: gamma.ppf(1 - al, k)), al)
                                                for al in ALPHAS]}
    if a.extra_channels:
        z_grid = PrecomputedChannels(a.extra_channels, cfg.get("calibrator", "fisher")).calibrated(vids, state.scenes)
        name_all, f_all = branch.fuse(channel_order(z_grid, z_obj, state.scenes))
        k = len(name_all.split("+"))
        res["analytic"]["grid+obj"] = {"k": k, "fused": name_all, "rows": [
            summarise(rates({v: f_all[v] for v in vids}, gt, warm, smooth, lambda v, al=al: gamma.ppf(1 - al, k)), al)
            for al in ALPHAS]}
    for tag in ("smoothed", "unsmoothed"):
        for r in res["empirical"][tag]:
            print(f"empirical {tag:10s} alpha {r['alpha']:<6} FAR {r['far']:.4f} [{r['far_ci95'][0]:.4f}, "
                  f"{r['far_ci95'][1]:.4f}] FAR/alpha {r['far_over_alpha']:.2f} TPR {r['tpr']:.3f}")
    for key, d in res["analytic"].items():
        for r in d["rows"]:
            print(f"analytic  {key:10s} alpha {r['alpha']:<6} FAR {r['far']:.4f} FAR/alpha {r['far_over_alpha']:.2f} "
                  f"TPR {r['tpr']:.3f}")
    return res


def subset(train: dict, scenes: bool, frac: float, seed: int) -> list[str]:
    groups = {}
    for v in sorted(train):
        groups.setdefault(scene_of(v) if scenes else "all", []).append(v)
    rng = np.random.default_rng(1000 + seed)
    keep = []
    for g in sorted(groups):
        names = groups[g]
        n = min(len(names), max(2, math.ceil(frac * len(names))))
        keep += sorted(rng.choice(names, n, replace=False).tolist())
    return sorted(keep)


def cmd_dataeff(a, cfg):
    gt = json.loads(Path(a.gt).read_text())
    vids = sorted(gt)
    branch = ObjectBranch(cfg, a.scenes, "cpu")
    train, test = branch.load(a.features, ("training", "testing"))
    fps = FPS[a.dataset]
    rows = []
    for seed in ([0] if a.frac >= 1 else a.seeds):
        keep = subset(train, a.scenes, a.frac, seed)
        t0 = time.perf_counter()
        state = branch.fit({v: train[v] for v in keep})
        fit_s = time.perf_counter() - t0
        tested = [v for v in vids if not state.scenes or scene_of(v) in state.groups]
        z = branch.calibrate(branch.score({v: test[v] for v in tested}, state), state)
        res = branch.evaluate(channel_order({}, z, state.scenes), {v: gt[v] for v in tested})
        name = [k for k in res if "+" in k][0]
        frames = int(sum(int(train[v]["n_frames"]) for v in keep))
        rows.append({"seed": seed, "videos": len(keep), "frames": frames, "minutes": frames / fps / 60,
                     "fit_seconds": fit_s, "loadavg": os.getloadavg()[0], "torch_threads": torch.get_num_threads(),
                     "test_videos": len(tested), "fused": name, **res[name],
                     "channels": {k: v for k, v in res.items() if "+" not in k}})
        print(f"frac {a.frac} seed {seed} videos {len(keep)} min {frames / fps / 60:.1f} fit {fit_s:.1f}s "
              f"causal {res[name]['causal']:.4f} macro {res[name]['causal_macro']:.4f}", flush=True)
    c = np.array([r["causal"] for r in rows]); m = np.array([r["causal_macro"] for r in rows])
    return {"config": cfg.get("name"), "dataset": a.dataset, "frac": a.frac, "rows": rows,
            "causal_mean": float(c.mean()), "causal_std": float(c.std()),
            "macro_mean": float(m.mean()), "macro_std": float(m.std())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["far", "dataeff"])
    p.add_argument("--config", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--scenes", action="store_true")
    p.add_argument("--extra-channels", nargs="*", default=[])
    p.add_argument("--frac", type=float, default=1.0)
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if os.environ.get("OMP_NUM_THREADS"):
        torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    cfg = load_config(a.config)
    res = (cmd_far if a.cmd == "far" else cmd_dataeff)(a, cfg)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
