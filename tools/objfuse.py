"""Fisher fusion of the archived frame channels with the object channels; frame AUC under the paper's protocols.

    python tools/objfuse.py --ds ped2 --runs /workspace/checkpoints/scores/runs --obj /workspace/sonuclar/obj \
        --gt /workspace/cache/ped2/gt.json [--prefixes 01_ 02_ ...]

Archived channels: I = AE residual FRX-Bayes (pool48_mc4), Fs = flow speed GMM-UBM, Fd = flow direction FRX-Bayes
(pool48_mc5, DIS medium). Object channels: V = vel_niw, A = app_niw, K = app_knn (tools/objscore.py).
Every channel is turned into -log p against its own cross-fitted normal reference (tools/fuse_runs.normaliser) and
the selected channels are summed. Protocols: causal median 9 after excluding `--warmup` frames per video
(strict causal), and offline per-video min-max + Gaussian sigma 3. No parameter is chosen on test labels here:
every listed combination is reported.
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuse_runs import normaliser  # noqa: E402

ARCH = {"I": ("pool48_mc4", "frx_bayes"), "Fs": ("pool48_mc5", "flow_ubm_m"), "Fd": ("pool48_mc5", "flow_rx_m")}
OBJ = {"V": "vel_niw", "A": "app_niw", "K": "app_knn", "M": "app_mix", "G": "appgeo_mix", "W": "velgeo_mix", "R": "velraw_mix"}


def causal_median(s, k=9):
    return np.array([np.median(s[max(0, i - k + 1): i + 1]) for i in range(len(s))])


def offline(s):
    s = (s - s.min()) / (s.max() - s.min() + 1e-12)
    return gaussian_filter1d(s, 3)


def channels(ds_key: str, runs: Path, obj: Path, vids):
    ch = {}
    for n, (tag, key) in ARCH.items():
        f = runs / f"{ds_key}_2d_noadv" / f"eval_{ds_key}_{tag}" / "scores.npz"
        if f.exists():
            S = np.load(f)
            ch[n] = ({v: S[f"{key}/{v}"] for v in vids}, S[f"ref/{key}"])
    if (obj / "scores.npz").exists():
        S = np.load(obj / "scores.npz")
        for n, key in OBJ.items():
            if f"ref/{key}" not in S.files:
                continue
            ch[n] = ({v: S[f"{key}/{v}"] for v in vids}, S[f"ref/{key}"])
    return ch


def extra_channels(paths, vids):
    """Every channel with a 'ref/<name>' key in the given scores.npz files."""
    ch = {}
    for p in paths or []:
        S = np.load(p)
        for k in S.files:
            if k.startswith("ref/"):
                n = k[4:]
                ch[n] = ({v: S[f"{n}/{v}"] for v in vids}, S[k])
    return ch


def evaluate(scores: dict, gt: dict, warmup: int):
    y_c, s_c, y_o, s_o, per = [], [], [], [], []
    for v, s in scores.items():
        g = np.asarray(gt[v])
        sc = causal_median(s[warmup:])
        y_c.append(g[warmup:]); s_c.append(sc)
        y_o.append(g); s_o.append(offline(s))
        if 0 < g[warmup:].sum() < len(g) - warmup:
            per.append(roc_auc_score(g[warmup:], sc))
    return {"causal": roc_auc_score(np.concatenate(y_c), np.concatenate(s_c)),
            "offline": roc_auc_score(np.concatenate(y_o), np.concatenate(s_o)),
            "causal_macro": float(np.mean(per)) if per else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, help="dataset key of the archived runs, e.g. ped2, avenue, shtech_01")
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--obj", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--max-combo", type=int, default=5)
    ap.add_argument("--extra", nargs="*", type=Path, help="more scores.npz files with <ch>/<video> and ref/<ch>")
    ap.add_argument("--only", nargs="*", help="restrict to these channel names")
    a = ap.parse_args()
    gt = json.loads(a.gt.read_text())
    vids = sorted(gt)
    ch = channels(a.ds, a.runs, a.obj, vids)
    ch.update(extra_channels(a.extra, vids))
    if a.only:
        ch = {n: c for n, c in ch.items() if n in a.only}
    z = {n: {v: normaliser(ref, "fisher")(sc[v]) for v in vids} for n, (sc, ref) in ch.items()}
    res = {}
    names = list(z)
    for r in range(1, min(len(names), a.max_combo) + 1):
        for combo in combinations(names, r):
            fused = {v: sum(z[n][v] for n in combo) for v in vids}
            res["+".join(combo)] = evaluate(fused, gt, a.warmup)
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]["causal"]):
        print(f"{k:16s} causal {v['causal']:.4f}  macro {v['causal_macro']:.4f}  offline {v['offline']:.4f}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
