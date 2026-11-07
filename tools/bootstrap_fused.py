"""Video-level paired bootstrap of pooled causal frame AUC for fused channel sets (tools/objfuse.py conventions).

    python tools/bootstrap_fused.py --gt gt.json --warmup 4 --reps 2000 --out ci.json \
        --method niw=fisher:/w/accivad_bayes/avenue/scores.npz:bvel,bpose,bdeep \
        --method acc=sum:/w/accivad/avenue/scores.npz:velocity,pose,deep \
        --pairs niw-acc

A method is rule:path:channels, several groups joined with '+' (e.g. cells from one file and objects from another):
  fisher : sum of -log p_j with p_j from the cross-fitted normal reference ref/<ch> of each file
  sum    : plain sum of the stored scores (for scores already scaled by their own method, e.g. AccI-VAD's).
Per video the fused score is computed once, the first `warmup` frames are dropped and the causal median 9 applied.
Every replicate resamples the SAME videos (with replacement) for all methods, so differences are paired;
intervals are percentile intervals. Videos are the resampling unit because frames within a video are dependent.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuse_runs import normaliser  # noqa: E402
from objfuse import causal_median  # noqa: E402


def fused(spec: str, vids):
    rule, rest = spec.split(":", 1)
    total = {v: 0.0 for v in vids}
    for grp in rest.split("+"):
        path, chs = grp.rsplit(":", 1)
        S = np.load(path)
        for c in chs.split(","):
            norm = normaliser(S[f"ref/{c}"], "fisher") if rule == "fisher" else (lambda x: x)
            for v in vids:
                total[v] = total[v] + norm(S[f"{c}/{v}"].astype(np.float64))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--method", action="append", required=True, help="name=rule:path:ch1,ch2[+path:ch]")
    ap.add_argument("--pairs", nargs="*", default=[], help="a-b differences to report")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    gt = json.loads(a.gt.read_text())
    vids = sorted(gt)
    M = dict(m.split("=", 1) for m in a.method)
    proc = {}
    for name, spec in M.items():
        f = fused(spec, vids)
        proc[name] = {v: causal_median(f[v][a.warmup:]) for v in vids}
    Y = {v: np.asarray(gt[v][a.warmup:]) for v in vids}

    def auc(name, sample):
        y = np.concatenate([Y[v] for v in sample])
        if y.min() == y.max():
            return None
        return roc_auc_score(y, np.concatenate([proc[name][v] for v in sample]))

    point = {n: auc(n, vids) for n in M}
    rng = np.random.default_rng(a.seed)
    reps = {n: [] for n in M}
    skipped = 0
    for _ in range(a.reps):
        sample = [vids[i] for i in rng.integers(0, len(vids), len(vids))]
        vals = {n: auc(n, sample) for n in M}
        if any(x is None for x in vals.values()):
            skipped += 1
            continue
        for n, x in vals.items():
            reps[n].append(x)
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    res = {"gt": str(a.gt), "videos": len(vids), "warmup": a.warmup, "replicates": a.reps, "skipped": skipped,
           "seed": a.seed, "methods": M,
           "auc": {n: {"point": point[n], "ci95": ci(reps[n])} for n in M}, "differences": {}}
    for p in a.pairs:
        x, y = p.split("-")
        d = np.asarray(reps[x]) - np.asarray(reps[y])
        res["differences"][p] = {"point": point[x] - point[y], "ci95": ci(d), "fraction_positive": float((d > 0).mean())}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({"auc": {n: round(v["point"], 4) for n, v in res["auc"].items()},
                      "diff": {k: [round(v["point"], 4)] + [round(c, 4) for c in v["ci95"]] for k, v in res["differences"].items()}}))


if __name__ == "__main__":
    main()
