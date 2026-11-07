"""Frame-AUC protocols of the paper (tools/objfuse.evaluate) and the paired video bootstrap (tools/bootstrap_fused.py).
The only module of vadkit that reads test labels."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score

from vadkit.temporal.median import causal_median


def offline(s: np.ndarray) -> np.ndarray:
    """Offline (non-causal) protocol: per-video min-max, then Gaussian sigma 3."""
    s = (s - s.min()) / (s.max() - s.min() + 1e-12)
    return gaussian_filter1d(s, 3)


def evaluate(scores: dict[str, np.ndarray], gt: dict, warmup: int = 4,
             smooth: Callable[[np.ndarray], np.ndarray] = causal_median) -> dict[str, float]:
    """causal: pooled AUC of smooth (default the causal median 9) after dropping `warmup` frames per video;
    causal_macro: mean of the per-video causal AUCs of videos with both classes; offline: pooled AUC of offline()."""
    y_c, s_c, y_o, s_o, per = [], [], [], [], []
    for v, s in scores.items():
        g = np.asarray(gt[v])
        sc = smooth(s[warmup:])
        y_c.append(g[warmup:]); s_c.append(sc)
        y_o.append(g); s_o.append(offline(s))
        if 0 < g[warmup:].sum() < len(g) - warmup:
            per.append(roc_auc_score(g[warmup:], sc))
    return {"causal": roc_auc_score(np.concatenate(y_c), np.concatenate(s_c)),
            "offline": roc_auc_score(np.concatenate(y_o), np.concatenate(s_o)),
            "causal_macro": float(np.mean(per)) if per else float("nan")}


def paired_bootstrap(scores_a: dict[str, np.ndarray], scores_b: dict[str, np.ndarray], gt: dict, warmup: int = 4,
                     reps: int = 2000, seed: int = 0) -> dict:
    """Pooled causal AUC of two methods with percentile 95% intervals; every replicate resamples the same videos of
    sorted(gt) with replacement for both methods, replicates with one class only are skipped (bootstrap_fused.py)."""
    vids = sorted(gt)
    proc = {n: {v: causal_median(S[v][warmup:]) for v in vids} for n, S in (("a", scores_a), ("b", scores_b))}
    Y = {v: np.asarray(gt[v][warmup:]) for v in vids}

    def auc(name, sample):
        y = np.concatenate([Y[v] for v in sample])
        if y.min() == y.max():
            return None
        return roc_auc_score(y, np.concatenate([proc[name][v] for v in sample]))

    point = {n: auc(n, vids) for n in proc}
    rng = np.random.default_rng(seed)
    rep = {n: [] for n in proc}
    skipped = 0
    for _ in range(reps):
        sample = [vids[i] for i in rng.integers(0, len(vids), len(vids))]
        vals = {n: auc(n, sample) for n in proc}
        if any(x is None for x in vals.values()):
            skipped += 1
            continue
        for n, x in vals.items():
            rep[n].append(x)
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    d = np.asarray(rep["a"]) - np.asarray(rep["b"])
    return {"videos": len(vids), "warmup": warmup, "replicates": reps, "skipped": skipped, "seed": seed,
            "auc": {n: {"point": point[n], "ci95": ci(rep[n])} for n in proc},
            "difference": {"point": point["a"] - point["b"], "ci95": ci(d), "fraction_positive": float((d > 0).mean())}}
