"""Bayesian NIW-mixture scores of the light object attributes of tools/objfeat2.py; normal training data only.

    python tools/objscore2.py --feat /workspace/feat2/avenue --gt /workspace/cache/avenue/gt.json \
        --out /workspace/sonuclar/obj2/avenue [--prefix 01_]

Attributes: ovel (8-d velocity), opose (34-d keypoints, matched persons only), oapp (embedding -> PCA 64).
For each attribute the mixture size and the NIW prior strengths are chosen by held-out predictive likelihood on
normal training videos (2-fold split by video); frame score = max over objects of -log posterior predictive density;
frames without objects of that attribute take the lowest normal reference score. Reference scores are cross-fitted
by video. Output: scores.npz (<attr>/<video>, ref/<attr>) and hyper.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aean.niw import NIWMixture, select_mixture  # noqa: E402


def load(d: Path, prefix: str):
    return {p.stem: dict(np.load(p)) for p in sorted(d.glob("*.npz")) if p.stem.startswith(prefix)}


def attr_rows(r: dict, attr: str, fn):
    if attr == "opose":
        k = r["pose_ok"] > 0
        return r["f"][k], fn(r["pose"][k].astype(np.float64))
    if attr == "ovel":
        return r["f"], fn(r["vel"].astype(np.float64))
    return r["f"], fn(r["app"].astype(np.float64))


def frame_scores(m, r, attr, fn):
    f, X = attr_rows(r, attr, fn)
    s = np.full(int(r["n_frames"]), -np.inf)
    if len(X):
        np.maximum.at(s, f, (-m.logpdf(torch.from_numpy(X))).cpu().numpy())
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--attrs", nargs="+", default=["ovel", "opose", "oapp"])
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train, test = load(a.feat / "training", a.prefix), load(a.feat / "testing", a.prefix)
    gt = {v: g for v, g in json.loads(a.gt.read_text()).items() if v.startswith(a.prefix)}
    for v in gt:
        assert int(test[v]["n_frames"]) == len(gt[v]), v
    names = sorted(train)
    folds = (names[0::2], names[1::2])
    rng = np.random.default_rng(0)
    out, hyper = {}, {}
    for attr in a.attrs:
        fn = lambda x: x
        if attr == "oapp":
            A = np.concatenate([r["app"].astype(np.float64) for r in train.values()])
            mu = A.mean(0)
            _, _, Vt = np.linalg.svd(A[rng.permutation(len(A))[:50000]] - mu, full_matrices=False)
            P = Vt[: a.pca].T
            fn = lambda x, mu=mu, P=P: (x - mu) @ P
        X = {v: torch.from_numpy(attr_rows(train[v], attr, fn)[1]) for v in names}
        fit_sets, eval_sets = [], []
        for fit_n, ev_n in (folds, folds[::-1]):
            Xe = torch.cat([X[v] for v in ev_n])
            fit_sets.append(torch.cat([X[v] for v in fit_n]))
            eval_sets.append(Xe[rng.permutation(len(Xe))[:20000]])
        if min(len(x) for x in fit_sets) < 10:
            print(attr, "skipped: too few training objects"); continue
        (K, k0, e), _ = select_mixture(fit_sets, eval_sets, device=dev)
        C = fit_sets[0].shape[1]
        ref = []
        for (fit_n, ev_n), Xf in zip((folds, folds[::-1]), fit_sets):
            m = NIWMixture(K, k0, C + 1 + e, device=dev).fit(Xf)
            ref += [frame_scores(m, train[v], attr, fn) for v in ev_n]
        ref = np.concatenate(ref)
        floor = ref[np.isfinite(ref)].min()
        full = NIWMixture(K, k0, C + 1 + e, device=dev).fit(torch.cat(list(X.values())))
        out[f"ref/{attr}"] = np.where(np.isfinite(ref), ref, floor)
        for v in gt:
            s = frame_scores(full, test[v], attr, fn)
            out[f"{attr}/{v}"] = np.where(np.isfinite(s), s, floor)
        hyper[attr] = {"K": K, "kappa0": k0, "nu0": C + 1 + e, "dim": C}
        print(attr, hyper[attr], flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / "scores.npz", **out)
    (a.out / "hyper.json").write_text(json.dumps(hyper, indent=1))


if __name__ == "__main__":
    main()
