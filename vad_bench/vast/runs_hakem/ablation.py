"""S1 scoring ablation of the object branch (docs/tez_notlari/hakem_duzeltmeleri_on_kayit_2026-11-06.md).

Rows (online = causal median 9, warmup 4; paired video bootstrap against D1, 2000 replicates, seed 0):
  D1  NIW mixture selected on normal data (the system; must reproduce kanit/r2_20261105/res/base/<ds>_obj.json)
  D2  ML-GMM, same K, full covariance, reg_covar 1e-6, k-means init, random_state 0 (sklearn GaussianMixture)
  D2b prior-free twin: aean.niw.PluginMixture, same K and k-means components as NIW, ML Gaussians, weights n_k / N
  D3  single Gaussian (K = 1, ML, reg_covar 1e-6)
  D4  exact 1-NN Euclidean distance to the training rows (faiss IndexFlatL2)
  C1  z-score calibration + sum;  C2  max of the Fisher-calibrated channels
  T1  no temporal smoothing;      L-<attr>  one attribute left out
  F1  f = 0.25, five draws (deploy_measures.subset, default_rng(1000 + seed)): D1, D2, D4
D2-D4 use the attribute rows, the training subsample, the 2-fold cross-fitted reference (sorted video names, folds
names[0::2] / names[1::2]), the floor of frames without objects and the Fisher calibration exactly as fit_attribute;
only the density changes. The fitted extractors (appearance PCA) and K come from the D1 fit.

  PYTHONPATH=. python vad_bench/vast/runs_hakem/ablation.py run  --dataset ped2 --features F --gt G --out J
  PYTHONPATH=. python vad_bench/vast/runs_hakem/ablation.py cost --dataset ped2 --features F --out J   (alone, 4 threads)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from aean.niw import PluginMixture
from vadkit.calibration.fisher import FisherCalibrator, ZCalibrator
from vadkit.density.niw import VideoRows, _rows, scene_of
from vadkit.eval.protocol import evaluate
from vadkit.pipeline.offline import ObjectBranch, channel_order, load_config, record_rows
from vadkit.temporal.median import causal_median

ROOT = Path(__file__).resolve().parents[3]
CFG = {"shanghaitech": ROOT / "vadkit/configs/r2/base_sht.yaml"}
REF = ROOT / "paper/ieee_access/kanit/r2_20261105/res/base"
DATAEFF = ROOT / "paper/ieee_access/kanit/kurulum_20261105"
_spec = importlib.util.spec_from_file_location("deploy_measures", ROOT / "vad_bench/vast/runs_r2/deploy_measures.py")
_dm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dm)
subset = _dm.subset                                     # the draws of the data-efficiency measurement (M2)


# ------------------------------------------------------------------------------------------------ densities
class GMM:
    """sklearn GaussianMixture with K components (fewer if there are fewer rows); score -log p."""

    def __init__(self, K: int):
        self.K = int(K)

    def fit(self, X):
        from sklearn.mixture import GaussianMixture
        self.m = GaussianMixture(min(self.K, len(X)), covariance_type="full", reg_covar=1e-6, init_params="kmeans",
                                 random_state=0).fit(X)
        return self

    def neglog(self, X):
        return -self.m.score_samples(X)

    def nbytes(self):
        return int(self.m.means_.nbytes + self.m.covariances_.nbytes + self.m.weights_.nbytes)


class Plugin:
    """aean.niw.PluginMixture: the NIW mixture's k-means components with ML Gaussians (no prior)."""

    def __init__(self, K, kappa0, nu0):
        self.args = (int(K), float(kappa0), float(nu0))

    def fit(self, X):
        self.m = PluginMixture(*self.args).fit(torch.from_numpy(np.asarray(X, np.float64)))
        return self

    def neglog(self, X):
        return -self.m.logpdf(torch.from_numpy(np.asarray(X, np.float64))).numpy()


class KNN:
    """Exact 1-NN Euclidean distance (faiss IndexFlatL2 on float32 rows)."""

    def fit(self, X):
        import faiss
        X = np.ascontiguousarray(X, np.float32)
        self.ix = faiss.IndexFlatL2(X.shape[1])
        self.ix.add(X)
        self.bytes = int(X.nbytes)
        return self

    def neglog(self, X):
        if not len(X):
            return np.zeros(0)
        d, _ = self.ix.search(np.ascontiguousarray(X, np.float32), 1)
        return np.sqrt(np.maximum(d[:, 0].astype(np.float64), 0.0))

    def nbytes(self):
        return self.bytes


def frame_scores(model, f, X, n):
    s = np.full(int(n), -np.inf)
    if len(X):
        np.maximum.at(s, f, model.neglog(X))
    return s


# ------------------------------------------------------------------------------------------------ channels
def vrows(ext, rec) -> VideoRows:
    return VideoRows(*record_rows(ext, rec), int(rec["n_frames"]), int(rec["startup"]))


def alt_channel(make, ext, train: dict, names: list[str], test: dict, tvids: list[str], every: int, drop: bool):
    """(raw test scores {video: array}, reference, floor) of one attribute with the density make(), mirroring
    vadkit.density.niw.fit_attribute: fit rows subsampled, 2-fold cross-fitted reference, floor, full model."""
    names = sorted(names)
    rows = {v: _rows(vrows(ext, train[v]), True, every, drop) for v in names}
    folds = (names[0::2], names[1::2])
    ref = []
    for fit_n, ev_n in (folds, folds[::-1]):
        m = make().fit(np.concatenate([rows[v][1] for v in fit_n]))
        for v in ev_n:
            n, st = int(train[v]["n_frames"]), int(train[v]["startup"])
            s = frame_scores(m, *rows[v], n)
            t = np.arange(n)
            ref.append(s[(t % every == 0) & ((t >= st) if drop else True)])
    ref = np.concatenate(ref)
    floor = float(ref[np.isfinite(ref)].min())
    full = make().fit(np.concatenate([rows[v][1] for v in names]))
    out = {}
    for v in tvids:
        r = vrows(ext, test[v])
        s = frame_scores(full, *_rows(r, False, 1, drop), r.n_frames)
        out[v] = np.where(np.isfinite(s), s, floor)
    return out, np.where(np.isfinite(ref), ref, floor), floor, full


def group_videos(names, scenes):
    g = {}
    for v in sorted(names):
        g.setdefault(scene_of(v) if scenes else "all", []).append(v)
    return g


def alt_calibrated(kind: str, state, train: dict, keep: list[str], test: dict, tvids: list[str], calib="fisher"):
    """{channel: {video: calibrated score}} of density kind for the attributes of state.channels."""
    cfg = state.config
    every, drop = int(cfg.get("train_every", 1)), bool(cfg.get("drop_startup", False))
    tr_g, te_g = group_videos(keep, state.scenes), group_videos(tvids, state.scenes)
    z = {a: {} for a in state.channels}
    for g, tv in te_g.items():
        for a in state.channels:
            st = state.groups[g][a]
            h = st.hyper
            make = {"D2": lambda: GMM(h["K"]), "D2b": lambda: Plugin(h["K"], h["kappa0"], h["nu0"]),
                    "D3": lambda: GMM(1), "D4": lambda: KNN()}[kind]
            raw, ref, _, _ = alt_channel(make, st.extractor, train, tr_g[g], test, tv, every, drop)
            cal = (FisherCalibrator() if calib == "fisher" else ZCalibrator()).fit(ref)
            for v in tv:
                z[a][v] = cal(raw[v])
    return z


def niw_calibrated(branch, state, test, tvids, calib="fisher"):
    raw = branch.score({v: test[v] for v in tvids}, state)
    if calib == "fisher":
        return branch.calibrate(raw, state)
    z = {}
    for a, d in raw.items():
        z[a] = {}
        for v, s in d.items():
            st = state.groups[state.group_of(v)][a]
            z[a][v] = ZCalibrator().fit(st.ref)(s)
    return z


# ------------------------------------------------------------------------------------------------ evaluation
def identity(s):
    return np.asarray(s, np.float64)


def fused(z: dict, names: list[str], how: str = "sum") -> dict:
    vids = next(iter(z.values())).keys()
    if how == "sum":
        return {v: sum(z[n][v] for n in names) for v in vids}
    return {v: np.max(np.stack([z[n][v] for n in names]), 0) for v in vids}


def boot(sa: dict, sb: dict, gt: dict, sm_a=causal_median, sm_b=causal_median, warmup=4, reps=2000, seed=0):
    """vadkit.eval.protocol.paired_bootstrap with a smoother per method (same resampling, same skip rule)."""
    vids = sorted(gt)
    proc = {"a": {v: sm_a(sa[v][warmup:]) for v in vids}, "b": {v: sm_b(sb[v][warmup:]) for v in vids}}
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
    return {"replicates": reps, "skipped": skipped, "seed": seed,
            "auc": {n: {"point": point[n], "ci95": ci(rep[n])} for n in proc},
            "difference": {"point": point["a"] - point["b"], "ci95": ci(d), "fraction_positive": float((d > 0).mean())}}


def hyper_table(state):
    return {g: {a: {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in st.hyper.items()}
                for a, st in grp.items()} for g, grp in state.groups.items()}


def cmd_run(a):
    cfg = load_config(CFG.get(a.dataset, ROOT / "vadkit/configs/r2/base.yaml"))
    scenes = a.dataset == "shanghaitech"
    gt = json.loads(Path(a.gt).read_text())
    vids = sorted(gt)
    branch = ObjectBranch(cfg, scenes, "cpu")
    train, test = branch.load(a.features, ("training", "testing"))
    out = {"dataset": a.dataset, "config": cfg["name"], "features": str(a.features), "rows": {}, "boot": {},
           "timing_s": {}, "env": {"torch_threads": torch.get_num_threads(), "nproc": os.cpu_count()}}
    log = lambda m: print(f"{time.strftime('%H:%M:%S')} {a.dataset} {m}", flush=True)

    # D1: the system, and its reproduction check
    t0 = time.perf_counter()
    state = branch.fit(train)
    out["timing_s"]["D1_fit"] = time.perf_counter() - t0
    chans = list(channel_order({}, {c: None for c in state.channels}, scenes))
    z1 = niw_calibrated(branch, state, test, vids)
    res1 = branch.evaluate(channel_order({}, z1, scenes), gt)
    key = [k for k in res1 if "+" in k][0]
    ref = json.loads((REF / f"{a.dataset}_obj.json").read_text())
    rkey = [k for k in ref if "+" in k][0]
    diff = abs(res1[key]["causal"] - ref[rkey]["causal"])
    out["reproduction"] = {"ours": res1[key]["causal"], "archived": ref[rkey]["causal"], "abs_diff": diff,
                           "fused": key}
    log(f"D1 {res1[key]['causal']:.6f} archived {ref[rkey]['causal']:.6f} diff {diff:.2e}")
    if diff > 1e-6:
        Path(a.out).write_text(json.dumps(out, indent=1))
        raise SystemExit(f"D1 does not reproduce the archived result (|diff| = {diff:.3g}); S1 stops")
    out["hyper"] = hyper_table(state)
    out["channels"] = chans
    out["rows"]["D1"] = {"fused": res1[key], "channels": {c: res1[c] for c in chans}}
    f1 = fused(z1, chans)
    sm = causal_median

    def record(code, z=None, scores=None, smooth=sm, extra=None):
        s = scores if scores is not None else fused(z, chans)
        r = evaluate({v: s[v] for v in vids}, gt, 4, smooth)
        row = {"fused": r}
        if z is not None:
            row["channels"] = {c: evaluate({v: z[c][v] for v in vids}, gt, 4, smooth) for c in chans}
        if extra:
            row.update(extra)
        out["rows"][code] = row
        out["boot"][code] = boot(s, f1, gt, smooth, sm)
        b = out["boot"][code]["difference"]
        log(f"{code} {r['causal']:.4f} macro {r['causal_macro']:.4f} diff {b['point']:+.4f} "
            f"[{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]")
        Path(a.out).write_text(json.dumps(out, indent=1))

    for code in ("D2", "D2b", "D3", "D4"):
        t0 = time.perf_counter()
        z = alt_calibrated(code, state, train, sorted(train), test, vids)
        out["timing_s"][code] = time.perf_counter() - t0
        record(code, z)
    record("C1", scores=fused(niw_calibrated(branch, state, test, vids, "z"), chans))
    record("C2", scores=fused(z1, chans, "max"))
    record("T1", scores=f1, smooth=identity)
    for c in chans:
        rest = [n for n in chans if n != c]
        record(f"L-{c}", scores=fused(z1, rest), extra={"channels_used": rest})

    # F1: f = 0.25, the five draws of the data-efficiency measurement
    de = json.loads((DATAEFF / f"dataeff_{a.dataset}_0.25.json").read_text())
    out["F1"] = []
    for seed in range(5):
        keep = subset(train, scenes, 0.25, seed)
        st = branch.fit({v: train[v] for v in keep})
        tested = [v for v in vids if not st.scenes or scene_of(v) in st.groups]
        g = {v: gt[v] for v in tested}
        ch = list(channel_order({}, {c: None for c in st.channels}, scenes))
        zz = fused(niw_calibrated(branch, st, test, tested), ch)
        row = {"seed": seed, "videos": len(keep), "test_videos": len(tested),
               "D1": evaluate({v: zz[v] for v in tested}, g)}
        arch = [r for r in de["rows"] if r["seed"] == seed][0]
        row["D1_archived_causal"] = arch["causal"]
        row["D1_abs_diff"] = abs(row["D1"]["causal"] - arch["causal"])
        for code in ("D2", "D4"):
            zc = fused(alt_calibrated(code, st, train, keep, test, tested), ch)
            row[code] = evaluate({v: zc[v] for v in tested}, g)
        out["F1"].append(row)
        log(f"F1 seed {seed}: D1 {row['D1']['causal']:.4f} (archived diff {row['D1_abs_diff']:.1e}) "
            f"D2 {row['D2']['causal']:.4f} D4 {row['D4']['causal']:.4f}")
        Path(a.out).write_text(json.dumps(out, indent=1))
    log("RUN_DONE")


def cmd_cost(a):
    """Memory and per-frame CPU scoring latency of D1 (NIW) and D4 (1-NN) for the three attributes together, batch =
    the objects of one frame, on the first a.frames test frames with objects (20 warm-up frames not timed)."""
    import faiss
    torch.set_num_threads(a.threads)
    faiss.omp_set_num_threads(a.threads)
    cfg = load_config(CFG.get(a.dataset, ROOT / "vadkit/configs/r2/base.yaml"))
    scenes = a.dataset == "shanghaitech"
    branch = ObjectBranch(cfg, scenes, "cpu")
    train, test = branch.load(a.features, ("training", "testing"))
    if scenes:                                    # one scene's models: the first scene with test videos
        g0 = sorted({scene_of(v) for v in test})[0]
        train = {v: r for v, r in train.items() if scene_of(v) == g0}
        test = {v: r for v, r in test.items() if scene_of(v) == g0}
    state = branch.fit(train)
    grp = next(iter(state.groups.values()))
    every = int(cfg.get("train_every", 1))
    knn, niw_bytes, knn_bytes = {}, 0, 0
    for c, st in grp.items():
        X = np.concatenate([_rows(vrows(st.extractor, train[v]), True, every, False)[1] for v in sorted(train)])
        knn[c] = KNN().fit(X)
        knn_bytes += knn[c].nbytes()
        niw_bytes += sum(np.asarray(x).nbytes for k, x in st.density.state().items() if k in ("centres", "n", "s1", "s2"))
    frames = []
    for v in sorted(test):
        rec = test[v]
        rows = {c: vrows(grp[c].extractor, rec) for c in grp}
        for t in np.unique(rec["f"]):
            frames.append({c: r.X[r.f == t] for c, r in rows.items()})
            if len(frames) >= a.frames + 20:
                break
        if len(frames) >= a.frames + 20:
            break
    res = {}
    for name, fn in (("D1", lambda c, X: grp[c].density.logpdf(X)), ("D4", lambda c, X: knn[c].neglog(X))):
        ms = []
        for i, fr in enumerate(frames):
            t0 = time.perf_counter()
            for c, X in fr.items():
                if len(X):
                    fn(c, X)
            if i >= 20:
                ms.append((time.perf_counter() - t0) * 1e3)
        res[name] = {"ms_per_frame_median": float(np.median(ms)), "ms_per_frame_p95": float(np.percentile(ms, 95)),
                     "frames": len(ms)}
    out = {"dataset": a.dataset, "group": next(iter(state.groups)), "threads": a.threads,
           "objects_per_frame_mean": float(np.mean([max(len(X) for X in fr.values()) for fr in frames[20:]])),
           "train_rows": {c: int(knn[c].ix.ntotal) for c in knn}, "dims": {c: int(knn[c].ix.d) for c in knn},
           "bytes": {"D1_niw_sufficient_stats": int(niw_bytes), "D4_knn_bank_float32": int(knn_bytes)},
           "latency": res, "loadavg": os.getloadavg()[0]}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "cost"])
    p.add_argument("--dataset", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--gt")
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--frames", type=int, default=300)
    a = p.parse_args()
    if a.cmd == "run" and os.environ.get("OMP_NUM_THREADS"):
        torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    (cmd_run if a.cmd == "run" else cmd_cost)(a)


if __name__ == "__main__":
    main()
