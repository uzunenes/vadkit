"""Calibration, fusion, causal median, evaluation, bootstrap, PCA and NIW state equal the reference code exactly."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from fuse_runs import normaliser  # noqa: E402
from objfuse import causal_median as ref_median, evaluate as ref_evaluate  # noqa: E402

from aean.niw import NIWMixture  # noqa: E402
from vadkit.attributes.appearance import AppearancePCA  # noqa: E402
from vadkit.attributes.pose import PoseKeypoints  # noqa: E402
from vadkit.calibration.fisher import FisherCalibrator, ZCalibrator  # noqa: E402
from vadkit.core.interfaces import Objects  # noqa: E402
from vadkit.density.niw import NIWDensity  # noqa: E402
from vadkit.eval.protocol import evaluate, paired_bootstrap  # noqa: E402
from vadkit.fusion.sum import SumFusion  # noqa: E402
from vadkit.temporal.median import CausalMedian, causal_median  # noqa: E402


def _videos(rng, n=9, both=True):
    """Random scores and labels; some videos all-normal, some with ties."""
    sc, gt = {}, {}
    for i in range(n):
        T = int(rng.integers(15, 80))
        s = rng.gamma(2.0, 1.0, T)
        s[: T // 5] = np.round(s[: T // 5], 1)
        g = np.zeros(T, int)
        if both and i % 3:
            a = int(rng.integers(5, T - 5)); g[a: a + int(rng.integers(2, 10))] = 1
            s[g == 1] += rng.normal(0.8, 1.0, int(g.sum()))
        sc[f"v{i:02d}"], gt[f"v{i:02d}"] = s, g.tolist()
    return sc, gt


@pytest.mark.parametrize("k", [1, 2, 9, 10])
def test_causal_median(k):
    rng = np.random.default_rng(k)
    for T in (0, 1, 5, 9, 37):
        s = rng.normal(size=T)
        s[::4] = 0.5
        assert np.array_equal(causal_median(s, k), ref_median(s, k))
        cm = CausalMedian(k)
        assert np.array_equal(np.array([cm.push(x) for x in s]), ref_median(s, k))
        cm.reset()
        assert len(cm.buf) == 0


@pytest.mark.parametrize("seed", range(4))
def test_calibrators(seed):
    rng = np.random.default_rng(seed)
    ref = rng.gamma(2.0, 1.0, int(rng.integers(50, 3000)))
    s = np.concatenate([rng.gamma(2.0, 2.0, 500), ref[:50], [ref.min() - 1, ref.max() + 1, np.quantile(ref, 0.9)]])
    for kind, cal in (("fisher", FisherCalibrator()), ("z", ZCalibrator())):
        cal.fit(ref)
        assert np.array_equal(cal(s), normaliser(ref, kind)(s))
        assert all(cal(x) == normaliser(ref, kind)(x) for x in s[:20])         # scalars (streaming)
    const = np.ones(100)                                                      # no reference value above u
    assert np.array_equal(FisherCalibrator().fit(const)(s), normaliser(const, "fisher")(s))


def test_sum_fusion():
    rng = np.random.default_rng(0)
    z = {n: rng.normal(size=50) * 10.0 ** rng.integers(-3, 3) for n in ("I", "Fs", "ovel", "oapp")}
    combo = ("Fs", "ovel", "oapp")
    assert np.array_equal(SumFusion(list(combo))(z), sum(z[n] for n in combo))
    assert np.array_equal(SumFusion()(z), sum(z[n] for n in z))
    with pytest.raises(KeyError):
        SumFusion(["missing"])(z)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("warmup", [0, 4])
def test_evaluate(seed, warmup):
    sc, gt = _videos(np.random.default_rng(seed))
    got, want = evaluate(sc, gt, warmup), ref_evaluate(sc, gt, warmup)
    assert got.keys() == want.keys() and all(got[k] == want[k] for k in want)


def test_paired_bootstrap(tmp_path):
    """Same JSON numbers as tools/bootstrap_fused.py (rule 'sum' = stored scores) run as a script."""
    rng = np.random.default_rng(1)
    a, gt = _videos(rng, n=8)
    b = {v: s + rng.normal(0, 1.0, len(s)) for v, s in a.items()}
    np.savez(tmp_path / "s.npz", **{f"a/{v}": s for v, s in a.items()}, **{f"b/{v}": s for v, s in b.items()})
    (tmp_path / "gt.json").write_text(json.dumps(gt))
    p = tmp_path / "s.npz"
    subprocess.run([sys.executable, str(ROOT / "tools" / "bootstrap_fused.py"), "--gt", str(tmp_path / "gt.json"),
                    "--method", f"a=sum:{p}:a", "--method", f"b=sum:{p}:b", "--pairs", "a-b", "--warmup", "4",
                    "--reps", "300", "--seed", "3", "--out", str(tmp_path / "ci.json")], check=True, capture_output=True)
    want = json.loads((tmp_path / "ci.json").read_text())
    got = json.loads(json.dumps(paired_bootstrap(a, b, gt, warmup=4, reps=300, seed=3)))
    assert got["skipped"] == want["skipped"] and got["videos"] == want["videos"]
    assert got["auc"] == want["auc"]
    assert got["difference"] == want["differences"]["a-b"]


def test_appearance_pca():
    """objscore2's oapp projection on random float16 embeddings, rng consumed identically, state round trip."""
    rng = np.random.default_rng(0)
    vids = [rng.normal(size=(int(rng.integers(30, 90)), 40)).astype(np.float16) for _ in range(5)]
    r1, r2 = np.random.default_rng(7), np.random.default_rng(7)
    pca = AppearancePCA(dim=8, max_rows=200).fit(vids, r1)
    A = np.concatenate([x.astype(np.float64) for x in vids])
    mu = A.mean(0)
    _, _, Vt = np.linalg.svd(A[r2.permutation(len(A))[:200]] - mu, full_matrices=False)
    P = Vt[:8].T
    for x in vids:
        assert np.array_equal(pca.transform(x), (x.astype(np.float64) - mu) @ P)
        assert np.array_equal(AppearancePCA.from_state(pca.state()).transform(x), (x.astype(np.float64) - mu) @ P)
    assert r1.integers(1 << 30) == r2.integers(1 << 30)
    objs = Objects(box=np.zeros((len(vids[0]), 4), np.float32), cls=np.zeros(len(vids[0]), np.int16),
                   conf=np.zeros(len(vids[0]), np.float32), src=np.zeros(len(vids[0]), np.uint8), attrs={"app": vids[0]})
    assert np.array_equal(pca(objs, None, (240, 360)), pca.transform(vids[0]))
    q = AppearancePCA.from_state(AppearancePCA(dim=8, max_rows=200, key="app_s2").fit(vids, r1).state())
    assert (q.dim, q.max_rows, q.key) == (8, 200, "app_s2")                 # the column survives a state round trip
    assert np.array_equal(q.record_rows({"f": np.zeros(len(vids[0]), int), "app_s2": vids[0]})[1], q.transform(vids[0]))


def test_pose_rows():
    rng = np.random.default_rng(0)
    rec = {"f": np.arange(20, dtype=np.int32) // 3, "pose": rng.normal(size=(20, 34)).astype(np.float32),
           "pose_ok": (rng.random(20) > 0.5).astype(np.uint8)}
    f, X = PoseKeypoints().record_rows(rec)
    k = rec["pose_ok"] > 0
    assert np.array_equal(f, rec["f"][k]) and np.array_equal(X, rec["pose"][k].astype(np.float64))
    objs = Objects(box=np.zeros((20, 4), np.float32), cls=np.zeros(20, np.int16), conf=np.zeros(20, np.float32),
                   src=np.zeros(20, np.uint8), pose=rec["pose"], pose_ok=rec["pose_ok"])
    assert np.array_equal(PoseKeypoints()(objs, None, (240, 360)), X)


@pytest.mark.parametrize("K", [1, 4])
def test_niw_density_state(K):
    """Given hyper-parameters: equal to NIWMixture on the sorted-order concatenation; state round trip exact."""
    rng = np.random.default_rng(K)
    rows = {f"v{i}": rng.normal(size=(int(rng.integers(20, 60)), 5)) for i in (3, 1, 2, 0)}
    d = NIWDensity(K, 10, 5 + 1 + 3).fit(rows)
    m = NIWMixture(K, 10, 9).fit(torch.from_numpy(np.concatenate([rows[v] for v in sorted(rows)])))
    X = rng.normal(size=(100, 5)) * 2
    assert np.array_equal(d.logpdf(X), m.logpdf(torch.from_numpy(X)).numpy())
    st = {k: np.asarray(v) for k, v in d.state().items()}                    # as after an npz round trip
    assert np.array_equal(NIWDensity.from_state(st).logpdf(X), d.logpdf(X))
