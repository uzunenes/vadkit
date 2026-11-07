"""HB-POT (vadkit/calibration/gpd.py, hier.py; docs/tez_notlari/hiyerarsik_kalibrasyon_on_kayit_2026-11-05.md):
T0 is FisherCalibrator bit for bit, the GPD tail reduces to the exponential one at xi = 0, t2 is t1 with one scene,
empirical Bayes shrinks small-J scenes more, the truncated mode keeps xi >= 0, the grid adapter refuses a group
calibrator without video ids, and the blind copy reads only 'ref/' arrays."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import genpareto

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import grid_ref_ids  # noqa: E402
import grid_refs_only  # noqa: E402

from vadkit.calibration import hier  # noqa: E402
from vadkit.calibration.fisher import FisherCalibrator  # noqa: E402
from vadkit.calibration.gpd import GPDTailCalibrator  # noqa: E402
from vadkit.core.registry import build  # noqa: E402
from vadkit.grid.precomputed import PrecomputedChannels  # noqa: E402


def _probe(rng, ref):
    return np.concatenate([rng.gamma(2.0, 2.0, 500), ref[:50], [ref.min() - 1, ref.max() + 1, np.quantile(ref, 0.9)],
                           np.quantile(ref, 0.9) + np.array([1e-12, 1e-6, 1.0, 1e3])])


@pytest.mark.parametrize("seed", range(5))
def test_t0_bit_identical_to_fisher(seed):
    rng = np.random.default_rng(seed)
    ref = rng.gamma(2.0, 1.0, int(rng.integers(50, 3000))) * 10.0 ** rng.integers(-2, 3)
    s = _probe(rng, ref)
    want = FisherCalibrator().fit(ref)
    for got in (GPDTailCalibrator().fit(ref), GPDTailCalibrator(0.0).fit(ref), build("calibrator", "fisher_gpd").fit(ref),
                hier.fit_groups({"a": ref}, {"a": np.arange(len(ref)) % 7}, "t0")["a"]):
        assert np.array_equal(got(s), want(s))
        assert all(got(x) == want(x) for x in s[:30])                     # scalars (streaming)
    const = np.ones(100)
    assert np.array_equal(GPDTailCalibrator().fit(const)(s), FisherCalibrator().fit(const)(s))


def test_gpd_xi0_is_exponential_tail():
    rng = np.random.default_rng(0)
    ref = rng.gamma(2.0, 1.0, 2000)
    s = _probe(rng, ref)
    c = GPDTailCalibrator(0.0, 2.5).fit(ref)
    up = s > c.u
    assert np.array_equal(c(s)[up], -np.log(0.1) + (s[up] - c.u) / 2.5)
    assert np.array_equal(c(s)[~up], FisherCalibrator().fit(ref)(s)[~up])  # body unchanged
    near = GPDTailCalibrator(1e-9, 2.5).fit(ref)                         # continuity in xi
    assert np.allclose(near(s), c(s), rtol=1e-6, atol=1e-6)
    y = np.linspace(0.01, 50, 200)
    assert np.allclose(c.tail_logpdf(y), genpareto.logpdf(y, 0.0, 0, 2.5))
    g = GPDTailCalibrator(0.4, 1.7).fit(ref)
    assert np.allclose(g.tail_logpdf(y), genpareto.logpdf(y, 0.4, 0, 1.7))
    z = g(s)                                                             # monotone, continuous at u (z = ln 10)
    o = np.argsort(s)
    assert np.all(np.diff(z[o]) >= 0)
    assert abs(g(np.array([c.u + 1e-12]))[0] - np.log(10.0)) < 1e-9
    with pytest.raises(ValueError):
        GPDTailCalibrator(-0.1)


@pytest.mark.parametrize("xi", [-0.3, 0.0, 0.3, 0.9])
def test_gpd_fit(xi):
    y = genpareto.rvs(xi, scale=2.0, size=4000, random_state=np.random.default_rng(1))
    x, sg = hier.gpd_fit(y)
    assert abs(x - xi) < 0.08 and abs(sg / 2.0 - 1) < 0.1
    sx, _, ss = genpareto.fit(y, floc=0)
    ll = lambda a, b: genpareto.logpdf(y, a, 0, b).sum()
    assert ll(x, sg) >= ll(sx, ss) - 1e-6                               # at least scipy's optimum
    xb, _ = hier.gpd_fit(y, (0.5, 1.0)) if xi < 0.5 else hier.gpd_fit(y, (-0.5, 0.0))
    assert xb == (0.5 if xi < 0.5 else 0.0)                              # bounded: the optimum sits on the bound


def _scene(rng, J, xi=0.4, per=80):
    """J videos of `per` frames with a GPD-like tail; returns (ref, ids)."""
    ref = np.concatenate([np.where(rng.random(per) < 0.85, rng.gamma(2.0, 1.0, per),
                                   3.0 + genpareto.rvs(xi, scale=1.0, size=per, random_state=rng)) for _ in range(J)])
    return ref, np.repeat(np.arange(J), per)


def test_t2_equals_t1_with_one_scene():
    ref, ids = _scene(np.random.default_rng(2), 12)
    f1 = hier.hb_fit({"s": ref}, {"s": ids}, "t1", B=40)
    f2 = hier.hb_fit({"s": ref}, {"s": ids}, "t2", B=40)
    assert f2["mu"] is None and f2["T"] is None
    assert f1["groups"]["s"]["xi"] == f2["groups"]["s"]["xi"] and f1["groups"]["s"]["sigma"] == f2["groups"]["s"]["sigma"]
    c1 = hier.fit_groups({"s": ref}, {"s": ids}, "t1", B=40)["s"]
    c2 = build("group_calibrator", {"name": "fisher_hb", "args": {"B": 40}}).fit_groups({"s": ref}, {"s": ids})["s"]
    s = _probe(np.random.default_rng(3), ref)
    assert np.array_equal(c1(s), c2(s))


def test_eb_shrinks_small_scenes_more():
    rng = np.random.default_rng(4)
    Js = np.array([4, 6, 10, 20, 40, 80])
    theta = np.stack([rng.normal([0.4, 0.0], [0.25, 0.3]) for _ in Js])
    Sigma = np.stack([np.diag([0.6, 0.8]) / J for J in Js])            # Sigma_s ~ 1/J_s
    mu, T = hier.eb_reml(theta, Sigma)
    assert np.all(np.diag(T) > 0)
    w = []
    for th, S in zip(theta, Sigma):
        post, V = hier.posterior(th, S, mu, T)
        a = (post - th) / (mu - th)                                       # pull towards mu, Sigma / (Sigma + tau^2)
        assert np.allclose(a, np.diag(S) / (np.diag(S) + np.diag(T)))
        assert np.allclose(V, np.linalg.inv(np.linalg.inv(T) + np.linalg.inv(S)))
        w.append(a)
    w = np.array(w)
    assert np.all(np.diff(w, axis=0) < 0)                                # smaller J -> larger pull
    # end to end: a 4-video scene is pulled more than an 80-video scene
    sc = {f"s{J}": _scene(rng, J, xi=x) for J, x in ((4, 0.1), (10, 0.5), (20, 0.3), (80, 0.7))}
    f = hier.hb_fit({g: r for g, (r, _) in sc.items()}, {g: i for g, (_, i) in sc.items()}, "t2", B=60)
    pull = {g: 1 - np.diag(f["T"] @ np.linalg.inv(f["T"] + p["Sigma"])) for g, p in f["groups"].items()}
    assert np.all(pull["s4"] >= pull["s80"]) and pull["s4"][0] > pull["s80"][0]    # tau_rho may be 0: both 1
    assert np.diag(f["groups"]["s4"]["Sigma"]).sum() > np.diag(f["groups"]["s80"]["Sigma"]).sum()


def test_truncated_mode():
    rng = np.random.default_rng(5)
    for _ in range(50):
        A = rng.normal(size=(2, 2))
        V = A @ A.T + 0.05 * np.eye(2)
        th = rng.normal([0.0, 0.0], [0.5, 1.0])
        m = hier.truncated_mode(th, V)
        assert m[0] >= 0
        if th[0] >= 0:
            assert np.array_equal(m, th)
        else:                                                            # argmax of the density on the line xi = 0
            P = np.linalg.inv(V)
            rho = np.linspace(m[1] - 3, m[1] + 3, 20001)
            q = [(np.array([0.0, r]) - th) @ P @ (np.array([0.0, r]) - th) for r in rho]
            assert m[0] == 0.0 and abs(rho[int(np.argmin(q))] - m[1]) < 1e-3
    assert np.array_equal(hier.truncated_mode(np.array([-0.2, 1.0]), np.diag([0.0, 0.3])), np.array([0.0, 1.0]))


def test_grid_adapter_group_calibrator(tmp_path):
    rng = np.random.default_rng(6)
    vids = ["01_0001", "01_0002", "02_0001"]
    ids = {}
    for sc in ("01", "02"):
        ref, i = _scene(rng, 6)
        ids[sc] = i
        np.savez(tmp_path / f"{sc}.npz", **{"ref/flow_rx_m": ref},
                 **{f"flow_rx_m/{v}": rng.gamma(2.0, 1.5, 40) for v in vids if v.startswith(sc)})
        np.savez(tmp_path / f"{sc}_ids.npz", **{"ids/flow_rx_m": i})
        np.savez(tmp_path / f"{sc}_bad.npz", **{"ids/flow_rx_m": i[:-1]})
    spec = [str(tmp_path / "{sc}.npz")]
    with pytest.raises(ValueError):
        PrecomputedChannels(spec, "fisher_hb")                           # no ids: no silent fallback
    with pytest.raises(ValueError):
        PrecomputedChannels(spec, "fisher_hb", [str(tmp_path / "{sc}_bad.npz")]).calibrated(vids, True)
    want = PrecomputedChannels(spec).calibrated(vids, True)
    t0 = {"name": "fisher_hb", "args": {"mode": "t0"}}
    got = PrecomputedChannels(spec, t0, [str(tmp_path / "{sc}_ids.npz")]).calibrated(vids, True)
    assert list(got) == list(want) == ["Fd"] and list(got["Fd"]) == list(want["Fd"])
    assert all(np.array_equal(got["Fd"][v], want["Fd"][v]) for v in vids)
    t2 = PrecomputedChannels(spec, {"name": "fisher_hb", "args": {"B": 30}},
                             [str(tmp_path / "{sc}_ids.npz")]).calibrated(vids, True)
    for v in vids:                                                       # same body below u
        m = want["Fd"][v] < np.log(10) - 1e-9
        assert m.any() and np.array_equal(t2["Fd"][v][m], want["Fd"][v][m])


def test_refs_only_and_ids(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    lens = {"01_001": 40, "01_002": 31, "01_003": 25}
    for v, n in lens.items():
        (tmp_path / "frames" / v).mkdir(parents=True)
        for t in range(n):
            (tmp_path / "frames" / v / f"{t:03d}.jpg").touch()
    n_i = sum(len(range(15, n, 9)) for n in lens.values())
    n_f = sum(len(range(0, n, 3)) for n in lens.values())
    np.savez(tmp_path / "scores.npz", **{"ref/frx_bayes": rng.normal(size=n_i), "ref/flow_ubm_m": rng.normal(size=n_f),
                                         "ref/flow_rx_m": rng.normal(size=n_f), "frx_bayes/01_0100": rng.normal(size=9),
                                         "gt/01_0100": np.zeros(9)})
    seen = []
    orig = np.lib.npyio.NpzFile.__getitem__
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", lambda self, k: (seen.append(k), orig(self, k))[1])
    grid_refs_only.copy_refs(tmp_path / "scores.npz", tmp_path / "refs.npz")
    assert seen and all(k.startswith("ref/") for k in seen)
    with np.load(tmp_path / "refs.npz") as R:
        assert sorted(R.files) == ["ref/flow_rx_m", "ref/flow_ubm_m", "ref/frx_bayes"]
    with pytest.raises(FileExistsError):
        grid_refs_only.copy_refs(tmp_path / "scores.npz", tmp_path / "refs.npz")
    out = grid_ref_ids.build([str(tmp_path / "refs.npz")], str(tmp_path / "frames"), "01")
    assert list(out["videos"]) == list(lens)
    assert np.array_equal(out["ids/frx_bayes"], np.repeat([0, 1, 2], [3, 2, 2]))
    assert np.array_equal(out["t/frx_bayes"], [15, 24, 33, 15, 24, 15, 24])
    assert np.array_equal(out["ids/flow_rx_m"], np.repeat([0, 2, 1], [14, 9, 11]))   # even, then odd videos
    with pytest.raises(ValueError):                                     # not a blind copy
        grid_ref_ids.build([str(tmp_path / "scores.npz")], str(tmp_path / "frames"), "01")
