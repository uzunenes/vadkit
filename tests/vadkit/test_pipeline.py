"""Pipeline: CLI eval reproduces the archived Ped2 obj2 results, streaming equals offline, state round trip, and the
fit / score / fusion path equals tools/objscore2.py + tools/objfuse.py (+ objfuse_sht.py per scene) on synthetic data.

The Ped2 fixture replaces the held-out (K, kappa0, nu0) search by the archived choice (hyper.json) to keep the file
fast on a CPU, so it cannot see rng errors; the synthetic caps test does (rows really drawn), and the search itself on
the Ped2 data is test_golden_scoring.py (slow). VADKIT_FULL_SELECTION=1 runs the real search here too. Exact
comparisons with the reference scripts run both sides on the CPU."""
import functools
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

import vadkit.density.niw as niw
from vadkit.cli import main
from vadkit.core.interfaces import AttributeExtractor, Objects
from vadkit.core.registry import register
from vadkit.eval.protocol import evaluate
from vadkit.io.frames import frame_paths, read
from vadkit.perception.flow import DISFlow
from vadkit.pipeline.offline import FittedState, ObjectBranch, Perception, load_config, offline_smoother
from vadkit.pipeline.stream import StreamingVAD, objects_at
from vadkit.temporal.median import causal_median

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_golden_scoring import _run, _synthetic  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FEAT = Path(os.environ.get("VADKIT_PED2_FEAT", "data/feat2tmp/feat2/ped2"))
GT = Path(os.environ.get("VADKIT_PED2_GT", "data/cache/ped2/gt.json"))
FRAMES = Path(os.environ.get("VADKIT_PED2_FRAMES", "data/frames/ped2"))
ARCHIVE = ROOT / "paper/ieee_access/kanit/vast_20261104/sonuclar/obj2/ped2"
ATTRS = ("ovel", "opose", "oapp")
ARCHIVED = all((ARCHIVE / f).is_file() for f in ("fuse.txt", "fuse.json", "hyper.json"))
needs_ped2 = pytest.mark.skipif(not ((FEAT / "training").is_dir() and GT.is_file() and ARCHIVED),
                                reason="Ped2 released features / archived obj2 results missing")


# ------------------------------------------------------------------------------------------------ Ped2 fixture
@pytest.fixture(scope="module")
def ped2(tmp_path_factory):
    """ObjectBranch(released_features) fitted on the released Ped2 training features, saved to a state dir."""
    cfg = load_config("released_features")
    branch = ObjectBranch(cfg)
    train, test = branch.load(FEAT)
    hyper = json.loads((ARCHIVE / "hyper.json").read_text())
    by_dim = {h["dim"]: (h["K"], h["kappa0"], h["nu0"] - h["dim"] - 1) for h in hyper.values()}
    archived = lambda fit_sets, eval_sets, device="cpu": (by_dim[fit_sets[0].shape[1]], {})
    with pytest.MonkeyPatch.context() as mp:
        if os.environ.get("VADKIT_FULL_SELECTION") != "1":
            mp.setattr(niw, "select_mixture", archived)
        state = branch.fit(train)
    d = tmp_path_factory.mktemp("ped2") / "state"
    state.save(d)
    return SimpleNamespace(cfg=cfg, branch=branch, test=test, state=state, dir=d, hyper=hyper)


def fused_offline(branch, state, test, warmup=4):
    """{video: causal median of the fused calibrated channels from frame warmup on}, and the channels."""
    z = branch.calibrate(branch.score(test, state), state)
    return {v: causal_median(f[warmup:]) for v, f in branch.fuse(z)[1].items()}, z


# ------------------------------------------------------------------------------------------------ (1) archived Ped2
@needs_ped2
def test_eval_reproduces_archived_ped2(ped2, capsys):
    res = main(["eval", "--config", "released_features", "--features", str(FEAT), "--gt", str(GT),
                "--state", str(ped2.dir)])
    printed = capsys.readouterr().out
    row = r"^(\S+)\s+causal (\S+)\s+macro (\S+)\s+offline (\S+)$"
    rows = {m[0]: tuple(map(float, m[1:])) for m in re.findall(row, (ARCHIVE / "fuse.txt").read_text(), re.M)}
    full = json.loads((ARCHIVE / "fuse.json").read_text())
    key = next(k for k in full if sorted(k.split("+")) == sorted(ATTRS))      # the pure three-attribute fusion
    assert key == "+".join(ATTRS) and set(res) == {*ATTRS, key} and key in printed
    assert {a: round(res[a]["causal"], 4) for a in ATTRS} == {"ovel": 0.8616, "opose": 0.5619, "oapp": 0.9424}
    for k in (*ATTRS, key):
        assert tuple(round(res[k][m], 4) for m in ("causal", "causal_macro", "offline")) == rows[k], k
        assert all(abs(res[k][m] - full[k][m]) < 1e-9 for m in full[k]), k
    assert round(res[key]["causal"], 4) == round(full[key]["causal"], 4) == 0.9385
    if os.environ.get("VADKIT_FULL_SELECTION") == "1":                       # else hyper.json is the fixture's input
        for a in ATTRS:
            assert {k: ped2.state.groups["all"][a].hyper[k] for k in ped2.hyper[a]} == ped2.hyper[a]


# ------------------------------------------------------------------------------------------------ (2) stream == offline
@needs_ped2
def test_streaming_equals_offline(ped2):
    vid = "Test004"
    r = ped2.test[vid]
    want, z = fused_offline(ped2.branch, ped2.state, {vid: r})
    sv = StreamingVAD(ped2.state)                                             # gap: the state's (startup 1)
    with pytest.raises(ValueError):
        StreamingVAD(ped2.state, gap=2)
    hw = tuple(r["frame_hw"])
    res = [sv.step_objects(t, objects_at(r, t), hw) for t in range(int(r["n_frames"]))]
    assert all(x.score is None for x in res[:4]) and [x.t for x in res] == list(range(len(res)))
    got = np.array([x.score for x in res[4:]])
    assert np.abs(got - want[vid]).max() < 1e-9
    for a in ATTRS:
        assert np.abs(np.array([x.channels[a] for x in res]) - z[a][vid]).max() < 1e-9
    sv.reset()
    assert sv.step_objects(0, objects_at(r, 0), hw).score is None and len(sv.smoother.buf) == 0


# ------------------------------------------------------------------------------------------------ (3) state round trip
@needs_ped2
def test_state_round_trip(ped2):
    assert {p.name for p in ped2.dir.iterdir()} == {"state.npz", "state.json"}
    loaded = FittedState.load(ped2.dir)
    assert loaded.gap == ped2.state.gap == 1 and loaded.channels == list(ATTRS)
    a, b = (ped2.branch.score(ped2.test, s) for s in (ped2.state, loaded))
    za, zb = (ped2.branch.calibrate(x, s) for x, s in ((a, ped2.state), (b, loaded)))
    assert a.keys() == b.keys() == set(ATTRS)
    assert all(np.array_equal(a[k][v], b[k][v]) and np.array_equal(za[k][v], zb[k][v]) for k in a for v in a[k])
    for name, st in loaded.groups["all"].items():
        ref = ped2.state.groups["all"][name]
        assert np.array_equal(st.ref, ref.ref) and st.floor == ref.floor and st.hyper == ref.hyper


# ------------------------------------------------------------------------------------------------ synthetic references
def _grid(path: Path, vids: dict, rng, keys=("frx_bayes", "flow_ubm_m", "flow_rx_m", "flow_ubm")):
    """Archived-style cell scores.npz: '<key>/<video>' and 'ref/<key>'."""
    path.parent.mkdir(parents=True, exist_ok=True)
    S = {}
    for k in keys:
        S[f"ref/{k}"] = rng.gamma(2.0, 1.0, 300)
        S.update({f"{k}/{v}": rng.gamma(2.0, 1.0, n) + 2.0 * (np.arange(n) % 7 == 0) for v, n in vids.items()})
    np.savez(path, **S)


def _config(tmp: Path) -> Path:
    """released_features with a small appearance PCA (8, as tools/objscore2.py --pca 8)."""
    cfg = load_config("released_features")
    cfg["attributes"][2]["args"]["dim"] = 8
    p = tmp / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_cli_equals_objscore2_objfuse(tmp_path):
    """fit / score / eval with grid channels == tools/objscore2.py scores.npz and tools/objfuse.py fuse.json exactly."""
    *_, gt = _synthetic(tmp_path, [f"Train{i:03d}" for i in range(1, 6)], ["Test001", "Test002", "Test003"], 0.6)
    rng = np.random.default_rng(5)
    n = {v: len(g) for v, g in gt.items()}
    runs = tmp_path / "runs" / "ped2_2d_noadv"
    _grid(runs / "eval_ped2_pool48_mc4" / "scores.npz", n, rng)
    _grid(runs / "eval_ped2_pool48_mc5" / "scores.npz", n, rng)
    _run(ROOT / "tools/objscore2.py", "--feat", tmp_path / "feat", "--gt", tmp_path / "gt.json", "--out",
         tmp_path / "obj", "--pca", 8)
    (tmp_path / "none").mkdir()
    _run(ROOT / "tools/objfuse.py", "--ds", "ped2", "--runs", tmp_path / "runs", "--obj", tmp_path / "none",
         "--gt", tmp_path / "gt.json", "--extra", tmp_path / "obj/scores.npz", "--max-combo", "6",
         "--out", tmp_path / "fuse.json")
    cfg, feat, st = str(_config(tmp_path)), str(tmp_path / "feat"), str(tmp_path / "state")
    cpu = ["--device", "cpu"]
    main(["fit", "--config", cfg, "--features", feat, "--out", st, *cpu])
    main(["score", "--config", cfg, "--features", feat, "--state", st, "--out", str(tmp_path / "s.npz"), *cpu])
    want, got = np.load(tmp_path / "obj/scores.npz"), np.load(tmp_path / "s.npz")
    assert set(got.files) == set(want.files) and all(np.array_equal(got[k], want[k]) for k in want.files)
    res = main(["eval", "--config", cfg, "--features", feat, "--gt", str(tmp_path / "gt.json"), *cpu,  # fits itself
                "--extra-channels", f"{runs}/eval_ped2_pool48_mc5/scores.npz:flow_ubm_m=Fs,flow_rx_m=Fd",
                f"{runs}/eval_ped2_pool48_mc4/scores.npz:frx_bayes=I"])            # objfuse's I, Fs, Fd order anyway
    ref = json.loads((tmp_path / "fuse.json").read_text())
    assert list(res) == ["I", "Fs", "Fd", *ATTRS, "I+Fs+Fd+ovel+opose+oapp"]
    assert all(res[k] == ref[k] for k in res)
    with pytest.raises(ValueError):                                           # frx_bayes of both files -> I twice
        main(["eval", "--config", cfg, "--features", feat, "--gt", str(tmp_path / "gt.json"), "--state", st, *cpu,
              "--extra-channels", f"{runs}/eval_ped2_pool48_mc4/scores.npz", f"{runs}/eval_ped2_pool48_mc5/scores.npz"])


def test_subsample_caps_equal_objscore2(tmp_path):
    """Caps that bite (PCA rows, held-out rows per fold): the shared rng must draw the same rows as tools/objscore2.py
    (a copy with its 50000 / 20000 caps lowered; vadkit through the config: appearance max_rows, density max_eval),
    so scores and hyper-parameters agree bit for bit."""
    train, _, gt = _synthetic(tmp_path, [f"Train{i:03d}" for i in range(1, 6)], ["Test001", "Test002"], 0.6, seed=4)
    names = sorted(train)
    assert sum(len(r["f"]) for r in train.values()) > 100                    # PCA cap 100
    assert min(int((train[v]["pose_ok"] > 0).sum()) for v in names) * 2 > 40  # every eval fold above the cap 40
    src = (ROOT / "tools/objscore2.py").read_text()
    assert src.count("[:50000]") == src.count("[:20000]") == 1
    (tmp_path / "objscore2_caps.py").write_text(src.replace("[:50000]", "[:100]").replace("[:20000]", "[:40]"))
    _run(tmp_path / "objscore2_caps.py", "--feat", tmp_path / "feat", "--gt", tmp_path / "gt.json", "--out",
         tmp_path / "obj", "--pca", 8, PYTHONPATH=str(ROOT))
    cfg = load_config("released_features")
    cfg["attributes"][2]["args"].update(dim=8, max_rows=100)
    cfg["density"] = {"name": "niw", "args": {"max_eval": 40}}               # forwarded to the held-out selection
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    args = ["--config", str(tmp_path / "cfg.yaml"), "--features", str(tmp_path / "feat"), "--device", "cpu"]
    main(["fit", *args, "--out", str(tmp_path / "state")])
    main(["score", *args, "--state", str(tmp_path / "state"), "--out", str(tmp_path / "s.npz")])
    want, got = np.load(tmp_path / "obj/scores.npz"), np.load(tmp_path / "s.npz")
    assert set(got.files) == set(want.files) and all(np.array_equal(got[k], want[k]) for k in want.files)
    hyper = json.loads((tmp_path / "obj/hyper.json").read_text())
    state = FittedState.load(tmp_path / "state")
    assert {a: {k: state.groups["all"][a].hyper[k] for k in h} for a, h in hyper.items()} == hyper


def test_scenes_equal_objscore2_prefix_and_objfuse_sht(tmp_path):
    """--scenes: per-scene fit (objscore2 --prefix), per-scene calibration and pooled AUC (tools/objfuse_sht.py). Scene
    07 has no matched persons, so opose is not a channel of the state: score files, eval and stream drop it everywhere
    (objfuse_sht intersection); the other attributes equal objscore2 --prefix bit for bit."""
    scenes = [f"{i:02d}" for i in range(1, 13)]                               # objfuse_sht.py reads scenes 01-12
    _, test, gt = _synthetic(tmp_path, [f"{s}_{i:03d}" for s in scenes for i in range(2)],
                             [f"{s}_0000" for s in scenes], 0.6, seed=2)
    for p in (tmp_path / "feat/training").glob("07_*.npz"):
        np.savez(p, **{**np.load(p), "pose_ok": np.zeros_like(np.load(p)["pose_ok"])})
    rng = np.random.default_rng(3)
    for s in scenes:
        g = {v: len(x) for v, x in gt.items() if v.startswith(s + "_")}
        _grid(tmp_path / f"runs/shtech_{s}_2d_noadv/eval_shtech_{s}_pool48_mc5/scores.npz", g, rng,
              ("frx_bayes", "flow_ubm_m", "flow_rx_m"))
        (tmp_path / f"cache/shtech_{s}").mkdir(parents=True)
        (tmp_path / f"cache/shtech_{s}/gt.json").write_text(json.dumps({v: gt[v] for v in g}))
    cfg, feat, st = str(_config(tmp_path)), str(tmp_path / "feat"), str(tmp_path / "state")
    common = ["--config", cfg, "--features", feat, "--device", "cpu"]
    main(["fit", *common, "--out", st, "--scenes"])
    with pytest.raises(SystemExit):                                           # one file per scene needs '{sc}'
        main(["score", *common, "--state", st, "--out", str(tmp_path / "all.npz")])
    main(["score", *common, "--state", st, "--out", str(tmp_path / "obj/{sc}/scores.npz")])
    for s in ("01", "07"):                                                    # per-scene rng, grouping, attributes
        _run(ROOT / "tools/objscore2.py", "--feat", tmp_path / "feat", "--gt", tmp_path / "gt.json", "--out",
             tmp_path / f"ref{s}", "--pca", 8, "--prefix", f"{s}_")
        want, got = np.load(tmp_path / f"ref{s}/scores.npz"), np.load(tmp_path / f"obj/{s}/scores.npz")
        assert ("ref/opose" in want.files) == (s == "01")
        assert set(got.files) == {k for k in want.files if "opose" not in k.split("/")}
        assert all(np.array_equal(got[k], want[k]) for k in got.files)
    _run(ROOT / "tools/objfuse_sht.py", "--runs", tmp_path / "runs", "--cache", tmp_path / "cache", "--extra",
         str(tmp_path / "obj/{sc}/scores.npz"), "--out", tmp_path / "fuse.json")
    grid = str(tmp_path / "runs/shtech_{sc}_2d_noadv/eval_shtech_{sc}_pool48_mc5/scores.npz")
    res = main(["eval", *common, "--gt", str(tmp_path / "gt.json"), "--state", st, "--extra-channels", grid])
    ref = json.loads((tmp_path / "fuse.json").read_text())
    assert list(res) == ["Fd", "Fs", "I", "oapp", "ovel", "Fd+Fs+I+oapp+ovel"]
    assert all(res[k] == ref[k] for k in res)
    with pytest.raises(FileNotFoundError):                                    # a template matching no scene file
        main(["eval", *common, "--gt", str(tmp_path / "gt.json"), "--state", st, "--extra-channels",
              str(tmp_path / "typo_{sc}/scores.npz")])
    with pytest.warns(UserWarning, match="no file for scenes"):               # files for 01 and 07 only
        main(["eval", *common, "--gt", str(tmp_path / "gt.json"), "--state", st, "--extra-channels",
              str(tmp_path / "ref{sc}/scores.npz")])
    state = FittedState.load(st)                                              # stream == offline on scene 01
    branch, v = ObjectBranch(state.config), "01_0000"
    sv = StreamingVAD(state, scene="01")
    assert state.gap == sv.gap == 2 and list(sv.attrs) == state.channels == ["ovel", "oapp"]
    want, _ = fused_offline(branch, state, {v: test[v]})
    got = [sv.step_objects(t, objects_at(test[v], t), (240, 360)).score for t in range(int(test[v]["n_frames"]))]
    assert np.abs(np.array(got[4:]) - want[v]).max() < 1e-9


# ------------------------------------------------------------------------------------------------ config stages
@register("attribute", "_bare")
class BareAttribute(AttributeExtractor):
    def __call__(self, objs, flow, frame_hw):
        return np.zeros((len(objs), 1))


@register("attribute", "_fitted_without_state")
class FittedWithoutState(BareAttribute):
    raw_key = "app"                                   # fitted (AttributeExtractor hook), but no state() / from_state()

    def record_rows(self, rec):
        return rec["f"], np.zeros((len(rec["f"]), 1))


def test_config_stages():
    """One temporal spec (causal_median, integer k; default 9) drives offline eval and the stream; density spec forms
    and args; clear errors for a stage gap, feature train_every / startup and extractors outside the hooks."""
    rng = np.random.default_rng(1)
    z = {"a": {f"v{i}": rng.gamma(2.0, 1.0, 80) for i in range(3)}}
    gt = {v: (np.arange(80) % 9 < 3).astype(int).tolist() for v in z["a"]}
    base = load_config("released_features")
    no_temporal = {k: v for k, v in base.items() if k != "temporal"}
    stream = lambda cfg: StreamingVAD(FittedState(cfg, {"all": {}}, False, 1))
    assert ObjectBranch(no_temporal).evaluate(z, gt)["a"] == evaluate(z["a"], gt, 4)       # default: median 9
    assert stream(no_temporal).smoother.k == 9
    res = {}
    for k in (9, 5):
        spec = {"name": "causal_median", "args": {"k": k}}
        assert all(np.array_equal(offline_smoother(spec)(s), causal_median(s, k)) for s in z["a"].values())
        res[k] = ObjectBranch({**base, "temporal": spec}).evaluate(z, gt)["a"]
        assert res[k] == evaluate(z["a"], gt, 4, functools.partial(causal_median, k=k))
        assert stream({**base, "temporal": spec}).smoother.k == k
    assert res[9] == evaluate(z["a"], gt, 4) != res[5]
    for bad in ("ema", {"name": "causal_median", "args": {"k": 4.5}}, {"name": "causal_median", "args": {"w": 3}}):
        for build_it in (ObjectBranch, stream):
            with pytest.raises(ValueError, match="causal_median"):
                build_it({**base, "temporal": bad})
    for spec in ("niw", {"name": "niw"}, {"name": "niw", "args": {"max_eval": 100, "min_rows": 5}},
                 {"name": "niw", "args": {"K": 2, "kappa0": 1.0, "nu0": 40.0}}):
        assert ObjectBranch({**base, "density": spec}).density["name"] == "niw"
    for spec in ({"name": "niw", "args": {"K": 4}}, {"name": "niw", "args": {"seed": 1}}, "gmm"):
        with pytest.raises(ValueError):
            ObjectBranch({**base, "density": spec})
    for stage, name in (("detector", "_replay"), ("flow", "dis")):           # a spec gap must equal the pipeline's
        with pytest.raises(ValueError, match="pipeline gap"):
            Perception({**base, "detector": {"name": "_replay"}, stage: {"name": name, "args": {"gap": 2}}}, 1)
    rec = lambda **kw: {"f": np.zeros(3, np.int32), "box": np.zeros((3, 4), np.float32), "n_frames": 5, **kw}
    with pytest.raises(ValueError, match="multiple"):
        ObjectBranch(base).fit({"Train001": rec(startup=1, train_every=5)})
    with pytest.raises(ValueError, match="one frame gap"):
        ObjectBranch(base).fit({"Train001": rec(startup=1), "Train002": rec(startup=2)})
    for kind, match in (("_bare", "record_rows"), ("_fitted_without_state", "state")):
        with pytest.raises(TypeError, match=match):
            ObjectBranch({**base, "attributes": [{"name": "x", "kind": kind}]}).fit(
                {"Train001": rec(startup=1, app=np.zeros((3, 4), np.float16))})


def test_density_args_forwarded(tmp_path):
    """Fixed (K, kappa0, nu0) from the config's density args reach every fit (no held-out selection)."""
    train, _, _ = _synthetic(tmp_path, [f"Train{i:03d}" for i in range(1, 5)], ["Test001"], 0.6)
    cfg = load_config("released_features")
    cfg["attributes"] = cfg["attributes"][:1]                                 # ovel, 8-d
    cfg["density"] = {"name": "niw", "args": {"K": 2, "kappa0": 3.0, "nu0": 12.0}}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(niw, "select_mixture", lambda *a, **k: pytest.fail("selection ran despite fixed hyper-parameters"))
        h = ObjectBranch(cfg, device="cpu").fit(train).groups["all"]["ovel"].hyper
    assert {k: h[k] for k in ("K", "kappa0", "nu0", "dim")} == {"K": 2, "kappa0": 3.0, "nu0": 12.0, "dim": 8}


# --------------------------------------------------------------------------------------------- extract / step / bench
class _Replay:
    """Released objects of one Ped2 video found by frame content, standing in for YOLO (CPU numerics differ)."""
    rec: dict = {}
    index: dict = {}
    t = -1

    @classmethod
    def at(cls) -> Objects:
        return objects_at(cls.rec, cls.t)


@register("detector", "_replay")
class ReplayDetector:
    def detect(self, bgr):
        _Replay.t = _Replay.index[hash(bgr.tobytes())]
        o = _Replay.at()
        return Objects(box=o.box, cls=o.cls, conf=o.conf, src=o.src)


@register("pose", "_replay")
class ReplayPose:
    def attach(self, bgr, objs):
        o = _Replay.at()
        assert np.array_equal(o.box, objs.box)
        objs.pose, objs.pose_ok = o.pose, o.pose_ok
        return objs


@register("encoder", "_replay")
class ReplayEncoder:
    dim = 512

    def encode(self, bgr, boxes):
        return _Replay.at().attrs["app"]


def _replay_frames(tmp: Path, vid: str, split: str) -> Path:
    """Frames root with a symlink to one real Ped2 video, indexed for the replay back-ends."""
    (tmp / split / "frames").mkdir(parents=True)
    (tmp / split / "frames" / vid).symlink_to(FRAMES / split / "frames" / vid)
    _Replay.rec = dict(np.load(FEAT / split / f"{vid}.npz"))
    _Replay.index = {hash(read(p).tobytes()): t for t, p in enumerate(frame_paths(tmp, split)[vid])}
    assert len(_Replay.index) == int(_Replay.rec["n_frames"])
    return tmp


@needs_ped2
@pytest.mark.skipif(not FRAMES.is_dir(), reason="Ped2 frames missing")
def test_extract_step_and_bench(ped2, tmp_path):
    """extract with replayed boxes / pose / app and real DIS: released vel at t >= gap, no rows before, train_every
    subsample, resume; StreamingVAD.step on the frames == offline on the extracted npz; bench JSON."""
    cfg = {**ped2.cfg, "detector": {"name": "_replay"}, "pose": {"name": "_replay"}, "flow": {"name": "dis"},
           "encoder": {"name": "_replay"}}
    branch = ObjectBranch(cfg)
    for vid, split, every in (("Train001", "training", 3), ("Test004", "testing", 1)):
        root = _replay_frames(tmp_path / f"frames_{vid}", vid, split)
        assert branch.extract(root, split, tmp_path / "feat", 1, every, "1/2") == []      # the only video is shard 0
        assert [p.name for p in branch.extract(root, split, tmp_path / "feat", 1, every)] == [f"{vid}.npz"]
        assert branch.extract(root, split, tmp_path / "feat", 1, every) == []
        got, rel = dict(np.load(tmp_path / "feat" / split / f"{vid}.npz")), dict(np.load(FEAT / split / f"{vid}.npz"))
        k = (rel["f"] >= 1) & (rel["f"] % every == 0)
        assert set(got) == set(rel) | {"train_every"} and int(got["train_every"]) == every
        assert all(got[c].dtype == rel[c].dtype and np.array_equal(got[c], rel[c][k])
                   for c in ("f", "box", "src", "cls", "conf", "vel", "pose", "pose_ok", "app"))
        assert int(got["startup"]) == 1 and int(got["n_frames"]) == int(rel["n_frames"])
    assert not list((tmp_path / "feat").rglob("*.part.npz"))
    ext = ObjectBranch.load(tmp_path / "feat")[1]
    want, _ = fused_offline(branch, ped2.state, ext)
    sv = StreamingVAD(ped2.state, cfg)
    got = [sv.step(read(p)).score for p in frame_paths(root, "testing")["Test004"]]
    assert got[:4] == [None] * 4 and np.abs(np.array(got[4:]) - want["Test004"]).max() < 1e-9
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    main(["bench", "--config", str(tmp_path / "cfg.yaml"), "--frames", str(root), "--dataset", "ped2", "--state",
          str(ped2.dir), "--out", str(tmp_path / "bench.json"), "--warm", "2", "--reps", "2", "--n", "5"])
    b = json.loads((tmp_path / "bench.json").read_text())
    assert len(b["reps"]) == 2 and b["gap"] == 1 and b["frame_hw"] == [240, 360]
    assert set(b["reps"][0]) == {"decode", "detect", "pose_wait", "pose", "flow", "embed", "score", "total"}
    assert all(0 <= s["median_ms"] <= s["p95_ms"] for rep in b["reps"] for s in rep.values())


# --------------------------------------------------------------------------------------------- flow batching (synthetic)
@register("detector", "_every_frame")
class EveryFrameDetector:
    """Two fixed boxes on every frame t with t % 4 != 3 (t counted from reset), none on the others."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.t = -1

    def detect(self, bgr):
        self.t += 1
        n = 0 if self.t % 4 == 3 else 2
        box = np.array([[0.1, 0.1, 0.5, 0.6], [0.4, 0.3, 0.9, 0.9]], np.float32)[:n]
        return Objects(box=box, cls=np.zeros(n, np.int16), conf=np.ones(n, np.float32), src=np.zeros(n, np.uint8))


@register("flow", "_counting")
class CountingFlow(DISFlow):
    """DIS that records the pairs per call; batch as NeuFlow / RAFT (default 8)."""
    calls: list = []

    def __init__(self, batch: int = 8):
        super().__init__()
        self.batch = batch

    def flow(self, prev_bgr, cur_bgr):
        CountingFlow.calls.append(len(prev_bgr))
        return super().flow(prev_bgr, cur_bgr)


def test_extract_batches_flow_pairs(tmp_path):
    """Per-frame Perception: flow batch 1 (unless the spec sets one); extract: pairs in batches of the back-end, rows
    equal to the per-frame path; train_every defaults to the config's; bad shards raise; step resizes to the width."""
    n_frames, gap = 26, 2
    vd = tmp_path / "frames/testing/frames/Test001"
    vd.mkdir(parents=True)
    tex = cv2.resize(np.random.default_rng(0).integers(0, 255, (30, 50, 3)).astype(np.uint8), (200, 120))
    for t in range(n_frames):                                                 # texture moving 2 px per frame
        cv2.imwrite(str(vd / f"{t}.jpg"), np.roll(tex, 2 * t, axis=1)[:, :96][:64])
    (tmp_path / "frames/training/frames").mkdir(parents=True)
    (tmp_path / "frames/training/frames/Train001").symlink_to(vd)
    cfg = {**load_config("released_features"), "detector": {"name": "_every_frame"}, "flow": {"name": "_counting"},
           "train_every": 2}
    perc = Perception(cfg, gap)
    assert perc.flow.batch == 1
    assert Perception({**cfg, "flow": {"name": "_counting", "args": {"batch": 3}}}, gap).flow.batch == 3
    want = [o.attrs["vel"] for o in (perc(read(p)) for p in frame_paths(tmp_path / "frames", "testing")["Test001"])
            if len(o)]
    CountingFlow.calls.clear()
    branch = ObjectBranch(cfg)
    assert [p.stem for p in branch.extract(tmp_path / "frames", "testing", tmp_path / "feat", gap)] == ["Test001"]
    with_objs = [t for t in range(gap, n_frames) if t % 4 != 3]
    assert len(with_objs) == 18 and CountingFlow.calls == [8, 8, 2]           # consecutive groups (velpilot flowvel)
    got = dict(np.load(tmp_path / "feat/testing/Test001.npz"))
    assert np.array_equal(got["vel"], np.concatenate(want)) and np.array_equal(np.unique(got["f"]), with_objs)
    assert int(got["train_every"]) == 1 and int(got["startup"]) == gap
    branch.extract(tmp_path / "frames", "training", tmp_path / "feat", gap)  # train_every: the config's 2
    tr = dict(np.load(tmp_path / "feat/training/Train001.npz"))
    assert int(tr["train_every"]) == 2 and len(tr["f"]) and not (tr["f"] % 2).any()
    for bad in ("2/2", "-1/2", "1"):
        with pytest.raises(ValueError, match="shard"):
            branch.extract(tmp_path / "frames", "testing", tmp_path / "other", gap, shard=bad)
    sv = StreamingVAD(FittedState(cfg, {"all": {}}, False, gap), {**cfg, "width": 48})
    sv.step_objects = lambda t, objs, hw: hw                                  # the frame size step hands on
    assert sv.step(read(vd / "0.jpg")) == (32, 48) and sv.perception.flow.batch == 1
