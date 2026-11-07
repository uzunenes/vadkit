"""Attribute scoring equals tools/objscore2.py and tools/velpilot.py score exactly (synthetic features, scripts run as
subprocesses), and reproduces the archived Ped2 causal AUCs of the released light features (skipped without data)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from vadkit.attributes.appearance import AppearancePCA
from vadkit.attributes.pose import PoseKeypoints
from vadkit.calibration.fisher import FisherCalibrator
from vadkit.density.niw import VideoRows, fit_attribute, fit_attribute_by_scene
from vadkit.eval.protocol import evaluate

ROOT = Path(__file__).resolve().parents[2]
FEAT = Path(os.environ.get("VADKIT_PED2_FEAT", "data/feat2tmp/feat2/ped2"))
GT = Path(os.environ.get("VADKIT_PED2_GT", "data/cache/ped2/gt.json"))
ARCHIVE = ROOT / "paper/ieee_access/kanit/vast_20261104/sonuclar/obj2/ped2"
DEV = "cuda" if torch.cuda.is_available() else "cpu"        # archived results: objscore2 on its default device


def load(d: Path) -> dict:
    return {p.stem: dict(np.load(p)) for p in sorted(d.glob("*.npz"))}


def score_like_objscore2(train: dict, test: dict, vids: list, attrs=("ovel", "opose", "oapp"), pca: int = 64,
                         device: str = "cpu"):
    """The objscore2 driver on the vadkit API: one default_rng(0) shared by the attributes in order."""
    names = sorted(train)
    rng = np.random.default_rng(0)
    out, hyper = {}, {}
    for attr in attrs:
        if attr == "ovel":
            rows = lambda r: (r["f"], r["vel"].astype(np.float64))
        elif attr == "opose":
            rows = PoseKeypoints().record_rows
        else:
            rows = AppearancePCA(pca).fit([train[v]["app"] for v in names], rng).record_rows
        vr = lambda r: VideoRows(*rows(r), int(r["n_frames"]), int(r["startup"]))
        fit = fit_attribute({v: vr(train[v]) for v in names}, {v: vr(test[v]) for v in vids}, rng, device)
        if fit is None:
            continue
        out[f"ref/{attr}"] = fit.ref
        out.update({f"{attr}/{v}": s for v, s in fit.scores.items()})
        hyper[attr] = {k: fit.hyper[k] for k in ("K", "kappa0", "nu0", "dim")}
    return out, hyper


def _record(rng, n_frames: int, pose_rate: float, shift: float = 0.0, startup: int = 2):
    k = rng.integers(0, 4, n_frames)
    k[rng.random(n_frames) < 0.15] = 0                                       # frames without objects -> floor
    f = np.repeat(np.arange(n_frames), k).astype(np.int32)
    N = len(f)
    box = np.sort(rng.random((N, 2, 2)), 1).reshape(N, 4)[:, [0, 2, 1, 3]].astype(np.float32)
    vel = (rng.gamma(2.0, 1.0, (N, 8)) + shift * (f[:, None] % 7 == 0)).astype(np.float32)
    return {"f": f, "box": box, "cls": np.zeros(N, np.int16), "conf": np.zeros(N, np.float32),
            "src": np.zeros(N, np.uint8), "vel": vel, "pose": rng.normal(size=(N, 34)).astype(np.float32),
            "pose_ok": (rng.random(N) < pose_rate).astype(np.uint8), "app": rng.normal(size=(N, 32)).astype(np.float16),
            "n_frames": n_frames, "startup": startup, "frame_hw": np.array([240, 360])}


def _synthetic(tmp: Path, train_names, test_names, pose_rate: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    train = {v: _record(rng, int(rng.integers(40, 70)), pose_rate) for v in train_names}
    test = {v: _record(rng, int(rng.integers(40, 70)), pose_rate, shift=3.0) for v in test_names}
    gt = {v: ((np.arange(int(r["n_frames"])) % 7 == 0) & (np.arange(int(r["n_frames"])) > 10)).astype(int).tolist()
          for v, r in test.items()}
    for split, recs in (("training", train), ("testing", test)):
        (tmp / "feat" / split).mkdir(parents=True)
        for v, r in recs.items():
            np.savez(tmp / "feat" / split / f"{v}.npz", **r)
    (tmp / "gt.json").write_text(json.dumps(gt))
    return load(tmp / "feat" / "training"), load(tmp / "feat" / "testing"), gt


def _run(*args, **env):
    """Reference script on the CPU (bit-exact comparisons; CUDA index_add_ is not deterministic)."""
    subprocess.run([sys.executable, *map(str, args)], check=True, capture_output=True, cwd=ROOT,
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": "", **env})


@pytest.mark.parametrize("pose_rate", [0.6, 0.02])         # 0.02: opose skipped, the rng still advances for oapp
def test_equals_objscore2(tmp_path, pose_rate):
    train, test, gt = _synthetic(tmp_path, [f"Train{i:03d}" for i in range(1, 6)], ["Test001", "Test002", "Test003"],
                                 pose_rate)
    _run(ROOT / "tools/objscore2.py", "--feat", tmp_path / "feat", "--gt", tmp_path / "gt.json", "--out",
         tmp_path / "out", "--pca", 8)
    want = dict(np.load(tmp_path / "out/scores.npz"))
    got, hyper = score_like_objscore2(train, test, list(gt), pca=8)
    assert ("ref/opose" in want) == (pose_rate > 0.5)
    assert got.keys() == want.keys()
    assert all(np.array_equal(got[k], want[k]) for k in want)
    assert hyper == json.loads((tmp_path / "out/hyper.json").read_text())


def test_equals_velpilot_scenes(tmp_path):
    """velpilot score --scenes --train-every 2 --drop-startup: per-scene model, reference, calibration, pooled AUC."""
    train, test, gt = _synthetic(tmp_path, [f"{s}_{i:03d}" for s in ("01", "02") for i in range(4)],
                                 [f"{s}_{i:04d}" for s in ("01", "02") for i in range(2)], 0.5, seed=1)
    _run(ROOT / "tools/velpilot.py", "score", "--feat", tmp_path / "feat", "--gt", tmp_path / "gt.json", "--out",
         tmp_path / "res.json", "--train-every", "2", "--drop-startup", "--scenes", "--save-scores", "--cpu")
    want, S = json.loads((tmp_path / "res.json").read_text()), np.load(tmp_path / "res.npz")
    vr = lambda r: VideoRows(r["f"], r["vel"].astype(np.float64), int(r["n_frames"]), int(r["startup"]))
    fits = fit_attribute_by_scene({v: vr(r) for v, r in train.items()}, {v: vr(r) for v, r in test.items()}, "cpu",
                                  train_every=2, drop_startup=True)
    z = {}
    for g, fit in fits.items():
        assert np.array_equal(fit.ref, S[f"ref/{g}"]) and fit.hyper == want["hyper"][g]
        cal = FisherCalibrator().fit(fit.ref)
        for v, s in fit.scores.items():
            assert np.array_equal(s, S[f"s/{v}"])
            z[v] = cal(s)
    res = evaluate({v: z[v] for v in sorted(gt)}, gt, 4)
    assert all(res[k] == want[k] for k in res)


@pytest.mark.slow                                           # full held-out selection on Ped2: ~16 min on a CPU
@pytest.mark.skipif(not (FEAT / "training").is_dir() or not GT.is_file(), reason="Ped2 released features missing")
def test_ped2_archived_aucs():
    """Released Ped2 light features -> archived obj2/ped2 single-channel results (fuse.txt: Fisher, warmup 4)."""
    gt = json.loads(GT.read_text())
    vids = sorted(gt)
    train, test = load(FEAT / "training"), load(FEAT / "testing")
    out, hyper = score_like_objscore2(train, test, vids, device=DEV)
    archived = {"ovel": 0.8616, "opose": 0.5619, "oapp": 0.9424}
    res = {}
    for attr in archived:
        cal = FisherCalibrator().fit(out[f"ref/{attr}"])
        res[attr] = evaluate({v: cal(out[f"{attr}/{v}"]) for v in vids}, gt, 4)
    assert {a: round(r["causal"], 4) for a, r in res.items()} == archived
    if (ARCHIVE / "fuse.json").is_file():                                    # full precision, all three protocols
        full = json.loads((ARCHIVE / "fuse.json").read_text())
        assert all(abs(res[a][k] - full[a][k]) < 1e-9 for a in archived for k in full[a])
    if (ARCHIVE / "hyper.json").is_file():
        assert hyper == json.loads((ARCHIVE / "hyper.json").read_text())
