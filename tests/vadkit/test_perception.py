"""Golden tests of vadkit io / perception / velocity against the released Ped2 light features (tools/objfeat2.py)."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from vadkit.attributes.velocity import Magnitude, OrientationHist, velocity_rows
from vadkit.core.interfaces import Objects
from vadkit.io.frames import VideoSource, frame_paths, read
from vadkit.perception.detector import motion_boxes

REPO = Path(__file__).resolve().parents[2]
FRAMES = Path(os.environ.get("VADKIT_PED2_FRAMES", "data/frames/ped2"))
FEAT2 = Path(os.environ.get("VADKIT_PED2_FEAT2", "data/feat2tmp/feat2/ped2"))
WEIGHTS = Path(os.environ.get("VADKIT_WEIGHTS", Path.home() / ".cache" / "vadkit"))   # ultralytics downloads here
needs_ped2 = pytest.mark.skipif(not (FRAMES.is_dir() and FEAT2.is_dir()), reason="Ped2 frames / released feat2 missing")


def released(split: str, vid: str) -> dict[str, np.ndarray]:
    """Released objfeat2 arrays of one video."""
    return dict(np.load(FEAT2 / split / f"{vid}.npz"))


def objects_at(r: dict[str, np.ndarray], t: int) -> Objects:
    """Released objects of frame t."""
    k = r["f"] == t
    return Objects(box=r["box"][k], cls=r["cls"][k], conf=r["conf"][k], src=r["src"][k])


def per_frame(f: np.ndarray, n: int, w: np.ndarray | None = None) -> np.ndarray:
    """Per-frame row counts (or sums of w)."""
    return np.bincount(f, weights=w, minlength=n)


def load_tool(name: str):
    """Import tools/<name>.py (a script, not a package)."""
    spec = importlib.util.spec_from_file_location(f"_ref_{name}", REPO / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------------ (1) frame order
def test_frame_paths_time_order(tmp_path: Path):
    vd = tmp_path / "training" / "frames" / "Train001"
    vd.mkdir(parents=True)
    for t in range(1011):                                   # hazirla_frames naming: 000.jpg ... 999.jpg, 1000.jpg
        (vd / f"{t:03d}.jpg").touch()
    assert sorted(p.name for p in vd.glob("*.jpg"))[101] == "1000.jpg"        # the text-sort bug
    assert [int(p.stem) for p in frame_paths(tmp_path, "training")["Train001"]] == list(range(1011))
    (vd / "0500.jpg").touch()                               # duplicate stem
    with pytest.raises(ValueError):
        frame_paths(tmp_path, "training")
    (vd / "0500.jpg").unlink(); (vd / "500.jpg").unlink()   # gap
    with pytest.raises(ValueError):
        frame_paths(tmp_path, "training")


def test_video_source_and_read(tmp_path: Path):
    for t in (0, 1, 2, 10, 11, 3, 4, 5, 6, 7, 8, 9):
        cv2.imwrite(str(tmp_path / f"{t}.png"), np.full((8, 12, 3), 10 * t, np.uint8))
    got = [(t, int(im[0, 0, 0])) for t, im in VideoSource(tmp_path, width=640)]
    assert got == [(t, 10 * t) for t in range(12)]
    big = tmp_path / "big" / "0.png"
    big.parent.mkdir()
    cv2.imwrite(str(big), np.zeros((300, 800, 3), np.uint8))
    assert read(big, 640).shape == (240, 640, 3) and read(big, 1000).shape == (300, 800, 3)


# ------------------------------------------------------------------------------------------------ (2) DIS velocity
@needs_ped2
def test_dis_velocity_reproduces_released_vel():
    from vadkit.perception.flow import DISFlow
    vp = load_tool("velpilot")
    flow, c8 = DISFlow(), OrientationHist()
    others = (Magnitude(), OrientationHist(height_norm=True), Magnitude(height_norm=True))
    n_rows = 0
    for split in ("training", "testing"):
        for vid, paths in frame_paths(FRAMES, split).items():
            r = released(split, vid)
            gap = int(r["startup"])
            assert gap == 1 and int(r["n_frames"]) == len(paths)
            k = r["f"] >= gap
            ts = np.unique(r["f"][k])
            ims = {t: read(paths[t]) for t in sorted({u for t in ts for u in (t - gap, t)})}
            fl = flow.flow([ims[t - gap] for t in ts], [ims[t] for t in ts])
            hw = tuple(ims[ts[0]].shape[:2])
            assert hw == tuple(r["frame_hw"])
            vel = np.concatenate([velocity_rows(F, r["box"][r["f"] == t]) for t, F in zip(ts, fl)])
            assert vel.dtype == np.float32 and np.array_equal(vel, r["vel"][k]), (split, vid)
            ref = {"vel": r["vel"][k], "box": r["box"][k], "frame_hw": r["frame_hw"]}
            rows = np.concatenate([c8(objects_at(r, t), F, hw) for t, F in zip(ts, fl)])
            assert np.array_equal(rows, vp.construct(ref, "c8"))
            for ext in others:
                assert np.array_equal(ext.from_vel(ref["vel"], ref["box"], hw), vp.construct(ref, ext.cons)), ext.cons
            n_rows += int(k.sum())
    assert n_rows > 0


@needs_ped2
def test_motion_boxes_reproduce_released():
    """Motion boxes from the released detector boxes and frames t-1, t equal the released motion boxes (exact)."""
    for split, vid in (("training", "Train001"), ("testing", "Test001"), ("testing", "Test004")):
        r, paths = released(split, vid), frame_paths(FRAMES, split)[vid]
        for t in range(int(r["startup"]), int(r["n_frames"])):
            o = objects_at(r, t)
            g = [cv2.cvtColor(read(paths[u]), cv2.COLOR_BGR2GRAY) for u in (t - 1, t)]
            assert np.array_equal(motion_boxes(g[0], g[1], o.box[o.src == 0]), o.box[o.src == 1]), (vid, t)


# ------------------------------------------------------------------------------------------------ (3) detector
@pytest.mark.slow
@needs_ped2
def test_yolo_motion_detector_box_counts():
    from vadkit.perception.detector import YoloMotionDetector
    det = YoloMotionDetector(str(WEIGHTS / "yolo11n.pt"), device="cpu")
    agree = []
    for vid in ("Test001", "Test002"):
        r, paths = released("testing", vid), frame_paths(FRAMES, "testing")[vid]
        det.reset()
        n = [len(det.detect(read(p))) for p in paths]
        assert n[:det.gap] == [0] * det.gap                                     # causal: nothing before t = gap
        agree.append(np.asarray(n)[det.gap:] == per_frame(r["f"], len(paths))[det.gap:])
    frac = np.concatenate(agree).mean()
    print(f"box-count agreement {frac:.4f}")
    assert frac >= 0.98


# ------------------------------------------------------------------------------------------------ (4) pose
@pytest.mark.slow
@needs_ped2
def test_yolo_pose_matches_released():
    from vadkit.perception.pose import YoloPose
    pose = YoloPose(str(WEIGHTS / "yolo11n-pose.pt"), device="cpu")
    r, paths = released("testing", "Test001"), frame_paths(FRAMES, "testing")["Test001"]
    ts = range(int(r["startup"]), len(paths))
    out = [pose.attach(read(paths[t]), objects_at(r, t)) for t in ts]
    ours = np.array([int(o.pose_ok.sum()) for o in out])
    frac = (ours == per_frame(r["f"], len(paths), r["pose_ok"])[int(r["startup"]):]).mean()
    k = r["f"] >= int(r["startup"])
    both = (np.concatenate([o.pose_ok for o in out]) == 1) & (r["pose_ok"][k] == 1)
    err = np.abs(np.concatenate([o.pose for o in out])[both] - r["pose"][k][both])
    print(f"pose_ok count agreement {frac:.4f}, keypoint median abs diff {np.median(err):.2e}")
    assert frac >= 0.95 and np.median(err) < 1e-2
    assert all(not o.pose_ok[o.src == 1].any() for o in out)


# ------------------------------------------------------------------------------------------------ (5) embedding
@pytest.mark.slow
@needs_ped2
def test_open_clip_encoder_matches_released_app():
    pytest.importorskip("open_clip")
    from vadkit.perception.embed import OpenClipEncoder
    enc = OpenClipEncoder(device="cpu")
    r, paths = released("testing", "Test001"), frame_paths(FRAMES, "testing")["Test001"]
    ts = range(int(r["startup"]), len(paths), 5)
    e = np.concatenate([enc.encode(read(paths[t]), r["box"][r["f"] == t]) for t in ts])
    ref = r["app"][np.isin(r["f"], ts)]
    assert e.dtype == np.float16 and e.shape == ref.shape
    cos = (e.astype(np.float64) * ref.astype(np.float64)).sum(1)
    print(f"app cosine: median {np.median(cos):.5f}, >= 0.99 on {(cos >= 0.99).mean():.4f}")
    assert (cos >= 0.99).mean() >= 0.95
    assert enc.encode(read(paths[0]), np.zeros((0, 4), np.float32)).shape == (0, ref.shape[1])


# ------------------------------------------------------------------------------------------------ flow ports, registry
@pytest.mark.slow
@needs_ped2
def test_raft_port_equals_velpilot():
    from vadkit.perception.flow import RAFTFlow
    paths = frame_paths(FRAMES, "testing")["Test001"]
    ims = [read(p) for p in paths[:3]]
    try:
        ours = RAFTFlow(device="cpu", batch=2).flow(ims[:2], ims[1:])
    except OSError as exc:                                  # weights not downloadable
        pytest.skip(f"RAFT weights unavailable: {exc}")
    ref = load_tool("velpilot").Flow("raft", "cpu", batch=2)(ims[:2], ims[1:])
    assert all(a.shape == ims[0].shape[:2] + (2,) and np.array_equal(a, b) for a, b in zip(ours, ref))


def test_neuflow_needs_cuda_and_registry():
    from vadkit.core.registry import _REG
    from vadkit.perception import detector, embed, flow, pose
    from vadkit.attributes import velocity
    assert _REG[("flow", "dis")] is flow.DISFlow and _REG[("flow", "raft")] is flow.RAFTFlow
    assert _REG[("flow", "neuflow")] is flow.NeuFlowFlow and _REG[("detector", "yolo_motion")] is detector.YoloMotionDetector
    assert _REG[("pose", "yolo")] is pose.YoloPose and _REG[("encoder", "open_clip")] is embed.OpenClipEncoder
    assert _REG[("attribute", "orientation_hist")] is velocity.OrientationHist
    assert _REG[("attribute", "magnitude")] is velocity.Magnitude
    if torch.cuda.is_available():
        pytest.skip("CUDA present: the CPU guard is not exercised")
    with pytest.raises(RuntimeError):
        flow.NeuFlowFlow("/nonexistent", device="cpu")
