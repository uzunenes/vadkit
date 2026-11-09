"""Native streaming grid branch (vadkit/grid/stream.py) on a synthetic scene with a randomly initialised autoencoder:
streaming == batch (evaluate.py functions) exactly, state round trip, and the fusion wiring of StreamingVAD."""
import json

import numpy as np
import pytest
import torch

from aean.models import build
from vadkit.calibration.fisher import FisherCalibrator
from vadkit.grid.stream import GridBranch, GridConfig, GridState, batch_scores, fit_grid
from vadkit.pipeline.offline import FittedState
from vadkit.pipeline.stream import StreamingVAD
from vadkit.temporal.median import causal_median

H, W = 48, 64


def video(n: int, seed: int, x0: int = 4, step: int = 2) -> np.ndarray:
    """Noise background with a bright square moving to the right."""
    rng = np.random.default_rng(seed)
    v = rng.integers(60, 90, (n, H, W)).astype(np.uint8)
    for t in range(n):
        x = (x0 + step * t) % (W - 10)
        v[t, 20:30, x:x + 10] = 200
    return v


@pytest.fixture(scope="module")
def grid(tmp_path_factory):
    torch.manual_seed(0)
    run = tmp_path_factory.mktemp("run")
    ae, disc, _ = build("2d", 16)
    torch.save({"ae": ae.state_dict(), "disc": disc.state_dict()}, run / "model.pt")
    (run / "args.json").write_text(json.dumps({"variant": "2d", "T": 16, "patch": 16}))
    train = {f"Train{i:03d}": video(40, i) for i in range(4)}
    cfg = GridConfig(gap=2, pool=16)
    state = fit_grid(train, run, cfg, "cpu")
    return run, state


def test_stream_equals_batch(grid):
    run, state = grid
    v = video(30, 99, step=6)
    b = batch_scores(state, v, run)
    br = GridBranch(state, run)
    rows = [br.step_gray(g) for g in v]
    assert all(set(r) == {"I"} for r in rows[:2]) and all(set(r) == {"I", "Fs", "Fd"} for r in rows[2:])
    for c in ("I", "Fs", "Fd"):
        s = np.array([r.get(c, np.nan) for r in rows])
        # I: identical code path, exact; Fs, Fd: the batch scores 256 frames per einsum, the stream one, so float32
        # sums may round differently (observed <= 2e-6)
        if c == "I":
            np.testing.assert_array_equal(s, b[c])
        else:
            np.testing.assert_allclose(s[2:], b[c][2:], rtol=1e-6, atol=1e-6)
    br.reset()
    assert br.step_gray(v[0]) == rows[0]


def test_state_round_trip(grid, tmp_path):
    run, state = grid
    state.save(tmp_path / "g")
    st2 = GridState.load(tmp_path / "g")
    assert st2.config == state.config and (st2.H, st2.W) == (H, W)
    v = video(12, 7)
    a, b = batch_scores(state, v, run), batch_scores(st2, v, run)
    for c in a:
        np.testing.assert_array_equal(a[c], b[c])
    assert len(state.ref["I"]) == 4 * len(range(15, 40, 15)) and len(state.ref["Fs"]) == 4 * len(range(0, 40, 3))


def test_wrong_weights_rejected(grid, tmp_path):
    run, state = grid
    other = tmp_path / "other"
    other.mkdir()
    ae, disc, _ = build("2d", 16)
    torch.save({"ae": ae.state_dict(), "disc": disc.state_dict()}, other / "model.pt")
    (other / "args.json").write_text((run / "args.json").read_text())
    with pytest.raises(ValueError, match="differs"):
        GridBranch(state, other)


def test_streaming_vad_fuses_grid(grid):
    """Grid-only detector: fused = sum of the Fisher-calibrated I, Fs, Fd; causal median from the warm-up on."""
    run, state = grid
    cfg = {"attributes": [], "fusion": "sum", "temporal": "causal_median", "warmup": 4, "calibrator": "fisher"}
    sv = StreamingVAD(FittedState(cfg, {"all": {}}, False, 2), cfg, grid=GridBranch(state, run))
    v = video(20, 5, step=5)
    out = [sv.step(np.repeat(g[..., None], 3, -1)) for g in v]
    br = GridBranch(state, run)                     # the same frames, grey (the colour frames are grey copies)
    st = [br.step_gray(g) for g in v]
    raw = {c: np.array([r.get(c, np.nan) for r in st]) for c in ("I", "Fs", "Fd")}
    cal = {c: FisherCalibrator().fit(state.ref[c])(raw[c]) for c in ("I", "Fs", "Fd")}
    fused = cal["I"][4:] + cal["Fs"][4:] + cal["Fd"][4:]
    assert all(o.score is None for o in out[:4])
    np.testing.assert_allclose([o.score for o in out[4:]], causal_median(fused, 9), rtol=0, atol=1e-12)
    assert list(out[5].channels) == ["I", "Fs", "Fd"]
    with pytest.raises(ValueError, match="warmup"):
        StreamingVAD(FittedState(cfg, {"all": {}}, False, 2), {**cfg, "warmup": 1}, grid=GridBranch(state, run))
