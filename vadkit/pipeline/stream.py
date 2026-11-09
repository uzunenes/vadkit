"""Streaming detector: one frame in, one fused causal score out, with a FittedState of pipeline/offline.py and,
optionally, the native grid branch (vadkit/grid/stream.py GridBranch: channels I, Fs, Fd on the same frame)."""
from __future__ import annotations

import numpy as np

from vadkit.core.interfaces import FrameResult, Objects
from vadkit.core.registry import build
from vadkit.io.frames import resize_width
from vadkit.pipeline.offline import FittedState, Perception, channel_order, object_rows, temporal_k
from vadkit.perception.detector import empty_objects
from vadkit.temporal.median import CausalMedian


def objects_at(rec: dict, t: int) -> Objects:
    """Objects of frame t of a feat2 npz record with their raw attributes, for replay through step_objects."""
    k = rec["f"] == t
    pose, ok = (rec["pose"][k], rec["pose_ok"][k]) if "pose" in rec else (None, None)
    return Objects(box=rec["box"][k], cls=rec["cls"][k], conf=rec["conf"][k], src=rec["src"][k], pose=pose, pose_ok=ok,
                   attrs={n: rec[n][k] for n in ("vel", "app") if n in rec})


class StreamingVAD:
    """Causal per-frame scoring of state.channels at the state's gap; frames t < warmup: score None, not smoothed."""

    def __init__(self, state: FittedState, config: dict | None = None, gap: int | None = None,
                 scene: str | None = None, device: str | None = None, grid=None):
        self.state, self.config, self.device = state, config or state.config, device
        self.gap = state.gap if gap is None else gap
        if self.gap is None:
            raise ValueError("the state records no frame gap; pass gap=")
        if state.gap is not None and self.gap != state.gap:
            raise ValueError(f"gap {gap} differs from the state's {state.gap} (the startup of its training features)")
        if state.scenes and scene is None:
            raise ValueError("a per-scene state needs the scene of the stream")
        group = state.groups[scene if state.scenes else "all"]
        self.attrs = {a: group[a] for a in state.channels}
        self.fusion = build("fusion", self.config.get("fusion", "sum"))
        self.smoother = CausalMedian(temporal_k(self.config.get("temporal", "causal_median")))   # as evaluate()
        self.warmup = int(self.config.get("warmup", 4))
        self.drop_startup = bool(state.config.get("drop_startup", False))
        self.perception: Perception | None = None
        # grid branch (optional): raw I, Fs, Fd of the original frame, calibrated against its own cross-fitted normal
        # references with the config's grid_calibrator (default: calibrator), as PrecomputedChannels does offline
        self.grid, self.grid_cal, self.grid_pool = grid, {}, None
        if grid is not None:
            spec = self.config.get("grid_calibrator") or self.config.get("calibrator", "fisher")
            self.grid_cal = {c: build("calibrator", spec).fit(r) for c, r in grid.state.ref.items()}
            if self.warmup < grid.cfg.gap:
                raise ValueError(f"warmup {self.warmup} < grid gap {grid.cfg.gap}: Fs, Fd start at t = gap")
            if self.config.get("parallel_grid", False):
                from concurrent.futures import ThreadPoolExecutor
                self.grid_pool = ThreadPoolExecutor(1, thread_name_prefix="vadkit-grid")
        self.reset()

    def reset(self) -> None:
        """Start of a new video."""
        self.t = 0
        self.smoother.reset()
        if self.perception is not None:
            self.perception.reset()
        if self.grid is not None:
            self.grid.reset()

    def step(self, bgr: np.ndarray) -> FrameResult:
        """Resize to the processing width, perceive (built on first use: detect, pose, flow, embed), step_objects."""
        if self.perception is None and (self.attrs or self.grid is None):     # grid-only: no perception back-ends
            self.perception = Perception(self.config, self.gap, self.device)
        fut = raw = None
        if self.grid is not None:          # the grid works on the original frame (its own grey size), not the 640 one
            if self.grid_pool is not None:
                fut = self.grid_pool.submit(self.grid.step, bgr)
            else:
                raw = self.grid.step(bgr)
        bgr = resize_width(bgr, int(self.config.get("width", 640)))
        objs = self.perception(bgr) if self.perception is not None else empty_objects()
        if fut is not None:
            raw = fut.result()
        hw = bgr.shape[:2]
        return self.step_objects(self.t, objs, hw) if raw is None else self.step_objects(self.t, objs, hw, raw)

    def step_objects(self, t: int, objs: Objects, frame_hw: tuple[int, int],
                     grid_raw: dict[str, float] | None = None) -> FrameResult:
        """Frame t from objects with raw attributes: max over objects of -log p (floor without), calibrate, fuse;
        grid_raw: raw grid scores of the frame (GridBranch.step), calibrated and fused in the order I, Fs, Fd, objects
        (channel_order, as offline). Grid channels missing at t < gap are left out of that frame's sum (t < warmup)."""
        if self.drop_startup and t < self.gap:
            objs = empty_objects()
        ch = {}
        for a, st in self.attrs.items():
            X = object_rows(st.extractor, objs, tuple(frame_hw))
            s = float((-st.density.logpdf(X)).max()) if len(X) else -np.inf
            ch[a] = float(st.calibrator(s if np.isfinite(s) else st.floor))
        if grid_raw:
            ch = channel_order({c: float(self.grid_cal[c](np.asarray([v]))[0]) for c, v in grid_raw.items()}, ch)
        fused = float(self.fusion(ch))
        self.t = t + 1
        return FrameResult(t, self.smoother.push(fused) if t >= self.warmup else None, ch)
