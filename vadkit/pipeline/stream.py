"""Streaming object branch: one frame in, one fused causal score out, with a FittedState of pipeline/offline.py."""
from __future__ import annotations

import numpy as np

from vadkit.core.interfaces import FrameResult, Objects
from vadkit.core.registry import build
from vadkit.io.frames import resize_width
from vadkit.pipeline.offline import FittedState, Perception, object_rows, temporal_k
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
                 scene: str | None = None, device: str | None = None):
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
        self.reset()

    def reset(self) -> None:
        """Start of a new video."""
        self.t = 0
        self.smoother.reset()
        if self.perception is not None:
            self.perception.reset()

    def step(self, bgr: np.ndarray) -> FrameResult:
        """Resize to the processing width, perceive (built on first use: detect, pose, flow, embed), step_objects."""
        if self.perception is None:
            self.perception = Perception(self.config, self.gap, self.device)
        bgr = resize_width(bgr, int(self.config.get("width", 640)))
        return self.step_objects(self.t, self.perception(bgr), bgr.shape[:2])

    def step_objects(self, t: int, objs: Objects, frame_hw: tuple[int, int]) -> FrameResult:
        """Frame t from objects with raw attributes: max over objects of -log p (floor without), calibrate, fuse."""
        if self.drop_startup and t < self.gap:
            objs = empty_objects()
        ch = {}
        for a, st in self.attrs.items():
            X = object_rows(st.extractor, objs, tuple(frame_hw))
            s = float((-st.density.logpdf(X)).max()) if len(X) else -np.inf
            ch[a] = float(st.calibrator(s if np.isfinite(s) else st.floor))
        fused = float(self.fusion(ch))
        self.t = t + 1
        return FrameResult(t, self.smoother.push(fused) if t >= self.warmup else None, ch)
