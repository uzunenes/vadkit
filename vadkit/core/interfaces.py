"""vadkit interfaces. Every stage is a small class with a fixed contract, so a YAML config can swap back-ends
(desktop PyTorch, ONNX Runtime, TensorRT, Hailo) without touching the rest. All stages are CAUSAL: a call for frame t
may use frames <= t only. Images are BGR uint8 (H, W, 3) at the processing width; boxes are normalised xyxy in [0, 1].
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------------------------- data
@dataclass
class Objects:
    """Objects of one frame. src: 0 detector box, 1 motion-contour box. pose: (N, 34) COCO-17 xy relative to the box
    centre / box height, pose_ok: (N,) 1 when a person keypoint set was matched."""
    box: np.ndarray                                     # (N, 4) float32, normalised xyxy
    cls: np.ndarray                                     # (N,) int16, -1 for motion boxes
    conf: np.ndarray                                    # (N,) float32, 0 for motion boxes
    src: np.ndarray                                     # (N,) uint8
    pose: np.ndarray | None = None                      # (N, 34) float32
    pose_ok: np.ndarray | None = None                   # (N,) uint8
    attrs: dict[str, np.ndarray] = field(default_factory=dict)   # name -> (N, D) per-object attribute rows

    def __len__(self) -> int:
        return len(self.box)


@dataclass
class FrameResult:
    t: int                                              # frame index in the video (time order)
    score: float | None                                 # fused score after temporal smoothing; None while warming up
    channels: dict[str, float] = field(default_factory=dict)     # calibrated (-log tail) score per channel, unsmoothed


# ---------------------------------------------------------------------------------------------- perception
class Detector(ABC):
    @abstractmethod
    def detect(self, bgr: np.ndarray) -> Objects: ...


class PoseEstimator(ABC):
    @abstractmethod
    def attach(self, bgr: np.ndarray, objs: Objects) -> Objects:
        """Fill objs.pose / objs.pose_ok for person boxes (matching rule of tools/objfeat2.py)."""


class FlowEstimator(ABC):
    @abstractmethod
    def flow(self, prev_bgr: list[np.ndarray], cur_bgr: list[np.ndarray]) -> list[np.ndarray]:
        """Batch of (H, W, 2) float32 flows prev -> cur in pixels of the processing frame."""


class AppearanceEncoder(ABC):
    @abstractmethod
    def encode(self, bgr: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """(N, D) L2-normalised embeddings of the box crops."""


# ---------------------------------------------------------------------------------------------- attributes
class AttributeExtractor(ABC):
    """Turns per-object raw quantities into attribute rows; may have a normal-data fitted part (e.g. PCA).

    Fitted part, explicit: an extractor that sets raw_key (the raw feat2 column it is fitted on) gets
    fit([that column of every normal training video, sorted names], rng) with the fit's generator, shared by the
    attributes in config order (tools/objscore2.py), and must override state() / from_state() so a FittedState can be
    saved and loaded. With raw_key None (default) nothing is fitted and nothing is saved."""
    name: str
    raw_key: str | None = None

    def fit(self, rows: list[np.ndarray], rng: np.random.Generator | None = None) -> "AttributeExtractor":
        return self

    def state(self) -> dict:
        """Fitted part as scalars and arrays (only called when raw_key is set)."""
        raise NotImplementedError(f"{type(self).__name__} sets raw_key, so it needs state() and from_state()")

    @classmethod
    def from_state(cls, state: dict) -> "AttributeExtractor":
        """Inverse of state()."""
        raise NotImplementedError(f"{cls.__name__} sets raw_key, so it needs state() and from_state()")

    @abstractmethod
    def __call__(self, objs: Objects, flow: np.ndarray | None, frame_hw: tuple[int, int]) -> np.ndarray: ...


# ---------------------------------------------------------------------------------------------- statistics
class DensityModel(ABC):
    """Fitted on NORMAL training rows only. Hyper-parameters are chosen on normal data (held-out likelihood)."""

    @abstractmethod
    def fit(self, rows_by_video: dict[str, np.ndarray]) -> "DensityModel": ...

    @abstractmethod
    def logpdf(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def state(self) -> dict: ...

    @classmethod
    @abstractmethod
    def from_state(cls, state: dict) -> "DensityModel": ...


class Calibrator(ABC):
    """Maps a raw channel score to -log P(normal score >= s), fitted on cross-fitted normal reference scores."""

    @abstractmethod
    def fit(self, ref: np.ndarray) -> "Calibrator": ...

    @abstractmethod
    def __call__(self, s: np.ndarray) -> np.ndarray: ...


class Fusion(ABC):
    @abstractmethod
    def __call__(self, calibrated: dict[str, np.ndarray]) -> np.ndarray: ...


class TemporalSmoother(ABC):
    """Streaming: push one frame score, get the smoothed score (None while the window is not ready)."""

    @abstractmethod
    def push(self, s: float) -> float | None: ...

    @abstractmethod
    def reset(self) -> None: ...
