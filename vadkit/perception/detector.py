"""YOLO boxes plus motion-contour boxes, the box rule of tools/objfeat2.py, computed causally from frames t-gap and t."""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from vadkit.core.interfaces import Detector, Objects
from vadkit.core.registry import register


def default_device(device: str | None) -> str:
    """'cuda' when available and no device is given, as tools/objfeat2.py."""
    if device is not None:
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def empty_objects() -> Objects:
    """Objects with no rows."""
    return Objects(box=np.zeros((0, 4), np.float32), cls=np.zeros(0, np.int16), conf=np.zeros(0, np.float32),
                   src=np.zeros(0, np.uint8))


def iou_cover(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fraction of each box in a covered by each box in b, (len(a), len(b)) (tools/objfeat2.py)."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    return inter / np.maximum(area[:, None], 1e-9)


def remove_covered(b: np.ndarray, thr: float = 0.6) -> np.ndarray:
    """Indices (sorted) of boxes not covered by more than thr of their area by a larger box (tools/objfeat2.py)."""
    if len(b) < 2:
        return np.arange(len(b))
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    order = area.argsort()
    keep = []
    for i, k in enumerate(order):
        larger = order[i + 1:]
        if not len(larger) or iou_cover(b[k:k + 1], b[larger]).max() <= thr:
            keep.append(k)
    return np.array(sorted(keep), int)


def motion_boxes(prev: np.ndarray, cur: np.ndarray, det: np.ndarray, min_frac: float = 1e-3,
                 thr: float = 18) -> np.ndarray:
    """Normalised boxes of moving blobs (|blur difference| > thr grey levels) outside det (tools/objfeat2.py)."""
    H, W = cur.shape
    d = cv2.absdiff(cv2.GaussianBlur(prev, (5, 5), 0), cv2.GaussianBlur(cur, (5, 5), 0))
    m = (d > thr).astype(np.uint8) * 255
    for x1, y1, x2, y2 in (det * [W, H, W, H]).astype(int):
        m[max(y1 - 2, 0):y2 + 3, max(x1 - 2, 0):x2 + 3] = 0
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        x, y, w, h = cv2.boundingRect(c)
        if (w + 1) * (h + 1) >= min_frac * H * W and w / max(h, 1) < 10 and h / max(w, 1) < 10:
            out.append([max(x - 2, 0) / W, max(y - 2, 0) / H, min(x + w + 2, W) / W, min(y + h + 2, H) / H])
    return np.asarray(out, np.float32).reshape(-1, 4)


@register("detector", "yolo_motion")
class YoloMotionDetector(Detector):
    """Stateful: detect(frame t) uses frames t-gap and t only; returns no objects for t < gap. reset() per video."""

    def __init__(self, weights: str = "yolo11n.pt", conf: float = 0.25, gap: int = 1, device: str | None = None,
                 motion_thr: float = 18, min_frac: float = 1e-3, cover_thr: float = 0.6):
        from ultralytics import YOLO
        self.model, self.conf, self.gap, self.device = YOLO(weights), conf, gap, default_device(device)
        self.motion_thr, self.min_frac, self.cover_thr = motion_thr, min_frac, cover_thr
        self.reset()

    def reset(self) -> None:
        """Forget past frames (start of a new video)."""
        self._grey: deque[np.ndarray] = deque(maxlen=self.gap + 1)

    def detect(self, bgr: np.ndarray) -> Objects:
        """Detector boxes (after remove_covered) followed by motion boxes (src 1, cls -1, conf 0)."""
        self._grey.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        if len(self._grey) <= self.gap:
            return empty_objects()
        r = self.model.predict(bgr, conf=self.conf, verbose=False, device=self.device)[0]
        b, c, cf = r.boxes.xyxyn.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()
        k = remove_covered(b, self.cover_thr); b, c, cf = b[k], c[k], cf[k]
        mb = motion_boxes(self._grey[0], self._grey[-1], b, self.min_frac, self.motion_thr)
        return Objects(box=(np.concatenate([b, mb]) if len(mb) else b).astype(np.float32),
                       cls=np.r_[c, -np.ones(len(mb))].astype(np.int16),
                       conf=np.r_[cf, np.zeros(len(mb))].astype(np.float32),
                       src=np.r_[np.zeros(len(b)), np.ones(len(mb))].astype(np.uint8))
