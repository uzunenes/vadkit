"""Box velocity attributes from optical flow: tools/objfeat2.velocity (8 bins) and the tools/velpilot.construct variants."""
from __future__ import annotations

import numpy as np

from vadkit.core.interfaces import AttributeExtractor, Objects
from vadkit.core.registry import register


def orientation_hist(flow: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(8,) float32 mean L1 flow magnitude per orientation bin over the box pixels (tools/objfeat2.velocity)."""
    H, W = flow.shape[:2]
    x1, y1, x2, y2 = (b * [W, H, W, H]).round().astype(int)
    f = flow[max(y1, 0):max(y2, y1 + 1), max(x1, 0):max(x2, x1 + 1)].reshape(-1, 2)
    if not len(f):
        return np.zeros(8, np.float32)
    mag = np.abs(f).sum(1)
    ang = (np.arctan2(f[:, 1], f[:, 0]) % (2 * np.pi)) / (2 * np.pi) * 8
    return (np.bincount(np.minimum(ang.astype(int), 7), weights=mag, minlength=8) / len(f)).astype(np.float32)


def velocity_rows(flow: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """(N, 8) float32 orientation histograms of the boxes: the stored 'vel' array of the feat2 npz."""
    return np.stack([orientation_hist(flow, b) for b in boxes]) if len(boxes) else np.zeros((0, 8), np.float32)


def construct(vel: np.ndarray, box: np.ndarray, frame_h: int, cons: str) -> np.ndarray:
    """float64 rows c8 / c1 / c8h / c1h from stored vel rows (tools/velpilot.construct)."""
    v = vel.astype(np.float64)
    if cons == "c8":
        return v
    h = np.maximum((box[:, 3] - box[:, 1]).astype(np.float64) * float(frame_h), 1.0)[:, None]
    return {"c1": v.sum(1, keepdims=True), "c8h": v / h, "c1h": v.sum(1, keepdims=True) / h}[cons]


class _Velocity(AttributeExtractor):
    """Shared part: flow -> vel rows -> construct(cons); cons = base, or base + 'h' (divided by box height in px)."""
    base = "c8"

    def __init__(self, height_norm: bool = False):
        self.cons = self.base + ("h" if height_norm else "")
        self.name = f"vel_{self.cons}"

    def from_vel(self, vel: np.ndarray, box: np.ndarray, frame_hw: tuple[int, int]) -> np.ndarray:
        """Rows from stored vel / box arrays (released or cached feat2 npz)."""
        return construct(vel, box, frame_hw[0], self.cons)

    def __call__(self, objs: Objects, flow: np.ndarray | None, frame_hw: tuple[int, int]) -> np.ndarray:
        """Rows of the objects of one frame from its flow t-gap -> t."""
        if flow is None:
            raise ValueError(f"{self.name} needs optical flow")
        return self.from_vel(velocity_rows(flow, objs.box), objs.box, frame_hw)


@register("attribute", "orientation_hist")
class OrientationHist(_Velocity):
    """8-bin velocity c8 (or c8h with height_norm), float64 rows."""
    base = "c8"


@register("attribute", "magnitude")
class Magnitude(_Velocity):
    """Mean L1 flow magnitude c1 = sum of the 8 bins (or c1h with height_norm), float64 rows."""
    base = "c1"
