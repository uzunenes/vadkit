"""opose attribute: COCO-17 keypoints of the matched persons only (tools/objscore2.py attr_rows 'opose')."""
from __future__ import annotations

import numpy as np

from vadkit.core.interfaces import AttributeExtractor, Objects


class PoseKeypoints(AttributeExtractor):
    """34-d keypoints relative to the box centre / box height, rows of objects with pose_ok > 0 only."""
    name = "opose"

    def __call__(self, objs: Objects, flow: np.ndarray | None = None, frame_hw: tuple[int, int] | None = None) -> np.ndarray:
        """(M, 34) float64 rows of the matched persons of one frame (M <= len(objs))."""
        if objs.pose is None:
            return np.zeros((0, 34))
        return objs.pose[objs.pose_ok > 0].astype(np.float64)

    def record_rows(self, rec: dict) -> tuple[np.ndarray, np.ndarray]:
        """(frame index, rows) of the matched persons of one feat2 npz record (keys f, pose, pose_ok)."""
        k = rec["pose_ok"] > 0
        return rec["f"][k], rec["pose"][k].astype(np.float64)
