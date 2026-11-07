"""YOLO-pose keypoints attached to person detector boxes, the matching rule of tools/objfeat2.py."""
from __future__ import annotations

import numpy as np

from vadkit.core.interfaces import Objects, PoseEstimator
from vadkit.core.registry import register
from vadkit.perception.detector import default_device, iou_cover


@register("pose", "yolo")
class YoloPose(PoseEstimator):
    """Each cls-0 box takes the pose box of largest mutual cover if > cover; keypoints relative to centre / height."""

    def __init__(self, weights: str = "yolo11n-pose.pt", conf: float = 0.25, device: str | None = None,
                 cover: float = 0.5):
        from ultralytics import YOLO
        self.model, self.conf, self.device, self.cover = YOLO(weights), conf, default_device(device), cover

    def predict(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Pose model output of one frame: normalised xyxy person boxes and their keypoints (None without any)."""
        pr = self.model.predict(bgr, conf=self.conf, verbose=False, device=self.device)[0]
        pb = pr.boxes.xyxyn.cpu().numpy()
        return pb, (pr.keypoints.xyn.cpu().numpy() if len(pb) and pr.keypoints is not None else None)

    def attach(self, bgr: np.ndarray, objs: Objects, pred: tuple | None = None) -> Objects:
        """Fill objs.pose (N, 34) and objs.pose_ok (N,); boxes without a matched person (and motion boxes) get zeros.
        pred: this frame's predict() output when it was computed ahead (Perception parallel_pose); same result."""
        b = objs.box
        P, ok = np.zeros((len(b), 34), np.float32), np.zeros(len(b), np.uint8)
        persons = np.flatnonzero(objs.cls == 0)
        if len(persons):          # the pose model is stateless, so skipping frames without persons changes nothing
            pb, kp = pred if pred is not None else self.predict(bgr)
            if kp is not None:
                # mutual coverage; row i depends on box i only, so matching after remove_covered equals objfeat2's
                # matching before it followed by the subset
                cov = np.minimum(iou_cover(b, pb), iou_cover(pb, b).T)
                for i in persons:
                    j = int(cov[i].argmax())
                    if cov[i, j] > self.cover:
                        cx, cy = (b[i, 0] + b[i, 2]) / 2, (b[i, 1] + b[i, 3]) / 2
                        P[i] = ((kp[j] - [cx, cy]) / max(b[i, 3] - b[i, 1], 1e-3)).reshape(-1); ok[i] = 1
        objs.pose, objs.pose_ok = P, ok
        return objs
