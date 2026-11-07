"""Registers every built-in back-end, so registry.build / available know them (heavy models still load on demand)."""
from vadkit.attributes.appearance import AppearancePCA
from vadkit.attributes.pose import PoseKeypoints
from vadkit.attributes import velocity  # noqa: F401  (orientation_hist, magnitude)
from vadkit.calibration.fisher import FisherCalibrator, ZCalibrator
from vadkit.calibration.gpd import GPDTailCalibrator
from vadkit.calibration.hier import HBGroupCalibrator
from vadkit.core.registry import register
from vadkit.density.niw import NIWDensity
from vadkit.fusion.sum import SumFusion
from vadkit.perception import detector, embed, flow, pose  # noqa: F401  (yolo_motion, yolo, open_clip, dis ...)
from vadkit.temporal.median import CausalMedian

for kind, name, cls in (("attribute", "pose_keypoints", PoseKeypoints), ("attribute", "appearance_pca", AppearancePCA),
                        ("density", "niw", NIWDensity), ("calibrator", "fisher", FisherCalibrator),
                        ("calibrator", "z", ZCalibrator), ("calibrator", "fisher_gpd", GPDTailCalibrator),
                        ("group_calibrator", "fisher_hb", HBGroupCalibrator), ("fusion", "sum", SumFusion),
                        ("temporal", "causal_median", CausalMedian)):
    register(kind, name)(cls)
