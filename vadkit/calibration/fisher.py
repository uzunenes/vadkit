"""Channel calibration against cross-fitted normal reference scores (tools/fuse_runs.normaliser)."""
from __future__ import annotations

import numpy as np

from vadkit.core.interfaces import Calibrator


class FisherCalibrator(Calibrator):
    """-log P(R >= s): smoothed empirical tail below the 90th percentile u of the reference, exponential tail
    (POT, GPD with xi = 0, mean excess beta) beyond u; calibrated channels add up as in Fisher's method."""

    def fit(self, ref: np.ndarray) -> "FisherCalibrator":
        """Store the sorted reference, its 90th percentile and the mean excess over it."""
        r = np.sort(ref)
        self.r, self.n = r, len(r)
        self.u = np.quantile(r, 0.9)
        self.beta = max(float((r[r > self.u] - self.u).mean()) if (r > self.u).any() else 1e-6, 1e-6)
        return self

    def __call__(self, s: np.ndarray) -> np.ndarray:
        """Calibrated score of raw scores s (array or scalar)."""
        p_emp = (self.n - np.searchsorted(self.r, s, side="left") + 1) / (self.n + 1)
        nlog_tail = -np.log(0.1) + (s - self.u) / self.beta
        return np.where(s > self.u, nlog_tail, -np.log(p_emp))


class ZCalibrator(Calibrator):
    """(s - mean r) / std r (fuse_runs.normaliser 'z')."""

    def fit(self, ref: np.ndarray) -> "ZCalibrator":
        """Store the reference mean and standard deviation."""
        self.m, self.sd = ref.mean(), ref.std() + 1e-12
        return self

    def __call__(self, s: np.ndarray) -> np.ndarray:
        """Standardised score."""
        return (s - self.m) / self.sd
