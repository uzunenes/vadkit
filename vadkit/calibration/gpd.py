"""Fisher calibrator with a generalised Pareto tail (HB-POT, stage 1 of
docs/tez_notlari/hiyerarsik_kalibrasyon_on_kayit_2026-11-05.md): the body below u = Q0.9 of the reference is the
smoothed empirical tail of FisherCalibrator; beyond u, z = -log 0.1 + (1/xi) log1p(xi y / sigma), y = s - u.
xi = 0 is the exponential tail of FisherCalibrator; with sigma = beta (mean excess) it is that calibrator bit for bit."""
from __future__ import annotations

import numpy as np

from vadkit.calibration.fisher import FisherCalibrator


class GPDTailCalibrator(FisherCalibrator):
    """FisherCalibrator body; GPD tail of shape xi >= 0 and scale sigma beyond u (sigma None: the mean excess beta).
    xi < 0 is refused: its finite end point would make z infinite beyond it (the arms truncate xi at 0)."""

    def __init__(self, xi: float = 0.0, sigma: float | None = None):
        if not xi >= 0:
            raise ValueError(f"xi must be >= 0, got {xi}")
        if sigma is not None and not sigma > 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self.xi, self.sigma_arg = float(xi), sigma

    def fit(self, ref: np.ndarray) -> "GPDTailCalibrator":
        """FisherCalibrator.fit (sorted reference, u, beta), then the tail scale."""
        super().fit(ref)
        self.sigma = self.beta if self.sigma_arg is None else float(self.sigma_arg)
        return self

    def __call__(self, s: np.ndarray) -> np.ndarray:
        """Calibrated score of raw scores s (array or scalar); the same operations as FisherCalibrator when xi = 0."""
        p_emp = (self.n - np.searchsorted(self.r, s, side="left") + 1) / (self.n + 1)
        if self.xi == 0.0:
            nlog_tail = -np.log(0.1) + (s - self.u) / self.sigma
        else:
            nlog_tail = -np.log(0.1) + np.log1p(self.xi * np.maximum(s - self.u, 0.0) / self.sigma) / self.xi
        return np.where(s > self.u, nlog_tail, -np.log(p_emp))

    def tail_logpdf(self, y: np.ndarray) -> np.ndarray:
        """log g(y) of the GPD tail at excesses y > 0 (the gate's tail log-score adds log 0.1)."""
        y = np.asarray(y, dtype=np.float64)
        if self.xi == 0.0:
            return -np.log(self.sigma) - y / self.sigma
        return -np.log(self.sigma) - (1.0 / self.xi + 1.0) * np.log1p(self.xi * y / self.sigma)
