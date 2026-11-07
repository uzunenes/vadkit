"""Sum of calibrated channels (tools/objfuse.py, tools/bootstrap_fused.py 'fisher')."""
from __future__ import annotations

import numpy as np

from vadkit.core.interfaces import Fusion


class SumFusion(Fusion):
    """Sum of the listed channels in list order, or of every given channel in dict order when channels is None."""

    def __init__(self, channels: list[str] | None = None):
        self.channels = list(channels) if channels is not None else None

    def __call__(self, calibrated: dict[str, np.ndarray]) -> np.ndarray:
        """Fused score; a listed channel missing from calibrated raises KeyError."""
        names = self.channels if self.channels is not None else list(calibrated)
        if not names:
            raise ValueError("no channels to fuse")
        return sum(calibrated[n] for n in names)
