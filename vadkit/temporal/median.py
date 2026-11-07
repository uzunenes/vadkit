"""Causal median smoothing of frame scores (tools/objfuse.causal_median), offline and streaming."""
from __future__ import annotations

from collections import deque

import numpy as np

from vadkit.core.interfaces import TemporalSmoother


def causal_median(s: np.ndarray, k: int = 9) -> np.ndarray:
    """Median of s[max(0, i - k + 1): i + 1] for every i (frame i uses frames <= i only)."""
    return np.array([np.median(s[max(0, i - k + 1): i + 1]) for i in range(len(s))])


class CausalMedian(TemporalSmoother):
    """Streaming causal_median: push(s) returns the median of the last min(n, k) pushed scores."""

    def __init__(self, k: int = 9):
        self.k = k
        self.buf: deque[float] = deque(maxlen=k)

    def push(self, s: float) -> float:
        """Add one frame score and return the smoothed score of that frame."""
        self.buf.append(s)
        return float(np.median(np.array(self.buf)))

    def reset(self) -> None:
        """Forget the window (start of a new video)."""
        self.buf.clear()
