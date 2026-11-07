"""oapp attribute: object embedding projected on the first principal axes of NORMAL training embeddings
(tools/objscore2.py: mean of all rows, SVD of up to 50000 centred rows drawn with the caller's rng)."""
from __future__ import annotations

import numpy as np

from vadkit.core.interfaces import AttributeExtractor, Objects


class AppearancePCA(AttributeExtractor):
    """(x - mu) @ P with P the top `dim` right singular vectors; reads raw embeddings from objs.attrs[key]."""
    name = "oapp"

    def __init__(self, dim: int = 64, max_rows: int = 50000, key: str = "app"):
        self.dim, self.max_rows, self.key = dim, max_rows, key

    @property
    def raw_key(self) -> str:
        """The feat2 column fit() reads (AttributeExtractor hook): the raw embeddings."""
        return self.key

    def fit(self, rows: list[np.ndarray], rng: np.random.Generator | None = None) -> "AppearancePCA":
        """Fit on the normal training embeddings of every video (in sorted video order to reproduce objscore2)."""
        rng = rng if rng is not None else np.random.default_rng(0)
        A = np.concatenate([np.asarray(r).astype(np.float64) for r in rows])
        self.mu = A.mean(0)
        _, _, Vt = np.linalg.svd(A[rng.permutation(len(A))[:self.max_rows]] - self.mu, full_matrices=False)
        self.P = Vt[:self.dim].T
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """(N, dim) float64 projections of raw (N, D) embeddings."""
        return (np.asarray(X).astype(np.float64) - self.mu) @ self.P

    def __call__(self, objs: Objects, flow: np.ndarray | None = None, frame_hw: tuple[int, int] | None = None) -> np.ndarray:
        """Projected embeddings of one frame's objects."""
        return self.transform(objs.attrs[self.key])

    def record_rows(self, rec: dict) -> tuple[np.ndarray, np.ndarray]:
        """(frame index, projections) of every object of one feat2 npz record (keys f and self.key)."""
        return rec["f"], self.transform(rec[self.key])

    def state(self) -> dict:
        """Arguments, mean and the (dim, D) principal axes."""
        return {"dim": self.dim, "max_rows": self.max_rows, "key": self.key, "mu": self.mu,
                "components": np.ascontiguousarray(self.P.T)}

    @classmethod
    def from_state(cls, state: dict) -> "AppearancePCA":
        """Rebuild a fitted projection from state() (same memory layout, so the matmul is bit-identical)."""
        p = cls(**{k: state[k] for k in ("dim", "max_rows", "key") if k in state})
        p.mu, p.P = np.asarray(state["mu"], np.float64), np.ascontiguousarray(state["components"], np.float64).T
        return p
