"""NIW-mixture attribute densities fitted on normal training videos (tools/objscore2.py, tools/velpilot.py score).

Hyper-parameters (K, kappa0, nu0) are chosen by held-out predictive likelihood on a 2-fold split of the sorted
training video names (aean.niw.select_mixture, eval halves subsampled to 20000 rows with the caller's rng). The normal
reference is cross-fitted by video; the frame score is the max over the frame's objects of -log p, and frames without
objects take the lowest finite reference score (the floor).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from aean.niw import NIWCells, NIWMixture, select_mixture
from vadkit.core.interfaces import DensityModel


@dataclass
class VideoRows:
    """Attribute rows of one video: frame index per row, (N, D) rows, frame count, first frame with features."""
    f: np.ndarray
    X: np.ndarray
    n_frames: int
    startup: int = 0


@dataclass
class AttributeFit:
    """fit_attribute result: floor-filled test frame scores, cross-fitted reference, hyper-parameters, full model."""
    scores: dict[str, np.ndarray]
    ref: np.ndarray
    floor: float
    hyper: dict
    density: "NIWDensity"


def _tensors(rows_by_video: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """float64 tensors in sorted video order."""
    return {v: torch.from_numpy(np.asarray(rows_by_video[v], np.float64)) for v in sorted(rows_by_video)}


def select_hyper(rows_by_video: dict[str, np.ndarray], rng: np.random.Generator, device: str = "cpu",
                 max_eval: int = 20000, min_rows: int = 10) -> dict | None:
    """{K, kappa0, nu0, dim} by 2-fold held-out likelihood (folds names[0::2], names[1::2] of the sorted names);
    None when a fold has fewer than min_rows fitting rows (the rng is consumed either way, as in objscore2)."""
    X = _tensors(rows_by_video)
    names = list(X)
    folds = (names[0::2], names[1::2])
    fit_sets, eval_sets = [], []
    for fit_n, ev_n in (folds, folds[::-1]):
        Xe = torch.cat([X[v] for v in ev_n])
        fit_sets.append(torch.cat([X[v] for v in fit_n]))
        eval_sets.append(Xe[rng.permutation(len(Xe))[:max_eval]])
    if min(len(x) for x in fit_sets) < min_rows:
        return None
    (K, k0, e), _ = select_mixture(fit_sets, eval_sets, device=device)
    C = fit_sets[0].shape[1]
    return {"K": K, "kappa0": k0, "nu0": C + 1 + e, "dim": C}


class NIWDensity(DensityModel):
    """aean.niw.NIWMixture with conjugate NIW priors; hyper-parameters given (K, kappa0, nu0 together), or chosen in
    fit() by select_hyper (held-out rows capped at max_eval, folds of at least min_rows rows)."""

    def __init__(self, K: int | None = None, kappa0: float | None = None, nu0: float | None = None,
                 rng: np.random.Generator | None = None, device: str = "cpu", max_eval: int = 20000,
                 min_rows: int = 10):
        if (K is None) != (kappa0 is None) or (K is None) != (nu0 is None):
            raise ValueError("NIWDensity: give K, kappa0 and nu0 together (fixed), or none (selected on normal data)")
        self.K, self.kappa0, self.nu0, self.rng, self.device = K, kappa0, nu0, rng, device
        self.max_eval, self.min_rows = int(max_eval), int(min_rows)

    def fit(self, rows_by_video: dict[str, np.ndarray]) -> "NIWDensity":
        """Select the hyper-parameters if unset, then fit on all rows concatenated in sorted video order."""
        if self.K is None:
            h = select_hyper(rows_by_video, self.rng if self.rng is not None else np.random.default_rng(0), self.device,
                             self.max_eval, self.min_rows)
            if h is None:
                raise ValueError("too few training rows to select the NIW mixture")
            self.K, self.kappa0, self.nu0 = h["K"], h["kappa0"], h["nu0"]
        X = torch.cat(list(_tensors(rows_by_video).values()))
        self.model = NIWMixture(self.K, self.kappa0, self.nu0, device=self.device).fit(X)
        return self

    def logpdf(self, X: np.ndarray) -> np.ndarray:
        """Log posterior predictive density of every row, (N,) float64."""
        return self.model.logpdf(torch.from_numpy(np.asarray(X, np.float64))).cpu().numpy()

    def frame_scores(self, f: np.ndarray, X: np.ndarray, n_frames: int) -> np.ndarray:
        """Per frame max over its rows of -log p; -inf for frames without rows."""
        s = np.full(int(n_frames), -np.inf)
        if len(X):
            np.maximum.at(s, f, -self.logpdf(X))
        return s

    def state(self) -> dict:
        """Hyper-parameters, k-means centres and per-component sufficient statistics (refitting from them is exact)."""
        m, c = self.model, self.model.cells
        return {"K": self.K, "kappa0": self.kappa0, "nu0": self.nu0, "alpha0": m.alpha0,
                "centres": m.centres.cpu().numpy(), "n": c.n.cpu().numpy(), "s1": c.s1.cpu().numpy(),
                "s2": c.s2.cpu().numpy()}

    @classmethod
    def from_state(cls, state: dict, device: str = "cpu") -> "NIWDensity":
        """Rebuild the fitted density of state()."""
        d = cls(int(state["K"]), float(state["kappa0"]), float(state["nu0"]), device=device)
        m = NIWMixture(d.K, d.kappa0, d.nu0, float(state["alpha0"]), device)
        m.centres = torch.as_tensor(np.asarray(state["centres"]), dtype=torch.float64, device=device)
        cells = NIWCells(m.centres.shape[1], d.K, device)
        cells.n, cells.s1, cells.s2 = (torch.as_tensor(np.asarray(state[k]), dtype=torch.float64, device=device)
                                       for k in ("n", "s1", "s2"))
        m.cells = cells.finalize(d.kappa0, d.nu0)
        m.logw = torch.log((cells.n + m.alpha0 / m.K) / (cells.n.sum() + m.alpha0))
        d.model = m
        return d


def _rows(r: VideoRows, sub: bool, train_every: int, drop_startup: bool) -> tuple[np.ndarray, np.ndarray]:
    """Rows kept for scoring: f >= startup with drop_startup; also f % train_every == 0 when sub (training)."""
    k = (r.f >= r.startup) if drop_startup else np.ones(len(r.f), bool)
    if sub:
        k &= (r.f % train_every == 0)
    return r.f[k], r.X[k]


def fit_attribute(train: dict[str, VideoRows], test: dict[str, VideoRows], rng: np.random.Generator,
                  device: str = "cpu", train_every: int = 1, drop_startup: bool = False,
                  density: dict | None = None) -> AttributeFit | None:
    """tools/objscore2.py for one attribute (rng shared by the caller across attributes), or tools/velpilot.py
    fit_group with train_every / drop_startup; None when too few training rows (objscore2 skips the attribute).
    density: NIWDensity arguments of the config (fixed K, kappa0, nu0, or the selection's max_eval / min_rows)."""
    proto = NIWDensity(**(density or {}), device=device)
    names = sorted(train)
    X = {v: _rows(train[v], True, train_every, drop_startup)[1] for v in names}
    folds = (names[0::2], names[1::2])
    if proto.K is None:
        hyper = select_hyper(X, rng, device, proto.max_eval, proto.min_rows)
        if hyper is None:
            return None
    elif min(sum(len(X[v]) for v in f) for f in folds) < proto.min_rows:
        return None
    else:
        hyper = {"K": proto.K, "kappa0": proto.kappa0, "nu0": proto.nu0, "dim": int(X[names[0]].shape[1])}
    nh = {k: hyper[k] for k in ("K", "kappa0", "nu0")}
    ref, keep = [], []
    for fit_n, ev_n in (folds, folds[::-1]):
        m = NIWDensity(**nh, device=device).fit({v: X[v] for v in fit_n})
        for v in ev_n:
            r = train[v]
            ref.append(m.frame_scores(*_rows(r, True, train_every, drop_startup), r.n_frames))
            t = np.arange(int(r.n_frames))
            keep.append((t % train_every == 0) & (t >= int(r.startup) if drop_startup else True))
    ref = np.concatenate(ref)[np.concatenate(keep)]
    floor = ref[np.isfinite(ref)].min()
    full = NIWDensity(**nh, device=device).fit(X)
    scores = {}
    for v, r in test.items():
        s = full.frame_scores(*_rows(r, False, train_every, drop_startup), r.n_frames)
        scores[v] = np.where(np.isfinite(s), s, floor)
    hyper["train_rows"] = int(sum(len(x) for x in X.values()))
    return AttributeFit(scores, np.where(np.isfinite(ref), ref, floor), float(floor), hyper, full)


def scene_of(video: str) -> str:
    """ShanghaiTech scene of a video name: the prefix before '_' (01_0014 -> 01)."""
    return video.split("_")[0]


def fit_attribute_by_scene(train: dict[str, VideoRows], test: dict[str, VideoRows], device: str = "cpu",
                           seed: int = 0, **kw) -> dict[str, AttributeFit | None]:
    """tools/velpilot.py score --scenes: one fit_attribute per scene of the test videos, each with default_rng(seed)."""
    out = {}
    for g in sorted({scene_of(v) for v in test}):
        tr = {v: r for v, r in train.items() if scene_of(v) == g}
        te = {v: test[v] for v in sorted(test) if scene_of(v) == g}
        out[g] = fit_attribute(tr, te, np.random.default_rng(seed), device, **kw)
    return out
