"""HB-POT: hierarchical (empirical Bayes) peaks-over-threshold calibration of grid channels across scenes, stage 1 of
docs/tez_notlari/hiyerarsik_kalibrasyon_on_kayit_2026-11-05.md (binding; design and equations in §3.1-3.4 of
hiyerarsik_kalibrasyon_on_kayit_TASLAK_2026-11-05.md). Per channel, every scene s keeps the FisherCalibrator body
below u_s = Q0.9 of its normal reference; beyond u_s the tail is a GPD with theta_s = (xi_s, rho_s),
rho_s = log(sigma_s / d_s), d_s = Q0.9 - Q0.5:

    theta_hat_s   GPD MLE of the excesses over u_s (location 0, xi in [-0.5, 1.5])
    Sigma_s       covariance of theta_hat over a video-cluster bootstrap (B = 500, seed 0), times J_s / (J_s - 1)
    (mu, T)       REML of theta_hat_s ~ N(mu, T + Sigma_s), T = diag(tau_xi^2, tau_rho^2) (truncation ignored)
    posterior     V_s = (T^-1 + Sigma_s^-1)^-1, theta~_s = V_s (T^-1 mu + Sigma_s^-1 theta_hat_s)
    mode          of the posterior truncated to xi >= 0 (closed form), sigma* = d_s exp(rho*)

Arms (no others): t0 = FisherCalibrator (xi = 0, sigma = mean excess), bit for bit; t1 = no pooling (T^-1 = 0, the
same truncated mode with V_s = Sigma_s); t2 = the truncated EB posterior mode (primary). With a single scene there is
nothing to pool and t2 is t1 by definition (Ped1, Ped2, Avenue)."""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq, minimize, minimize_scalar

from vadkit.calibration.gpd import GPDTailCalibrator

Q = 0.9                      # tail threshold quantile (FisherCalibrator)
XI_BOUNDS = (-0.5, 1.5)
B_BOOT = 500
SEED = 0
MODES = ("t0", "t1", "t2")


# ---------------------------------------------------------------------------------------------------- GPD MLE
def _profile(xi: float, y: np.ndarray) -> tuple[float, float]:
    """Log-likelihood of excesses y (location 0) maximised over sigma at shape xi, and that sigma. The score equation
    in t = xi / sigma, sum(t y / (1 + t y)) = n xi / (1 + xi), is monotone in t with one root (-1 < xi)."""
    n = len(y)
    if abs(xi) < 1e-10:
        sigma = float(y.mean())
        return -n * math.log(sigma) - n, sigma
    target = n * xi / (1.0 + xi)
    f = lambda t: float(np.sum(t * y / (1.0 + t * y))) - target
    if xi > 0:                                          # f(0) = -target < 0, f(inf) = n / (1 + xi) > 0
        hi = 1.0 / float(y.mean())
        while f(hi) <= 0:
            hi *= 4.0
        t = brentq(f, 0.0, hi, xtol=1e-300, rtol=1e-13, maxiter=500)
    else:                                               # f(0) = -target > 0, f(-1/max y) = -inf
        lo = -(1.0 - 1e-12) / float(y.max())
        t = brentq(f, lo, 0.0, xtol=1e-300, rtol=1e-13, maxiter=500)
    sigma = xi / t
    return -n * math.log(sigma) - (1.0 + 1.0 / xi) * float(np.sum(np.log1p(t * y))), sigma


def gpd_fit(exc: np.ndarray, xi_bounds: tuple[float, float] = XI_BOUNDS) -> tuple[float, float]:
    """(xi, sigma) MLE of GPD excesses (location 0) with xi in xi_bounds: profile likelihood in xi on a 21-point grid,
    refined by bounded Brent between the neighbours of the best grid point (the bounds themselves are grid points)."""
    y = np.asarray(exc, dtype=np.float64)
    if len(y) < 3 or not np.all(y > 0) or not np.all(np.isfinite(y)):
        raise ValueError(f"gpd_fit needs >= 3 finite positive excesses, got {len(y)}")
    scale = float(y.mean())
    ys = y / scale                                      # scale-free: sigma is returned times `scale`
    lo, hi = xi_bounds
    grid = np.linspace(lo, hi, 21)
    ll = [_profile(float(x), ys)[0] for x in grid]
    i = int(np.argmax(ll))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    res = minimize_scalar(lambda x: -_profile(float(x), ys)[0], bounds=(a, b), method="bounded",
                          options={"xatol": 1e-9})
    xi = float(res.x) if -res.fun > ll[i] else float(grid[i])
    return xi, _profile(xi, ys)[1] * scale


# ------------------------------------------------------------------------------------------- per-scene estimate
def scene_theta(ref: np.ndarray, q: float = Q, xi_bounds: tuple[float, float] = XI_BOUNDS) -> dict:
    """theta_hat = (xi, log(sigma / d)) of one reference, with u = Q_q (as FisherCalibrator), d = u - Q_0.5."""
    ref = np.asarray(ref, dtype=np.float64)
    u = float(np.quantile(ref, q))
    d = u - float(np.quantile(ref, 0.5))
    if not d > 0:
        raise ValueError(f"d = Q{q} - Q0.5 = {d} must be > 0")
    exc = ref[ref > u] - u
    xi, sigma = gpd_fit(exc, xi_bounds)
    return {"theta": np.array([xi, math.log(sigma / d)]), "u": u, "d": d, "sigma": sigma, "n_exc": int(len(exc))}


def _groups(ref: np.ndarray, vid: np.ndarray) -> list[np.ndarray]:
    ref, vid = np.asarray(ref), np.asarray(vid)
    if ref.shape != vid.shape:
        raise ValueError(f"reference {ref.shape} and video ids {vid.shape} differ in size")
    return [ref[vid == j] for j in np.unique(vid)]


def video_bootstrap(ref: np.ndarray, vid: np.ndarray, B: int = B_BOOT, seed: int = SEED, q: float = Q,
                    xi_bounds: tuple[float, float] = XI_BOUNDS) -> dict:
    """Video-cluster bootstrap of theta_hat: each replicate draws J videos with replacement (rng default_rng(seed),
    one per call), refits u, d and theta_hat; Sigma = sample covariance (ddof 1) times J / (J - 1)."""
    g = _groups(ref, vid)
    J = len(g)
    if J < 2:
        raise ValueError("the video bootstrap needs >= 2 videos")
    rng = np.random.default_rng(seed)
    th = np.empty((B, 2))
    for b in range(B):
        th[b] = scene_theta(np.concatenate([g[j] for j in rng.integers(0, J, J)]), q, xi_bounds)["theta"]
    return {"Sigma": np.cov(th.T) * J / (J - 1), "J": J, "replicates": th}


# --------------------------------------------------------------------------------------------- empirical Bayes
def _reml_parts(tau: np.ndarray, theta: np.ndarray, Sigma: np.ndarray) -> tuple[float, np.ndarray]:
    """Negative REML log-likelihood at T = diag(tau^2) and the GLS mean."""
    M = Sigma + np.diag(np.asarray(tau, dtype=np.float64) ** 2)[None]
    W = np.linalg.inv(M)
    A = W.sum(0)
    mu = np.linalg.solve(A, np.einsum("sij,sj->i", W, theta))
    r = theta - mu
    ld = np.linalg.slogdet(M)[1].sum()
    return 0.5 * (ld + float(np.einsum("si,sij,sj->", r, W, r)) + np.linalg.slogdet(A)[1]), mu


def eb_reml(theta: np.ndarray, Sigma: np.ndarray, tau_max: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """(mu, T) of theta_hat_s ~ N(mu, T + Sigma_s), T = diag(tau^2), by REML: grid over tau in {0} U logspace(-3, 1)
    per coordinate, refined by L-BFGS-B on tau in [0, tau_max] from the best grid point (kept if not improved)."""
    theta, Sigma = np.asarray(theta, dtype=np.float64), np.asarray(Sigma, dtype=np.float64)
    if len(theta) < 2:
        raise ValueError("REML needs >= 2 scenes (with one scene t2 is t1)")
    grid = np.concatenate([[0.0], np.logspace(-3, 1, 25)])
    best, best_tau = np.inf, None
    for a in grid:
        for b in grid:
            v = _reml_parts(np.array([a, b]), theta, Sigma)[0]
            if v < best:
                best, best_tau = v, np.array([a, b])
    res = minimize(lambda t: _reml_parts(t, theta, Sigma)[0], best_tau, method="L-BFGS-B",
                   bounds=[(0.0, tau_max)] * 2)
    tau = np.asarray(res.x, dtype=np.float64) if res.fun < best else best_tau
    return _reml_parts(tau, theta, Sigma)[1], np.diag(tau ** 2)


def posterior(theta_hat: np.ndarray, Sigma: np.ndarray, mu: np.ndarray | None = None,
              T: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(theta~, V) of the Normal-Normal model; T None is T^-1 = 0 (t1: theta_hat, Sigma). Written as
    theta~ = mu + T (T + Sigma)^-1 (theta_hat - mu), V = T (T + Sigma)^-1 Sigma, which also holds for a singular T."""
    theta_hat, Sigma = np.asarray(theta_hat, dtype=np.float64), np.asarray(Sigma, dtype=np.float64)
    if T is None:
        return theta_hat.copy(), Sigma.copy()
    K = T @ np.linalg.inv(T + Sigma)
    V = K @ Sigma
    return mu + K @ (theta_hat - mu), (V + V.T) / 2


def truncated_mode(theta: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Mode of N(theta, V) truncated to xi >= 0: theta itself if xi >= 0, else xi = 0 and
    rho = rho~ - (V[rho, xi] / V[xi, xi]) xi~ (the ratio is 0 when V[xi, xi] = 0, a point mass in xi)."""
    xi, rho = float(theta[0]), float(theta[1])
    if xi >= 0:
        return np.array([xi, rho])
    ratio = float(V[1, 0]) / float(V[0, 0]) if V[0, 0] > 0 else 0.0
    return np.array([0.0, rho - ratio * xi])


# ---------------------------------------------------------------------------------------------------- groups
def _check(refs: dict, ids: dict) -> None:
    if ids is None or set(ids) != set(refs) or any(ids[g] is None for g in refs):
        raise ValueError("a group calibrator needs the video id of every reference value of every group")
    for g in refs:
        if np.shape(refs[g]) != np.shape(ids[g]):
            raise ValueError(f"group {g!r}: reference {np.shape(refs[g])} and video ids {np.shape(ids[g])} differ")


def _boot_job(args):
    ref, vid, B, seed, q, xi_bounds = args
    return video_bootstrap(ref, vid, B, seed, q, xi_bounds)["Sigma"]


def hb_fit(refs: dict, ids: dict, mode: str = "t2", B: int = B_BOOT, seed: int = SEED, q: float = Q,
           xi_bounds: tuple[float, float] = XI_BOUNDS, workers: int = 1, sigmas: dict | None = None) -> dict:
    """Per-group tail parameters of one channel: {"groups": {g: {...}}, "mu", "T"}; each group holds xi, sigma (of
    the arm) and, for t1 / t2, theta_hat, Sigma, J, d, theta~, V. sigmas: precomputed {g: Sigma} (skips the
    bootstrap). With one group t2 is t1."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} not in {MODES}")
    _check(refs, ids)
    gs = list(refs)
    if mode == "t0":
        return {"groups": {g: {"xi": 0.0, "sigma": None} for g in gs}, "mu": None, "T": None, "mode": mode}
    est = {g: scene_theta(refs[g], q, xi_bounds) for g in gs}
    if sigmas is None:
        jobs = [(np.asarray(refs[g], dtype=np.float64), np.asarray(ids[g]), B, seed, q, xi_bounds) for g in gs]
        if workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(workers) as ex:
                sig = list(ex.map(_boot_job, jobs))
        else:
            sig = [_boot_job(j) for j in jobs]
        sigmas = dict(zip(gs, sig))
    mu = T = None
    if mode == "t2" and len(gs) > 1:
        mu, T = eb_reml(np.stack([est[g]["theta"] for g in gs]), np.stack([sigmas[g] for g in gs]))
    out = {}
    for g in gs:
        th, V = posterior(est[g]["theta"], sigmas[g], mu, T)
        star = truncated_mode(th, V)
        out[g] = {"xi": float(star[0]), "sigma": est[g]["d"] * math.exp(float(star[1])), "theta_hat": est[g]["theta"],
                  "Sigma": sigmas[g], "J": int(len(np.unique(ids[g]))), "u": est[g]["u"], "d": est[g]["d"],
                  "n_exc": est[g]["n_exc"], "theta_post": th, "V": V, "theta_star": star}
    return {"groups": out, "mu": mu, "T": T, "mode": mode}


def calibrators(fit: dict, refs: dict) -> dict:
    """{g: GPDTailCalibrator} of hb_fit's tail parameters on each group's reference (t0: sigma = mean excess)."""
    return {g: GPDTailCalibrator(p["xi"], p["sigma"]).fit(refs[g]) for g, p in fit["groups"].items()}


def fit_groups(refs_by_scene: dict, video_ids: dict, mode: str = "t2", **kw) -> dict:
    """{scene: calibrator} of one channel; refs_by_scene {scene: normal reference}, video_ids {scene: video id of each
    reference value} (required for every mode; missing ids raise)."""
    return calibrators(hb_fit(refs_by_scene, video_ids, mode, **kw), refs_by_scene)


class HBGroupCalibrator:
    """Group calibrator 'fisher_hb' (registry kind 'group_calibrator'): fit_groups(refs, ids) -> {group: Calibrator}."""

    def __init__(self, mode: str = "t2", B: int = B_BOOT, seed: int = SEED, workers: int = 1):
        if mode not in MODES:
            raise ValueError(f"mode {mode!r} not in {MODES}")
        self.mode, self.B, self.seed, self.workers = mode, B, seed, workers

    def fit_groups(self, refs: dict, ids: dict) -> dict:
        self.fit_ = hb_fit(refs, ids, self.mode, B=self.B, seed=self.seed, workers=self.workers)
        return calibrators(self.fit_, refs)
