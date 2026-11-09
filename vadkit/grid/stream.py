"""Native streaming grid branch: residual channel I (AEAN autoencoder residual, FRX-Bayes cell statistics) and the DIS
flow channels Fs (cell GMM-UBM of the speed) and Fd (FRX-Bayes of the 9-d cell flow vector), one frame at a time.

The fit replays evaluate.py (the archived pool48_mc4 / mc5 runs of the paper: 2d autoencoder, stride 8, statistics on
t = T-1 :: 3, alpha 0.5, cross-fitted references of I on t = T-1 :: 15 with video-parity folds; flow channels by
aean.channels.fit_channel, DIS medium, 8 x 8 cells, pool 48) by calling the same aean functions, so the per-frame
numerics are those of the research code. The streaming step computes, from the current and the gap-th previous frame
only, the same raw frame scores as the batch path (tests/vadkit/test_grid_stream.py); calibration and fusion happen in
StreamingVAD with the other channels.

State files are plain npz + json (no pickle); the autoencoder weights are referenced by path and sha256, not copied.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import torch

from aean.channels import CellRX, CellUBM, fit_channel, flow_video, frame_scores_from_maps, speed
from aean.data import to_unit
from aean.models import build as build_ae
from aean.scoring import BackgroundStats, frame_score, local_mean_max, residual_of

CHANNELS = ("I", "Fs", "Fd")          # vadkit/grid/precomputed.py GRID_NAMES order


@dataclass
class GridConfig:
    """Paper configuration (mc4.sh / mc5.sh defaults of evaluate.py); size None keeps the native frame size (UCSD)."""
    gap: int = 1
    size: tuple[int, int] | None = None          # (H, W) of the grey frame; Avenue / ShanghaiTech (180, 320)
    stride: int = 8
    pool: int = 48
    cell: int = 8                                # flow cells
    alpha: float = 0.5
    stats_every: int = 3
    ref_every: int = 15                          # I references on t = T-1 :: 15: the archived mc4 runs (3 Oct 2026,
                                                 # 12:06) predate commit 34af7eb, which changed evaluate.py to :: 9
    flow_preset: str = "medium"

    @staticmethod
    def for_dataset(ds: str) -> "GridConfig":
        """Gap and grey size of the paper's caches (prepare_ucsd.py native, prepare_avenue / shanghaitech 320x180)."""
        if ds.startswith("ped"):
            return GridConfig(gap=1)
        return GridConfig(gap=2, size=(180, 320))


def to_gray(bgr: np.ndarray, size: tuple[int, int] | None) -> np.ndarray:
    """BGR (or grey) uint8 -> grey uint8 at the grid size (INTER_AREA, as prepare_avenue.py)."""
    g = bgr if bgr.ndim == 2 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if size is not None and g.shape != tuple(size):
        g = cv2.resize(g, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return g


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_ae(run: str | Path, device: str):
    """AEAN autoencoder of a train.py run (model.pt + args.json) -> (ae in eval mode, args)."""
    run = Path(run)
    cfg = json.loads((run / "args.json").read_text())
    if cfg["variant"] != "2d":
        raise ValueError(f"{run}: variant {cfg['variant']!r}; the paper's grid branch uses the 2d autoencoder")
    ae, _, _ = build_ae(cfg["variant"], cfg["T"])
    ae.load_state_dict(torch.load(run / "model.pt", map_location=device, weights_only=True)["ae"])
    return ae.to(device).eval(), cfg


@torch.no_grad()
def residual(ae, gray: np.ndarray | torch.Tensor, patch: int, stride: int, device: str) -> torch.Tensor:
    """Residual (1, H, W) of one grey frame (evaluate.py residual_at for the 2d variant)."""
    x = to_unit(torch.as_tensor(gray, device=device))[None]
    return residual_of(ae, x, "2d", patch, stride)


def pixel_map(cell_map: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Cell map (gy, gx) -> pixel map (H, W), as aean.channels.frame_scores_from_maps."""
    ch, cw = math.ceil(H / cell_map.shape[0]), math.ceil(W / cell_map.shape[1])
    return cell_map.repeat_interleave(ch, 0).repeat_interleave(cw, 1)[:H, :W]


def local_mean(m: torch.Tensor, P: int) -> torch.Tensor:
    """P x P local mean of a pixel map (H, W), the map whose maximum is the frame score (local_mean_max)."""
    p = P // 2
    s = torch.nn.functional.pad(m.double()[None], (p + 1, p, p + 1, p)).cumsum(1).cumsum(2)[0]
    return ((s[P:, P:] - s[:-P, P:] - s[P:, :-P] + s[:-P, :-P]) / (P * P)).float()[:m.shape[0], :m.shape[1]]


# ---------------------------------------------------------------------------------------------- state
@dataclass
class GridState:
    """Fitted cell statistics of one camera (scene) and the normal reference scores of I, Fs, Fd."""
    config: GridConfig
    H: int
    W: int
    res: dict[str, np.ndarray]        # mu_b (G,1), prec_b (G,1,1): FRX-Bayes residual statistics, 16 x 16 cells
    ubm: dict[str, np.ndarray]        # w (Gf,K), mu (Gf,K,2), P (K,2,2), ldP (K,)
    rx: dict[str, np.ndarray]         # mu_b (Gf,9), prec_b (Gf,9,9)
    ref: dict[str, np.ndarray]        # I, Fs, Fd cross-fitted normal frame scores
    ae_patch: int = 16
    ae_path: str = ""
    ae_sha256: str = ""
    meta: dict = field(default_factory=dict)

    def save(self, out: str | Path) -> None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        arrays = {f"{g}/{k}": v for g, d in (("res", self.res), ("ubm", self.ubm), ("rx", self.rx), ("ref", self.ref))
                  for k, v in d.items()}
        np.savez(out / "grid.npz", **arrays)
        (out / "grid.json").write_text(json.dumps(
            {"config": asdict(self.config), "H": self.H, "W": self.W, "ae_patch": self.ae_patch,
             "ae_path": self.ae_path, "ae_sha256": self.ae_sha256, "meta": self.meta}, indent=1))

    @staticmethod
    def load(path: str | Path) -> "GridState":
        path = Path(path)
        j = json.loads((path / "grid.json").read_text())
        c = j["config"]
        c["size"] = tuple(c["size"]) if c.get("size") else None
        groups: dict[str, dict] = {"res": {}, "ubm": {}, "rx": {}, "ref": {}}
        with np.load(path / "grid.npz") as z:
            for k in z.files:
                g, n = k.split("/", 1)
                groups[g][n] = z[k]
        return GridState(GridConfig(**c), j["H"], j["W"], groups["res"], groups["ubm"], groups["rx"], groups["ref"],
                         j["ae_patch"], j["ae_path"], j["ae_sha256"], j.get("meta", {}))


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _flow_feats(videos: dict[str, np.ndarray], cfg: GridConfig, workers: int) -> dict[str, np.ndarray]:
    fn = partial(flow_video, gap=cfg.gap, cell=cfg.cell, preset=cfg.flow_preset)
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            return dict(zip(videos, ex.map(fn, videos.values())))
    return {k: fn(v) for k, v in videos.items()}


def fit_grid(train: dict[str, np.ndarray], ae_run: str | Path, cfg: GridConfig, device: str = "cpu",
             workers: int = 1, flow_cache: dict[str, np.ndarray] | None = None) -> GridState:
    """Fit on the grey normal training videos {name: uint8 (N, H, W)} (sorted by name, as load_split), exactly as
    evaluate.py: residual statistics and the parity-fold references of I, then fit_channel for Fs and Fd."""
    train = {k: train[k] for k in sorted(train)}
    ae, acfg = load_ae(ae_run, device)
    P, T, s = acfg["patch"], acfg["T"], cfg.stats_every
    H, W = next(iter(train.values())).shape[1:]
    stats = BackgroundStats(1, H, W, P, device, cfg.alpha)
    folds = [BackgroundStats(1, H, W, P, device, cfg.alpha) for _ in range(2)]
    for vi, v in enumerate(train.values()):
        vt = to_unit(torch.from_numpy(v).to(device))
        for t in range(T - 1, len(vt), s):
            d = residual_of(ae, vt[t:t + 1], "2d", P, cfg.stride)
            stats.update(d)
            folds[vi % 2].update(d)
    stats.finalize()
    folds = [f.finalize() if f.n.sum() > 0 else stats for f in folds]
    ref_i = []
    for vi, v in enumerate(train.values()):
        vt = to_unit(torch.from_numpy(v).to(device))
        f = folds[1 - vi % 2]
        for t in range(T - 1, len(vt), cfg.ref_every):
            d = residual_of(ae, vt[t:t + 1], "2d", P, cfg.stride)
            ref_i.append(frame_score(f.bayes_delta(d).reshape(H, W), cfg.pool))
    feats = flow_cache if flow_cache is not None else _flow_feats(train, cfg, workers)
    feats = {k: feats[k] for k in train}
    ubm, _, ref_fs, _ = fit_channel({k: speed(f) for k, f in feats.items()}, partial(CellUBM, device),
                                    H=H, W=W, pool=cfg.pool)
    C, gy, gx = next(iter(feats.values())).shape[1:]
    rx, _, ref_fd, _ = fit_channel(feats, partial(CellRX, C, gy, gx, device), H=H, W=W, pool=cfg.pool)
    run = Path(ae_run)
    return GridState(cfg, H, W,
                     res={"mu_b": _np(stats.mu_b), "prec_b": _np(stats.prec_b)},
                     ubm={"w": _np(ubm.w), "mu": _np(ubm.mu), "P": _np(ubm.P), "ldP": _np(ubm.ldP)},
                     rx={"mu_b": _np(rx.stats.mu_b), "prec_b": _np(rx.stats.prec_b)},
                     ref={"I": np.asarray(ref_i, np.float64), "Fs": np.asarray(ref_fs, np.float64),
                          "Fd": np.asarray(ref_fd, np.float64)},
                     ae_patch=P, ae_path=str(run), ae_sha256=sha256(run / "model.pt"),
                     meta={"train_videos": list(train), "T": T, "device": device})


# ---------------------------------------------------------------------------------------------- streaming
class GridBranch:
    """Causal grid branch: step(bgr) -> raw frame scores {I, Fs, Fd} of the newest frame (Fs, Fd from t >= gap)."""

    def __init__(self, state: GridState, ae_run: str | Path | None = None, device: str = "cpu",
                 parallel_flow: bool = False, check_sha: bool = True):
        self.state, self.cfg, self.device = state, state.config, device
        run = Path(ae_run or state.ae_path)
        if check_sha and state.ae_sha256 and sha256(run / "model.pt") != state.ae_sha256:
            raise ValueError(f"{run}/model.pt differs from the autoencoder the state was fitted with")
        self.ae, acfg = load_ae(run, device)
        self.P = acfg["patch"]
        t = lambda a: torch.as_tensor(a, device=device)
        self.res = BackgroundStats(1, state.H, state.W, self.P, device, self.cfg.alpha)
        self.res.mu_b, self.res.prec_b = t(state.res["mu_b"]), t(state.res["prec_b"])
        gy, gx = math.ceil(state.H / self.cfg.cell), math.ceil(state.W / self.cfg.cell)
        self.ubm = CellUBM(device)
        self.ubm.w, self.ubm.mu, self.ubm.P, self.ubm.ldP = (t(state.ubm[k]) for k in ("w", "mu", "P", "ldP"))
        self.ubm.C, self.ubm.g = int(state.ubm["mu"].shape[-1]), (gy, gx)
        self.rx = CellRX(int(state.rx["mu_b"].shape[-1]), gy, gx, device, self.cfg.alpha)
        self.rx.stats.mu_b, self.rx.stats.prec_b = t(state.rx["mu_b"]), t(state.rx["prec_b"])
        from concurrent.futures import ThreadPoolExecutor
        self.pool = ThreadPoolExecutor(1, thread_name_prefix="vadkit-gridflow") if parallel_flow else None
        self.keep_maps = False
        self.maps: dict[str, torch.Tensor] = {}
        self.reset()

    def reset(self) -> None:
        self.hist: deque[np.ndarray] = deque(maxlen=self.cfg.gap + 1)
        self.t = 0

    def _flow(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Cell flow features (9, gy, gx), float16 log1p, of the pair (t - gap, t): flow_video on two frames."""
        return flow_video(np.stack([a, b]), gap=1, cell=self.cfg.cell, preset=self.cfg.flow_preset)[-1]

    def step(self, bgr: np.ndarray) -> dict[str, float]:
        return self.step_gray(to_gray(bgr, self.cfg.size))

    @torch.no_grad()
    def step_gray(self, g: np.ndarray) -> dict[str, float]:
        """Raw scores of the newest grey frame (no calibration); Fs, Fd are absent for t < gap."""
        H, W = self.state.H, self.state.W
        if g.shape != (H, W):
            raise ValueError(f"grey frame {g.shape}, the state was fitted on {(H, W)}")
        self.hist.append(g)
        fut = None
        have_flow = len(self.hist) > self.cfg.gap
        if have_flow and self.pool is not None:
            fut = self.pool.submit(self._flow, self.hist[0], g)     # overlaps the autoencoder
        d = residual(self.ae, g, self.P, self.cfg.stride, self.device)
        m_i = self.res.bayes_delta(d).reshape(H, W)
        out = {"I": frame_score(m_i, self.cfg.pool)}
        maps = {"I": m_i} if self.keep_maps else None
        if have_flow:
            f = (fut.result() if fut is not None else self._flow(self.hist[0], g))[None]
            for name, model, x in (("Fs", self.ubm, speed(f)), ("Fd", self.rx, f)):
                cm = model.maps(x)
                out[name] = float(frame_scores_from_maps(cm, H, W, self.cfg.pool)[0])
                if maps is not None:
                    maps[name] = pixel_map(cm[0], H, W)
        if maps is not None:
            self.maps = {k: local_mean(v.float(), self.cfg.pool) for k, v in maps.items()}
        self.t += 1
        return out


def batch_scores(state: GridState, video: np.ndarray, ae_run: str | Path | None = None,
                 device: str = "cpu") -> dict[str, np.ndarray]:
    """Raw scores of a whole grey video with the batch functions of evaluate.py (flow_video over the video,
    model.maps over all frames); frames t < gap of Fs / Fd are NaN (evaluate.py copies a later flow there)."""
    br = GridBranch(state, ae_run, device)
    H, W, cfg = state.H, state.W, state.config
    vt = to_unit(torch.from_numpy(video).to(device))
    with torch.no_grad():
        i = np.array([frame_score(br.res.bayes_delta(residual_of(br.ae, vt[t:t + 1], "2d", br.P, cfg.stride))
                                  .reshape(H, W), cfg.pool) for t in range(len(vt))])
        f = flow_video(video, gap=cfg.gap, cell=cfg.cell, preset=cfg.flow_preset)
        fs = frame_scores_from_maps(br.ubm.maps(speed(f)), H, W, cfg.pool).astype(np.float64)
        fd = frame_scores_from_maps(br.rx.maps(f), H, W, cfg.pool).astype(np.float64)
    fs[:cfg.gap] = fd[:cfg.gap] = np.nan
    return {"I": i, "Fs": fs, "Fd": fd}


__all__ = ["CHANNELS", "GridConfig", "GridState", "GridBranch", "fit_grid", "batch_scores", "to_gray", "load_ae",
           "local_mean_max"]
