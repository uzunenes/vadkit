"""Object branch, offline: frames -> feat2-format npz cache -> fit on normal training videos -> frame scores ->
calibration -> fusion -> evaluation (tools/objfeat2.py, tools/objscore2.py, tools/objfuse.py, tools/objfuse_sht.py)."""
from __future__ import annotations

import functools
import inspect
import json
import os
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from vadkit import __version__
from vadkit.attributes.velocity import velocity_rows
from vadkit.core.interfaces import AttributeExtractor, Calibrator, Objects
from vadkit.core.registry import _REG, available, build
from vadkit.density.niw import NIWDensity, VideoRows, _rows, fit_attribute, scene_of
from vadkit.eval.protocol import evaluate
from vadkit.grid.precomputed import GRID_NAMES
from vadkit.io.frames import frame_paths, read
from vadkit.perception.detector import default_device, empty_objects
from vadkit.temporal.median import causal_median

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
BASE_COLUMNS = {"f": (np.int32, ()), "box": (np.float32, (4,)), "src": (np.uint8, ()), "cls": (np.int16, ()),
                "conf": (np.float32, ())}


# ---------------------------------------------------------------------------------------------- config
def load_config(path: str | Path) -> dict:
    """YAML config from a file, or the name of a built-in one (vadkit/configs/<name>.yaml)."""
    p = Path(path)
    if not p.exists() and (CONFIGS / f"{path}.yaml").exists():
        p = CONFIGS / f"{path}.yaml"
    return yaml.safe_load(p.read_text())


def gap_of(config: dict, dataset: str | None = None) -> int:
    """Frame gap of the config: an int, or {dataset: gap}."""
    g = config.get("gap", 1)
    if isinstance(g, dict):
        if dataset not in g:
            raise KeyError(f"no gap for dataset {dataset!r} in the config (known: {sorted(g)})")
        return int(g[dataset])
    return int(g)


def build_with(kind: str, spec: dict | str, **extra):
    """registry.build plus the keyword arguments of extra (not None) that the back-end accepts and the spec lacks."""
    spec = {"name": spec} if isinstance(spec, str) else spec
    if spec["name"] in available(kind):
        params = inspect.signature(_REG[(kind, spec["name"])]).parameters
        extra = {k: v for k, v in extra.items() if k in params and v is not None}
    return build(kind, {"name": spec["name"], "args": {**extra, **spec.get("args", {})}})


def attribute(spec: dict) -> AttributeExtractor:
    """Extractor of a config attribute {name: channel, kind: registry name, args: {...}}."""
    return build("attribute", {"name": spec["kind"], "args": spec.get("args", {})})


def record_rows(ext: AttributeExtractor, rec: dict) -> tuple[np.ndarray, np.ndarray]:
    """(frame index, rows) of a feat2 npz record: velocity rows from the stored vel arrays, else ext.record_rows."""
    if hasattr(ext, "from_vel"):
        return rec["f"], ext.from_vel(rec["vel"], rec["box"], tuple(rec["frame_hw"]))
    if not hasattr(ext, "record_rows"):
        raise TypeError(f"{type(ext).__name__} has neither from_vel nor record_rows(rec): the object branch reads "
                        "stored feat2 records (vel / pose / app)")
    return ext.record_rows(rec)


def _saves_state(ext: AttributeExtractor) -> bool:
    """ext overrides AttributeExtractor.state() and from_state() (required when it sets raw_key)."""
    cls = type(ext)
    return (cls.state is not AttributeExtractor.state
            and getattr(cls.from_state, "__func__", None) is not AttributeExtractor.from_state.__func__)


def object_rows(ext: AttributeExtractor, objs: Objects, frame_hw: tuple[int, int]) -> np.ndarray:
    """Rows of one frame's objects carrying the raw feat2 quantities (attrs 'vel', 'app', pose / pose_ok); no flow."""
    if not len(objs):
        return np.zeros((0, 0))
    if hasattr(ext, "from_vel"):
        return ext.from_vel(objs.attrs["vel"], objs.box, frame_hw)
    return ext(objs, None, frame_hw)


def _flow_rows(perc: "Perception", pending: list) -> None:
    """Flows of the pending (cols, frame t - gap, frame t) pairs in one call; their 'vel' rows; empties pending."""
    flows = perc.flow.flow([p for _, p, _ in pending], [c for _, _, c in pending])
    for (cols, _, _), F in zip(pending, flows):
        cols["vel"] = velocity_rows(F, cols["box"])
    pending.clear()


def temporal_k(spec: dict | str = "causal_median") -> int:
    """Window of the config's temporal stage, the one stage that offline evaluate() and StreamingVAD both run:
    'causal_median' (k 9, the paper's protocol) or {name: causal_median, args: {k: <int>}}; anything else raises."""
    spec = {"name": spec} if isinstance(spec, str) else spec
    args = spec.get("args") or {}
    k = args.get("k", 9)
    if spec.get("name") != "causal_median" or set(args) - {"k"} or type(k) is not int or k < 1:
        raise ValueError(f"temporal {spec!r}: only causal_median with an integer k >= 1 is supported (offline "
                         "evaluation and the stream run the same smoother)")
    return k


def offline_smoother(spec: dict | str = "causal_median") -> Callable[[np.ndarray], np.ndarray]:
    """Array form of the temporal stage, per video: causal_median(s, k) (tools/objfuse.causal_median at k 9)."""
    return functools.partial(causal_median, k=temporal_k(spec))


def density_spec(spec: dict | str = "niw") -> dict:
    """{name, args} of the config's density: 'niw' or {name: niw, args: {...}}; args are NIWDensity's (fixed K, kappa0,
    nu0 together, or max_eval / min_rows of the held-out selection) and are forwarded to fit_attribute."""
    spec = {"name": spec} if isinstance(spec, str) else spec
    spec = {"name": spec["name"], "args": dict(spec.get("args") or {})}
    if spec["name"] != "niw":
        raise ValueError(f"density {spec['name']!r}: the object branch fits NIW mixtures (cross-fitted reference, "
                         f"held-out selection on normal data); known densities: {available('density')}")
    bad = set(spec["args"]) - {"K", "kappa0", "nu0", "max_eval", "min_rows"}
    if bad:
        raise ValueError(f"density args {sorted(bad)}: niw takes K, kappa0, nu0, max_eval, min_rows")
    build("density", spec)                       # K, kappa0, nu0 all or none
    return spec


def parse_shard(spec: str) -> tuple[int, int]:
    """'i/n' -> (i, n) with 0 <= i < n (tools/velpilot.shard)."""
    try:
        i, n = map(int, spec.split("/"))
    except ValueError:
        raise ValueError(f"shard {spec!r}: expected 'i/n'") from None
    if not 0 <= i < n:
        raise ValueError(f"shard {spec!r}: need 0 <= i < n")
    return i, n


def save_npz(dst: Path, **arrays) -> None:
    """Atomic write (tools/velpilot.save_npz): a killed run never leaves a truncated file that resume would skip."""
    tmp = dst.with_name(dst.stem + ".part.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, dst)


def _sync() -> None:
    """Wait for queued CUDA work (latency timing)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------------------------- perception
class Perception:
    """Detector, pose, flow t-gap -> t, embedding per frame into objs.pose / attrs 'vel', 'app'; none for t < gap.
    batch: flow pairs per call of the flow back-end, 1 per frame (tools/velpilot.py bench, not NeuFlow's padded 8);
    None keeps the back-end's default (extract batches pairs, as velpilot flowvel); a batch in the spec's args wins."""
    STAGES = ("detect", "pose_wait", "pose", "flow", "embed")

    def __init__(self, config: dict, gap: int, device: str | None = None, batch: int | None = 1):
        if not config.get("detector"):
            raise ValueError("the config has no perception back-ends (detector); replay stored objects instead")
        for k in ("detector", "pose", "flow", "encoder"):
            spec = config.get(k)
            if isinstance(spec, dict) and (spec.get("args") or {}).get("gap", gap) != gap:
                raise ValueError(f"{k} args set gap {spec['args']['gap']}, the pipeline gap is {gap}")
        self.gap = gap
        self.det = build_with("detector", config["detector"], gap=gap, device=device)
        self.pose, self.enc = (build_with(k, config[k], device=device) if config.get(k) else None
                               for k in ("pose", "encoder"))
        self.flow = build_with("flow", config["flow"], device=device, batch=batch) if config.get("flow") else None
        self.clock: dict[str, float] | None = None        # stage -> ms of the last frame, when set (bench)
        # parallel_flow: the flow of frame t (CPU DIS, or another device) runs in a worker thread while the detector,
        # pose and encoder run; same inputs and functions, so the objects are identical to the sequential order
        # parallel_pose: the pose model runs on every frame (after the start-up gap) in a second worker thread while the
        # detector runs; its output is used only for frames with person boxes, so the objects are again identical
        from concurrent.futures import ThreadPoolExecutor
        self.pool = (ThreadPoolExecutor(max_workers=1, thread_name_prefix="vadkit-flow")
                     if config.get("parallel_flow", False) and self.flow is not None else None)
        self.pose_pool = (ThreadPoolExecutor(max_workers=1, thread_name_prefix="vadkit-pose")
                          if config.get("parallel_pose", False) and self.pose is not None
                          and hasattr(self.pose, "predict") else None)
        self.reset()

    @property
    def columns(self) -> dict[str, tuple[type, tuple]]:
        """feat2 npz columns written by extract: dtype and row shape."""
        c = dict(BASE_COLUMNS)
        if self.flow:
            c["vel"] = (np.float32, (8,))
        if self.pose:
            c.update(pose=(np.float32, (34,)), pose_ok=(np.uint8, ()))
        if self.enc:
            c["app"] = (np.float16, (getattr(self.enc, "dim", 0),))
        return c

    def reset(self) -> None:
        """Start of a new video."""
        if hasattr(self.det, "reset"):
            self.det.reset()
        self.hist: deque[np.ndarray] = deque(maxlen=self.gap + 1)

    def _run(self, stage: str, fn, *args):
        """fn(*args), timed into self.clock[stage] when timing."""
        if self.clock is None:
            return fn(*args)
        _sync(); s = time.perf_counter()
        out = fn(*args)
        _sync(); self.clock[stage] = (time.perf_counter() - s) * 1e3
        return out

    @property
    def prev(self) -> np.ndarray:
        """Frame t - gap of the newest frame t (the flow source)."""
        return self.hist[0]

    def __call__(self, bgr: np.ndarray, flow: bool = True) -> Objects:
        """Objects of the newest frame with their raw attributes; flow False leaves 'vel' to the caller (batched)."""
        self.hist.append(bgr)
        if self.clock is not None:
            self.clock.update(dict.fromkeys(self.STAGES, 0.0))
        fut = pfut = None
        if self.pool is not None and flow and len(self.hist) > self.gap:
            fut = self.pool.submit(self.flow.flow, [self.prev], [bgr])     # overlaps detect / pose (/ embed)
        if self.pose_pool is not None and len(self.hist) > self.gap:
            pfut = self.pose_pool.submit(self.pose.predict, bgr)          # overlaps detect
        objs = self._run("detect", self.det.detect, bgr)
        if len(self.hist) <= self.gap or not len(objs):
            for f in (fut, pfut):
                if f is not None:
                    f.result()                              # keep at most one job per worker in flight
            return empty_objects()
        if pfut is not None:
            pred = self._run("pose_wait", pfut.result)
            objs = self._run("pose", self.pose.attach, bgr, objs, pred)
        elif self.pose:
            objs = self._run("pose", self.pose.attach, bgr, objs)
        if fut is not None:                     # embed while the flow finishes; attrs keep the order vel, app
            app = self._run("embed", self.enc.encode, bgr, objs.box) if self.enc else None
            objs.attrs["vel"] = velocity_rows(self._run("flow", fut.result)[0], objs.box)   # clock: the wait only
            if app is not None:
                objs.attrs["app"] = app
            return objs
        if self.flow and flow:
            objs.attrs["vel"] = velocity_rows(self._run("flow", self.flow.flow, [self.prev], [bgr])[0], objs.box)
        if self.enc:
            objs.attrs["app"] = self._run("embed", self.enc.encode, bgr, objs.box)
        return objs


# ---------------------------------------------------------------------------------------------- fitted state
@dataclass
class AttributeState:
    """What one attribute channel needs at test time."""
    extractor: AttributeExtractor
    density: NIWDensity
    calibrator: Calibrator
    ref: np.ndarray
    floor: float
    hyper: dict


def _split(state: dict, prefix: str, arrays: dict) -> dict:
    """JSON-ready scalars of state; its arrays go to arrays['<prefix>/<key>']."""
    meta = {}
    for k, v in state.items():
        if isinstance(v, np.ndarray) and v.ndim:
            arrays[f"{prefix}/{k}"] = v
        else:
            meta[k] = v.item() if isinstance(v, (np.ndarray, np.generic)) else v
    return meta


def _join(meta: dict, prefix: str, arrays: dict) -> dict:
    """Inverse of _split."""
    return {**meta, **{k[len(prefix) + 1:]: v for k, v in arrays.items() if k.startswith(prefix + "/")}}


@dataclass
class FittedState:
    """Fitted object branch {group: {attribute: AttributeState}}; group 'all', or the ShanghaiTech scene; gap: the
    frame gap (startup) of the training features."""
    config: dict
    groups: dict[str, dict[str, AttributeState]]
    scenes: bool = False
    gap: int | None = None

    @property
    def channels(self) -> list[str]:
        """Attributes fitted in every group, in config order: the channels scored, fused and streamed (objfuse_sht)."""
        return [s["name"] for s in self.config["attributes"] if all(s["name"] in g for g in self.groups.values())]

    def group_of(self, video: str) -> str:
        """Group whose models score a video."""
        return scene_of(video) if self.scenes else "all"

    def save(self, path: str | Path) -> None:
        """<path>/state.npz (arrays) and <path>/state.json (config, scalars); nothing pickled."""
        arrays, meta = {}, {}
        for g, attrs in self.groups.items():
            for a, st in attrs.items():
                p = f"{g}/{a}"
                arrays[f"{p}/ref"] = st.ref
                ext = (_split(st.extractor.state(), f"{p}/extractor", arrays) if st.extractor.raw_key is not None
                       else None)
                meta.setdefault(g, {})[a] = {"floor": st.floor, "hyper": st.hyper, "extractor": ext,
                                             "density": _split(st.density.state(), f"{p}/density", arrays),
                                             "calibrator": _split(vars(st.calibrator), f"{p}/calibrator", arrays)}
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.savez(path / "state.npz", **arrays)
        (path / "state.json").write_text(json.dumps({"vadkit": __version__, "scenes": self.scenes, "gap": self.gap,
                                                     "channels": self.channels, "config": self.config,
                                                     "groups": meta}, indent=1))

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "FittedState":
        """Inverse of save()."""
        path = Path(path)
        meta = json.loads((path / "state.json").read_text())
        with np.load(path / "state.npz") as A:
            arrays = dict(A)
        cfg = meta["config"]
        specs = {s["name"]: s for s in cfg["attributes"]}
        groups = {}
        for g, attrs in meta["groups"].items():
            groups[g] = {}
            for a, m in attrs.items():
                p = f"{g}/{a}"
                ext = attribute(specs[a])
                if m["extractor"] is not None:
                    ext = type(ext).from_state(_join(m["extractor"], f"{p}/extractor", arrays))
                cal = build("calibrator", cfg.get("calibrator", "fisher"))
                cal.__dict__.update(_join(m["calibrator"], f"{p}/calibrator", arrays))
                density = NIWDensity.from_state(_join(m["density"], f"{p}/density", arrays), device)
                groups[g][a] = AttributeState(ext, density, cal, arrays[f"{p}/ref"], m["floor"], m["hyper"])
        state = cls(cfg, groups, meta["scenes"], meta.get("gap"))
        if meta.get("channels", state.channels) != state.channels:
            raise ValueError(f"{path}: state.json channels {meta['channels']} differ from its groups' {state.channels}")
        return state


# ---------------------------------------------------------------------------------------------- branch
class ObjectBranch:
    """extract -> load -> fit -> score -> calibrate -> fuse -> evaluate, set up by a config dict (vadkit/configs)."""

    def __init__(self, config: dict, scenes: bool = False, device: str | None = None):
        self.config, self.scenes, self.device = config, scenes, device
        self.state: FittedState | None = None
        self.density = density_spec(config.get("density", "niw"))
        temporal_k(config.get("temporal", "causal_median"))       # the smoother of evaluate() and of the stream

    def extract(self, frames_root: str | Path, split: str, out_dir: str | Path, gap: int,
                train_every: int | None = None, shard: str = "0/1") -> list[Path]:
        """Write missing <out_dir>/<split>/<video>.npz (feat2 + startup, train_every: default the config's); rows for
        t >= gap only; flow pairs of the frames with objects in batches of the flow back-end (tools/velpilot.py
        flowvel); shard 'i/n': videos i::n of the sorted list (tools/velpilot.shard)."""
        every = int(self.config.get("train_every", 1) if train_every is None else train_every)
        every = every if split == "training" else 1
        width = int(self.config.get("width", 640))
        i, n = parse_shard(shard)
        perc, written = None, []
        for vid, paths in list(frame_paths(frames_root, split).items())[i::n]:
            dst = Path(out_dir) / split / f"{vid}.npz"
            if dst.exists():
                continue
            perc = perc or Perception(self.config, gap, self.device, batch=None)
            chunk = int(getattr(perc.flow, "batch", 1) or 1)
            rows, pending, last = [], [], None
            use_par = bool(self.config.get("parallel_flow", False)) and perc.pool is not None
            for t in range(gap, len(paths)):
                if t % every:
                    continue
                if last != t - 1:                          # (re)fill the causal history t-gap .. t-1
                    perc.reset()
                    for u in range(t - gap, t):
                        perc(read(paths[u], width), flow=False)
                bgr = read(paths[t], width)
                if use_par:
                    objs, last = perc(bgr, flow=True), t
                    if not len(objs):
                        continue
                    cols = {"f": np.full(len(objs), t, np.int32), "box": objs.box, "src": objs.src, "cls": objs.cls,
                            "conf": objs.conf, "pose": objs.pose, "pose_ok": objs.pose_ok, **objs.attrs}
                    rows.append(cols)
                else:
                    objs, last = perc(bgr, flow=False), t
                    if not len(objs):
                        continue
                    cols = {"f": np.full(len(objs), t, np.int32), "box": objs.box, "src": objs.src, "cls": objs.cls,
                            "conf": objs.conf, "pose": objs.pose, "pose_ok": objs.pose_ok, **objs.attrs}
                    rows.append(cols)
                    if perc.flow:
                        pending.append((cols, perc.prev, bgr))
                        if len(pending) == chunk:
                            _flow_rows(perc, pending)
            if pending:
                _flow_rows(perc, pending)
            out = {k: (np.concatenate([c[k] for c in rows]) if rows else np.zeros((0, *shape))).astype(dt)
                   for k, (dt, shape) in perc.columns.items()}
            dst.parent.mkdir(parents=True, exist_ok=True)
            save_npz(dst, **out, n_frames=len(paths), startup=gap, frame_hw=np.array(read(paths[0], width).shape[:2]),
                     train_every=every)
            written.append(dst)
        return written

    @staticmethod
    def load(features_dir: str | Path, splits: tuple[str, ...] = ("training", "testing")) -> tuple[dict, ...]:
        """Records {video: {key: array}} of each split of a feat2-format directory, videos sorted."""
        return tuple({p.stem: dict(np.load(p)) for p in sorted((Path(features_dir) / s).glob("*.npz"))
                      if not p.name.endswith(".part.npz")} for s in splits)

    def fit(self, train: dict[str, dict]) -> FittedState:
        """Per group one default_rng(seed) shared by the attributes in config order, as tools/objscore2.py."""
        cfg = self.config
        every, drop = int(cfg.get("train_every", 1)), bool(cfg.get("drop_startup", False))
        for v, r in train.items():   # features written with a training subsample need a multiple of it (velpilot)
            if every % int(r.get("train_every", 1)):
                raise ValueError(f"{v} was extracted with train_every {int(r['train_every'])}; the config's "
                                 f"train_every {every} must be a multiple of it")
        gaps = {int(r["startup"]) for r in train.values()}
        if len(gaps) != 1:
            raise ValueError(f"training features of one fit need one frame gap (startup), got {sorted(gaps)}")
        groups = {}
        for v in sorted(train):
            groups.setdefault(scene_of(v) if self.scenes else "all", []).append(v)
        out = {}
        for g, names in groups.items():
            rng = np.random.default_rng(int(cfg.get("seed", 0)))
            out[g] = {}
            for spec in cfg["attributes"]:
                ext = attribute(spec)
                if ext.raw_key is not None:     # fitted on its raw column of the normal training videos (oapp PCA)
                    if not _saves_state(ext):
                        raise TypeError(f"attribute {spec['name']!r} sets raw_key {ext.raw_key!r}, so it needs "
                                        "state() and from_state()")
                    ext.fit([train[v][ext.raw_key] for v in names], rng)        # rng shared (objscore2)
                vr = {v: VideoRows(*record_rows(ext, train[v]), int(train[v]["n_frames"]), int(train[v]["startup"]))
                      for v in names}
                fit = fit_attribute(vr, {}, rng, default_device(self.device), every, drop, self.density["args"])
                if fit is not None:                     # None: too few training rows, the attribute is skipped
                    cal = build("calibrator", cfg.get("calibrator", "fisher")).fit(fit.ref)
                    out[g][spec["name"]] = AttributeState(ext, fit.density, cal, fit.ref, fit.floor, fit.hyper)
        self.state = FittedState(cfg, out, self.scenes, gaps.pop())
        return self.state

    def score(self, test: dict[str, dict], state: FittedState | None = None,
              channels: list[str] | None = None) -> dict[str, dict[str, np.ndarray]]:
        """{channel: {video: max over objects of -log p, floor without}}; channels default: state.channels."""
        state = state or self.state
        drop = bool(state.config.get("drop_startup", False))
        out = {}
        for a in state.channels if channels is None else channels:
            out[a] = {}
            for v in sorted(test):
                r, st = test[v], state.groups[state.group_of(v)][a]
                vr = VideoRows(*record_rows(st.extractor, r), int(r["n_frames"]), int(r["startup"]))
                s = st.density.frame_scores(*_rows(vr, False, 1, drop), vr.n_frames)
                out[a][v] = np.where(np.isfinite(s), s, st.floor)
        return out

    def calibrate(self, raw: dict[str, dict[str, np.ndarray]],
                  state: FittedState | None = None) -> dict[str, dict[str, np.ndarray]]:
        """Calibrated channels with the calibrator of each video's group."""
        state = state or self.state
        return {a: {v: state.groups[state.group_of(v)][a].calibrator(s) for v, s in d.items()} for a, d in raw.items()}

    def fuse(self, z: dict[str, dict[str, np.ndarray]]) -> tuple[str, dict[str, np.ndarray]]:
        """('+'-joined fused channel names, {video: fused score}) of the config's fusion (default every channel)."""
        fusion = build("fusion", self.config.get("fusion", "sum"))
        names = getattr(fusion, "channels", None) or list(z)
        return "+".join(names), {v: fusion({n: z[n][v] for n in z}) for v in next(iter(z.values()))}

    def evaluate(self, z: dict[str, dict[str, np.ndarray]], gt: dict, warmup: int | None = None) -> dict[str, dict]:
        """evaluate() of every channel and of the fused set with the config's temporal stage, keyed like the fuse.json
        of tools/objfuse.py."""
        warmup = int(self.config.get("warmup", 4)) if warmup is None else warmup
        smooth = offline_smoother(self.config.get("temporal", "causal_median"))
        vids = sorted(gt)
        res = {n: evaluate({v: z[n][v] for v in vids}, gt, warmup, smooth) for n in z}
        name, fused = self.fuse(z)
        if name not in res:
            res[name] = evaluate({v: fused[v] for v in vids}, gt, warmup, smooth)
        return res


def channel_order(grid: dict, obj: dict, scenes: bool = False) -> dict:
    """I, Fs, Fd, other grid, then object channels (tools/objfuse.py ARCH, then extras); per scene sorted with the
    grid names first (tools/objfuse_sht.py)."""
    if set(grid) & set(obj):
        raise ValueError(f"channels {sorted(set(grid) & set(obj))} come from both the grid files and the object branch")
    z, arch = {**grid, **obj}, list(GRID_NAMES.values())
    flat = lambda n: arch.index(n) if n in arch else len(arch) + (n in obj)       # stable sort keeps the given order
    return {n: z[n] for n in sorted(z, key=(lambda n: (n not in arch, n)) if scenes else flat)}
