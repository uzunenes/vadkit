"""Archived grid-branch (cell) channels from scores.npz files, calibrated against their own normal reference so fusion
can include the grid branch now (tools/objfuse.py channels / extra_channels, tools/objfuse_sht.py per scene)."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from vadkit.core.registry import available, build
from vadkit.density.niw import scene_of

GRID_NAMES = {"frx_bayes": "I", "flow_ubm_m": "Fs", "flow_rx_m": "Fd"}     # tools/objfuse.py ARCH keys -> channel


def parse_spec(spec: str) -> tuple[str, dict[str, str] | None]:
    """'path[:key=name,...]' -> (path, {key: channel} or None for every 'ref/<key>' of the file)."""
    path, sep, sel = spec.rpartition(":")
    if not sep or "=" not in sel or "/" in sel:
        return spec, None
    return path, dict(kv.split("=", 1) for kv in sel.split(","))


def is_group_calibrator(spec: str | dict) -> bool:
    """True for a registry 'group_calibrator' (fitted on the references of every group of a channel at once)."""
    return (spec if isinstance(spec, str) else spec["name"]) in available("group_calibrator")


class PrecomputedChannels:
    """scores.npz channels ('<key>/<video>', 'ref/<key>'), named GRID_NAMES.get(key, key); '{sc}' in a path = scene.

    calibrator: a 'calibrator' spec (fitted per group on the group's reference) or a 'group_calibrator' spec (e.g.
    fisher_hb, vadkit/calibration/hier.py: fitted on the references of every group of a channel together), which also
    needs ref_ids: npz templates (one, or one per spec) with 'ids/<key>', the training video index of every value of
    'ref/<key>' (tools/grid_ref_ids.py); sizes must match, nothing falls back silently."""

    def __init__(self, specs: list[str], calibrator: str | dict = "fisher", ref_ids: list[str] | None = None):
        self.sources = [parse_spec(s) for s in specs]
        self.calibrator = calibrator
        self.group = is_group_calibrator(calibrator)
        ref_ids = list(ref_ids or [])
        if ref_ids and len(ref_ids) not in (1, len(specs)):
            raise ValueError(f"{len(ref_ids)} ref-id files for {len(specs)} score files (give one, or one per file)")
        self.ref_ids = (ref_ids * len(specs) if len(ref_ids) == 1 else ref_ids) or None
        if self.group and self.ref_ids is None:
            raise ValueError(f"group calibrator {calibrator!r} needs ref_ids (--extra-ref-ids): the video id of every "
                             "reference value")

    def _load(self, vids: list[str], scene: str | None) -> dict[str, tuple[dict, np.ndarray, np.ndarray | None]]:
        """{channel: ({video: raw frame scores}, normal reference, video ids of the reference or None)}."""
        ch = {}
        for i, (path, sel) in enumerate(self.sources):
            p = Path(path.format(sc=scene)) if scene is not None else Path(path)
            if scene is not None and not p.exists():
                continue
            ids_file = None
            if self.ref_ids is not None:
                ids_file = Path(self.ref_ids[i].format(sc=scene)) if scene is not None else Path(self.ref_ids[i])
            with np.load(p) as S:
                keys = sel or {k[4:]: GRID_NAMES.get(k[4:], k[4:]) for k in S.files if k.startswith("ref/")}
                for key, name in keys.items():
                    if name in ch:
                        raise ValueError(f"channel {name!r} given twice; select keys with 'path:key=name'")
                    ref, ids = S[f"ref/{key}"], None
                    if ids_file is not None:
                        with np.load(ids_file) as I:
                            if f"ids/{key}" not in I.files:
                                raise KeyError(f"{ids_file}: no 'ids/{key}' for {p}")
                            ids = I[f"ids/{key}"]
                        if ids.shape != ref.shape:
                            raise ValueError(f"{ids_file} 'ids/{key}' {ids.shape} != {p} 'ref/{key}' {ref.shape}")
                    ch[name] = ({v: S[f"{key}/{v}"] for v in vids}, ref, ids)
        return ch

    def channels(self, vids: list[str], scene: str | None = None) -> dict[str, tuple[dict, np.ndarray]]:
        """{channel: ({video: raw frame scores}, normal reference)}; missing scene files are skipped (objfuse_sht)."""
        return {n: (s, ref) for n, (s, ref, _) in self._load(vids, scene).items()}

    def calibrated(self, vids: list[str], scenes: bool = False) -> dict[str, dict[str, np.ndarray]]:
        """{channel: {video: calibrated score}}; with scenes, calibrated per scene, channels of every scene only; a
        template without a file for any scene raises, one missing some scenes warns. Two passes: every group's
        channels are loaded first, then each channel is calibrated (per group, or by the group calibrator over all
        groups at once)."""
        groups = {}
        for v in vids:
            groups.setdefault(scene_of(v) if scenes else None, []).append(v)
        for path, _ in self.sources if scenes else ():
            miss = [g for g in groups if not Path(path.format(sc=g)).exists()]
            if len(miss) == len(groups):
                raise FileNotFoundError(f"{path}: no file for any scene")
            if miss:
                warnings.warn(f"{path}: no file for scenes {miss}, so its channels are left out (tools/objfuse_sht.py)")
        loaded = {g: self._load(vs, g) for g, vs in groups.items()}
        names = [n for n in next(iter(loaded.values()), {}) if all(n in d for d in loaded.values())]
        out = {}
        for n in names:
            if self.group:
                cals = build("group_calibrator", self.calibrator).fit_groups(
                    {g: d[n][1] for g, d in loaded.items()}, {g: d[n][2] for g, d in loaded.items()})
            else:
                cals = {g: build("calibrator", self.calibrator).fit(d[n][1]) for g, d in loaded.items()}
            out[n] = {v: cals[g](loaded[g][n][0][v]) for g, vs in groups.items() for v in vs}
        return out
