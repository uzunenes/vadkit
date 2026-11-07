"""Training video id (and frame index) of every value of the archived grid normal references 'ref/<key>' (HB-POT,
docs/tez_notlari/hiyerarsik_kalibrasyon_on_kayit_2026-11-05.md; reconstruction of §3.3 of the TASLAK), from the
training frame counts; every size is asserted against the reference arrays. Run it on the blind copies of
tools/grid_refs_only.py ('ref/' arrays only).

  frame-rule keys (evaluate.py 1a', e.g. frx_bayes): sorted training videos, frames t = i_start, i_start + i_step, ...
      (range(T - 1, n, 3 * stats_every); T = 16, stats_every = 3 -> 15, 24, ...)
  fit_channel keys (aean/channels.py fit_channel, folds 2: flow_*, deep_*, dino_*): videos names[0::2] then
      names[1::2] of the sorted names, frames [::flow_every] (3)

Output per scene (or once without --scenes): npz with 'videos' (sorted names), 'lens', 'ids/<key>' (index into
videos) and 't/<key>' (frame index).

    python tools/grid_ref_ids.py --refs /workspace/hb/refs/shanghaitech/{sc}_mc5.npz \
        --frames /workspace/frames/shanghaitech/training/frames --scenes 01 ... 12 \
        --out /workspace/hb/refids/shanghaitech/{sc}.npz
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

KEYS = ("frx_bayes", "flow_ubm_m", "flow_rx_m")
FIT_CHANNEL = ("flow", "deep_", "dino_")


def video_lengths(frames: str | Path, scene: str | None = None) -> dict[str, int]:
    """{video: number of frames} of the sorted training videos (of the scene: name prefix before '_')."""
    names = sorted(os.listdir(frames))
    if scene is not None:
        names = [v for v in names if v.split("_")[0] == scene]
    if not names:
        raise FileNotFoundError(f"{frames}: no training videos" + (f" of scene {scene}" if scene else ""))
    return {v: len(os.listdir(os.path.join(frames, v))) for v in names}


def ref_ids(key: str, lens: dict[str, int], i_start: int = 15, i_step: int = 9,
            flow_every: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """(video index, frame index) of every reference value of key, in the order of 'ref/<key>'."""
    names = list(lens)
    if key.startswith(FIT_CHANNEL):
        order = list(range(0, len(names), 2)) + list(range(1, len(names), 2))
        t = [np.arange(0, lens[names[k]], flow_every) for k in order]
    else:
        order = list(range(len(names)))
        t = [np.arange(i_start, lens[names[k]], i_step) for k in order]
    ids = np.concatenate([np.full(len(tt), k, dtype=np.int32) for k, tt in zip(order, t)])
    return ids, np.concatenate(t).astype(np.int32)


def build(refs: list[str], frames: str, scene: str | None, keys=KEYS, **rule) -> dict[str, np.ndarray]:
    """The id arrays of one scene; asserts len(ids/<key>) == len(ref/<key>) for every key found in refs."""
    lens = video_lengths(frames, scene)
    out = {"videos": np.array(list(lens)), "lens": np.array(list(lens.values()), dtype=np.int64)}
    for path in refs:
        p = path.format(sc=scene) if scene is not None else path
        with np.load(p) as S:
            if any(not k.startswith("ref/") for k in S.files):
                raise ValueError(f"{p} holds non-'ref/' arrays; run tools/grid_refs_only.py first")
            for key in keys:
                if f"ref/{key}" not in S.files:
                    continue
                if f"ids/{key}" in out:
                    raise ValueError(f"'ref/{key}' found twice (scene {scene})")
                ids, t = ref_ids(key, lens, **rule)
                n = len(S[f"ref/{key}"])
                assert len(ids) == n, f"{p} ref/{key}: {n} values, the video lengths give {len(ids)}"
                out[f"ids/{key}"], out[f"t/{key}"] = ids, t
    missing = [k for k in keys if f"ids/{k}" not in out]
    if missing:
        raise KeyError(f"scene {scene}: no 'ref/' array for {missing} in {refs}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--refs", nargs="+", required=True, help="blind ref npz files (templates with '{sc}')")
    p.add_argument("--frames", required=True, help="<frames>/training/frames (one directory per video)")
    p.add_argument("--scenes", nargs="*", default=None, help="ShanghaiTech scenes (default: one group, all videos)")
    p.add_argument("--keys", nargs="+", default=list(KEYS))
    p.add_argument("--i-start", type=int, default=15, help="T - 1 of the AE run (args.json T = 16)")
    p.add_argument("--i-step", type=int, default=9, help="3 * stats_every (evaluate.py default 3)")
    p.add_argument("--flow-every", type=int, default=3, help="fit_channel every")
    p.add_argument("--out", required=True, help="output npz (template with '{sc}' when --scenes)")
    a = p.parse_args(argv)
    for sc in a.scenes or [None]:
        arr = build(a.refs, a.frames, sc, a.keys, i_start=a.i_start, i_step=a.i_step, flow_every=a.flow_every)
        out = Path(a.out.format(sc=sc) if sc is not None else a.out)
        if out.exists():
            raise FileExistsError(f"{out} exists; not overwritten")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **arr)
        print(f"{out}: {len(arr['videos'])} videos, " +
              ", ".join(f"{k[4:]} {len(v)}" for k, v in arr.items() if k.startswith("ids/")), flush=True)


if __name__ == "__main__":
    main()
