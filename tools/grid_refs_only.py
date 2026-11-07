"""Blind copy of the normal references of archived grid scores (HB-POT gate,
docs/tez_notlari/hiyerarsik_kalibrasyon_on_kayit_2026-11-05.md §3): writes only the 'ref/<key>' arrays of each
scores.npz to a new file, so the normal-data gate never opens test scores or 'gt/' arrays. Only the archive member
names are listed; no other array is read. '{sc}' in SRC / DST is the scene (--scenes).

    python tools/grid_refs_only.py --src RUNS/shtech_{sc}_2d_noadv/eval_shtech_{sc}_pool48_mc5/scores.npz \
        --dst /workspace/hb/refs/shanghaitech/{sc}_mc5.npz --scenes 01 02 ... 12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def copy_refs(src: str | Path, dst: str | Path) -> dict[str, tuple]:
    """Write the 'ref/<key>' arrays of src to dst (a new file; an existing one raises); returns {key: shape}."""
    dst = Path(dst)
    if dst.exists():
        raise FileExistsError(f"{dst} exists; not overwritten")
    with np.load(src) as S:
        refs = {k: S[k] for k in S.files if k.startswith("ref/")}         # S.files lists names, reads no array
    if not refs:
        raise ValueError(f"{src}: no 'ref/' arrays")
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **refs)
    return {k: v.shape for k, v in refs.items()}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="scores.npz (template with '{sc}' when --scenes)")
    p.add_argument("--dst", required=True, help="output npz (template with '{sc}' when --scenes)")
    p.add_argument("--scenes", nargs="*", default=None)
    a = p.parse_args(argv)
    for sc in a.scenes or [None]:
        src = a.src.format(sc=sc) if sc is not None else a.src
        dst = a.dst.format(sc=sc) if sc is not None else a.dst
        shapes = copy_refs(src, dst)
        print(f"{src} -> {dst}: " + ", ".join(f"{k} {s}" for k, s in shapes.items()), flush=True)


if __name__ == "__main__":
    main()
