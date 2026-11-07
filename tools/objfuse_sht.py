"""ShanghaiTech: per-scene Fisher calibration of every channel, then pooled frame AUC over all test videos.

    python tools/objfuse_sht.py --runs /workspace/checkpoints/final_2026-10-07/runs \
        --extra '/workspace/sonuclar/obj2/shanghaitech/{sc}/scores.npz' --out fuse.json

Archived channels per scene: I (AE residual FRX-Bayes), Fs, Fd from eval_shtech_<sc>_pool48_mc5. Extra channels: any
scores.npz (path template with {sc}) with <ch>/<video> and ref/<ch>; a channel absent for a scene is skipped for the
whole dataset. Scenes 01-12 (scene 13 has no test videos). Same protocols as tools/objfuse.py.
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuse_runs import normaliser  # noqa: E402
from objfuse import evaluate, extra_channels  # noqa: E402

ARCH = {"I": "frx_bayes", "Fs": "flow_ubm_m", "Fd": "flow_rx_m"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--extra", nargs="*", default=[], help="path templates with {sc}")
    ap.add_argument("--cache", type=Path, default=Path("/workspace/cache"))
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--max-combo", type=int, default=6)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    z, gt_all = {}, {}
    scenes = [f"{i:02d}" for i in range(1, 13)]
    per_scene = {}
    for sc in scenes:
        gt = json.loads((a.cache / f"shtech_{sc}" / "gt.json").read_text())
        vids = sorted(gt)
        gt_all.update(gt)
        S = np.load(a.runs / f"shtech_{sc}_2d_noadv" / f"eval_shtech_{sc}_pool48_mc5" / "scores.npz")
        ch = {n: ({v: S[f"{k}/{v}"] for v in vids}, S[f"ref/{k}"]) for n, k in ARCH.items()}
        ch.update(extra_channels([Path(t.format(sc=sc)) for t in a.extra if Path(t.format(sc=sc)).exists()], vids))
        per_scene[sc] = {n: {v: normaliser(ref, "fisher")(s[v]) for v in vids} for n, (s, ref) in ch.items()}
    names = sorted(set.intersection(*[set(d) for d in per_scene.values()]), key=lambda n: (n not in ARCH, n))
    if a.only:
        names = [n for n in names if n in a.only]
    for n in names:
        z[n] = {v: s for sc in scenes for v, s in per_scene[sc][n].items()}
    res = {}
    for r in range(1, min(len(names), a.max_combo) + 1):
        for combo in combinations(names, r):
            fused = {v: sum(z[n][v] for n in combo) for v in gt_all}
            res["+".join(combo)] = evaluate(fused, gt_all, a.warmup)
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]["causal"])[:40]:
        print(f"{k:28s} causal {v['causal']:.4f}  macro {v['causal_macro']:.4f}  offline {v['offline']:.4f}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
