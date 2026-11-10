"""Post-hoc diagnosis of the grid branch without the autoencoder residual channel I
(paper Section VI, Limitations; Supplementary Section S8).

    PYTHONPATH=. python tools/flow_grid_diag.py --dataset shanghaitech --features F --gt G \\
        --extra-channels <npz>:frx_bayes=I,flow_ubm_m=Fs,flow_rx_m=Fd --out flow_grid_posthoc.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from vadkit.eval.protocol import evaluate, paired_bootstrap
from vadkit.grid.precomputed import PrecomputedChannels
from vadkit.pipeline.offline import ObjectBranch, channel_order, load_config

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 0.5


def coverage(train: dict) -> float:
    n = k = 0
    for r in train.values():
        m = np.asarray(r["cls"]) == 0
        n += int(m.sum())
        k += int((np.asarray(r["pose_ok"])[m] > 0).sum())
    return k / max(n, 1)


def run_dataset(dataset: str, features: str, gt_path: str, extra_channels: list[str]) -> dict:
    scenes = dataset == "shanghaitech"
    cfg = load_config(ROOT / ("vadkit/configs/r2/base_sht.yaml" if scenes else "vadkit/configs/r2/base.yaml"))
    gt = json.loads(Path(gt_path).read_text())
    vids = sorted(gt)
    br = ObjectBranch(cfg, scenes, "cpu")
    train, test = br.load(features, ("training", "testing"))
    gate_on = coverage(train) >= THRESHOLD
    st = br.fit(train)
    z_obj = br.calibrate(br.score({v: test[v] for v in vids}, st), st)
    z_grid = PrecomputedChannels(extra_channels, cfg.get("calibrator", "fisher")).calibrated(vids, st.scenes)
    z_flow = {c: z_grid[c] for c in ("Fs", "Fd")}
    keep_obj = {c: z_obj[c] for c in z_obj if gate_on or c != "opose"}

    _, f_g0 = br.fuse(channel_order(z_grid, {}, scenes))
    _, f_g1 = br.fuse(channel_order(z_flow, {}, scenes))
    _, f_c0 = br.fuse(channel_order(z_grid, z_obj, scenes))
    _, f_c1 = br.fuse(channel_order(z_flow, z_obj, scenes))
    _, f_c3 = br.fuse(channel_order(z_flow, keep_obj, scenes))

    return {
        "gt_video_summary": {
            "total_videos": len(vids),
            "videos_ge_80pct_anomalous": sum(1 for v in vids if float(np.mean(gt[v])) >= 0.80),
            "videos_100pct_anomalous": sum(1 for v in vids if float(np.mean(gt[v])) == 1.0),
        },
        "channels": {c: evaluate({v: z_grid[c][v] for v in vids}, gt) for c in ("I", "Fs", "Fd")},
        "grid_only_full": evaluate(f_g0, gt),
        "grid_only_flow": evaluate(f_g1, gt),
        "obj_plus_full_grid": evaluate(f_c0, gt),
        "obj_plus_flow_grid": {
            "eval": evaluate(f_c1, gt),
            "boot_vs_full_grid": paired_bootstrap(f_c1, f_c0, gt, 4, 2000, 0)["difference"],
        },
        "pose_gate_plus_flow_grid": {
            "eval": evaluate(f_c3, gt),
            "boot_vs_full_grid": paired_bootstrap(f_c3, f_c0, gt, 4, 2000, 0)["difference"],
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--extra-channels", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out_path = Path(a.out)
    data = json.loads(out_path.read_text()) if out_path.exists() else {"datasets": {}}
    data.setdefault("datasets", {})[a.dataset] = run_dataset(a.dataset, a.features, a.gt, a.extra_channels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
