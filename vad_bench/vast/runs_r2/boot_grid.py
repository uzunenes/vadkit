"""Paired video bootstrap of the grid branch (H2 / T3 of docs/tez_notlari/tur2_on_kayit_2026-11-05.md).

Same fit, calibration and fusion as `python -m vadkit eval`; compares, on the same resampled test videos,
  grid+obj : I + Fs + Fd + object channels   vs   obj : object channels alone,
and also reports the grid branch alone (I + Fs + Fd). Run from the repo root:
  python vad_bench/vast/runs_r2/boot_grid.py --config C --features F --gt G [--scenes] --extra-channels ... --out J
"""
import argparse
import json
from pathlib import Path

from vadkit.eval.protocol import paired_bootstrap
from vadkit.grid.precomputed import PrecomputedChannels
from vadkit.pipeline.offline import ObjectBranch, channel_order, load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--scenes", action="store_true")
    p.add_argument("--extra-channels", nargs="+", required=True)
    p.add_argument("--reps", type=int, default=2000)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cfg = load_config(a.config)
    gt = json.loads(Path(a.gt).read_text())
    vids = sorted(gt)
    branch = ObjectBranch(cfg, a.scenes, None)
    test = branch.load(a.features, ("testing",))[0]
    state = branch.fit(branch.load(a.features, ("training",))[0])
    z_grid = PrecomputedChannels(a.extra_channels, cfg.get("calibrator", "fisher")).calibrated(vids, state.scenes)
    z_obj = branch.calibrate(branch.score({v: test[v] for v in vids}, state), state)
    warm = int(cfg.get("warmup", 4))
    _, f_obj = branch.fuse(channel_order({}, z_obj, state.scenes))
    _, f_all = branch.fuse(channel_order(z_grid, z_obj, state.scenes))
    _, f_grid = branch.fuse(channel_order(z_grid, {}, state.scenes))
    res = {"grid+obj_vs_obj": paired_bootstrap(f_all, f_obj, gt, warm, a.reps),
           "grid_vs_obj": paired_bootstrap(f_grid, f_obj, gt, warm, a.reps)}
    for k, r in res.items():
        d = r["difference"]
        print(f"{k:16s} a {r['auc']['a']['point']:.4f}  b {r['auc']['b']['point']:.4f}  "
              f"diff {d['point']:+.4f} CI [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  P(>0) {d['fraction_positive']:.3f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
