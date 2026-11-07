"""E1 pose coverage gate (docs/tez_notlari/e1_poz_kapsama_kapisi_on_kayit_2026-11-06.md).

Coverage c = person boxes with pose_ok > 0 / person boxes (cls == 0) on the normal training videos; the pose channel is
fused only if c >= 0.5. Checks (a) the gated object branch equals the S1 row (L-opose when gated off, D1 when on) and
the ungated rows equal the archived res/base results; reports (c) object + grid gated vs current with a paired video
bootstrap (2000 replicates, seed 0), and the same for the object branch.

  PYTHONPATH=. python vad_bench/vast/runs_hakem/e1.py --dataset ped1 --features F --gt G --s1 S1.json --out J \
      --extra-channels <npz>:frx_bayes=I ...
"""
import argparse
import json
from pathlib import Path

import numpy as np

from vadkit.eval.protocol import evaluate, paired_bootstrap
from vadkit.grid.precomputed import PrecomputedChannels
from vadkit.pipeline.offline import ObjectBranch, channel_order, load_config

ROOT = Path(__file__).resolve().parents[3]
REF = ROOT / "paper/ieee_access/kanit/r2_20261105/res/base"
THRESHOLD = 0.5


def coverage(train: dict) -> tuple[int, int, float]:
    n = k = 0
    for r in train.values():
        m = np.asarray(r["cls"]) == 0
        n += int(m.sum())
        k += int((np.asarray(r["pose_ok"])[m] > 0).sum())
    return n, k, k / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--s1", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--extra-channels", nargs="+", required=True)
    a = p.parse_args()
    scenes = a.dataset == "shanghaitech"
    cfg = load_config(ROOT / ("vadkit/configs/r2/base_sht.yaml" if scenes else "vadkit/configs/r2/base.yaml"))
    gt = json.loads(Path(a.gt).read_text())
    vids = sorted(gt)
    br = ObjectBranch(cfg, scenes, "cpu")
    train, test = br.load(a.features, ("training", "testing"))
    n, k, c = coverage(train)
    gate_on = c >= THRESHOLD
    st = br.fit(train)
    z = br.calibrate(br.score({v: test[v] for v in vids}, st), st)
    zg = PrecomputedChannels(a.extra_channels, cfg.get("calibrator", "fisher")).calibrated(vids, st.scenes)
    keep = {c_: z[c_] for c_ in z if gate_on or c_ != "opose"}
    fused = {"obj": br.fuse(channel_order({}, z, scenes))[1], "obj_gated": br.fuse(channel_order({}, keep, scenes))[1],
             "grid": br.fuse(channel_order(zg, z, scenes))[1], "grid_gated": br.fuse(channel_order(zg, keep, scenes))[1]}
    res = {k_: evaluate({v: s[v] for v in vids}, gt) for k_, s in fused.items()}
    out = {"dataset": a.dataset, "coverage": {"person_boxes": n, "with_pose": k, "c": c, "threshold": THRESHOLD,
                                              "gate_on": bool(gate_on)}, "results": res, "checks": {}}
    for kind, key in (("obj", "obj"), ("grid", "grid")):
        arch = json.loads((REF / f"{a.dataset}_{kind}.json").read_text())
        ka = max((x for x in arch if "+" in x), key=lambda x: x.count("+"))
        out["checks"][f"{kind}_reproduces_archived"] = abs(res[key]["causal"] - arch[ka]["causal"])
    s1 = json.loads(Path(a.s1).read_text())
    row = "D1" if gate_on else "L-opose"
    out["checks"]["gated_obj_equals_s1_" + row] = abs(res["obj_gated"]["causal"] - s1["rows"][row]["fused"]["causal"])
    bad = {k_: v for k_, v in out["checks"].items() if v > 1e-6}
    if bad:
        Path(a.out).write_text(json.dumps(out, indent=1))
        raise SystemExit(f"{a.dataset}: reproduction failed {bad}")
    out["boot"] = {"obj_gated_vs_obj": paired_bootstrap(fused["obj_gated"], fused["obj"], gt, 4, 2000, 0),
                   "grid_gated_vs_grid": paired_bootstrap(fused["grid_gated"], fused["grid"], gt, 4, 2000, 0)}
    d = res["grid_gated"]["causal"] - res["grid"]["causal"]
    out["guard_c"] = {"grid_gated_minus_grid": d, "holds": bool(d >= -0.005) if not gate_on else None}
    for k_, b in out["boot"].items():
        x = b["difference"]
        print(f"{a.dataset} c={c:.3f} gate={'on' if gate_on else 'off'} {k_}: {x['point']:+.4f} "
              f"[{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("E1_DONE", flush=True)


if __name__ == "__main__":
    main()
