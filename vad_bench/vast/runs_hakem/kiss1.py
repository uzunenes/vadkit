"""KISS-1 (docs/tez_notlari/kiss1_hiz_buyuklugu_on_kayit_2026-11-06.md): a fourth object channel 'omag' (mean L1 flow
magnitude, vadkit kind 'magnitude') appended to the base configuration. Same fit, calibration, fusion and protocol as
`python -m vadkit eval`; paired video bootstrap KISS-1 vs base for the object branch and for object + grid.

  PYTHONPATH=. python vad_bench/vast/runs_hakem/kiss1.py --dataset avenue --features F --gt G --out J \
      [--extra-channels <npz>:frx_bayes=I ...]
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from vadkit.eval.protocol import paired_bootstrap
from vadkit.grid.precomputed import PrecomputedChannels
from vadkit.pipeline.offline import ObjectBranch, channel_order, load_config

ROOT = Path(__file__).resolve().parents[3]
REF = ROOT / "paper/ieee_access/kanit/r2_20261105/res/base"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--extra-channels", nargs="*", default=[])
    a = p.parse_args()
    scenes = a.dataset == "shanghaitech"
    cfg_b = load_config(ROOT / ("vadkit/configs/r2/base_sht.yaml" if scenes else "vadkit/configs/r2/base.yaml"))
    cfg_k = copy.deepcopy(cfg_b)
    cfg_k["name"] = cfg_b["name"] + "_kiss1"
    cfg_k["attributes"] = cfg_b["attributes"] + [{"name": "omag", "kind": "magnitude"}]
    gt = json.loads(Path(a.gt).read_text())
    vids = sorted(gt)
    out = {"dataset": a.dataset, "config_base": cfg_b["name"], "config_kiss1": cfg_k["name"], "checks": {}}
    fused, res = {}, {}
    zgrid = None
    for tag, cfg in (("base", cfg_b), ("kiss1", cfg_k)):
        br = ObjectBranch(cfg, scenes, "cpu")
        train, test = br.load(a.features, ("training", "testing"))
        st = br.fit(train)
        z = br.calibrate(br.score({v: test[v] for v in vids}, st), st)
        out.setdefault("hyper", {})[tag] = {g: {c: {k: (float(v) if isinstance(v, float) else int(v))
                                                    for k, v in s.hyper.items()} for c, s in grp.items()}
                                           for g, grp in st.groups.items()}
        res[tag] = {"obj": br.evaluate(channel_order({}, z, scenes), gt)}
        fused[tag] = {"obj": br.fuse(channel_order({}, z, scenes))[1]}
        if a.extra_channels:
            if zgrid is None:
                zgrid = PrecomputedChannels(a.extra_channels, cfg.get("calibrator", "fisher")).calibrated(vids, st.scenes)
            res[tag]["grid"] = br.evaluate(channel_order(zgrid, z, scenes), gt)
            fused[tag]["grid"] = br.fuse(channel_order(zgrid, z, scenes))[1]
        if tag == "base":
            z_base = z
        else:
            out["checks"]["first_three_channels_identical"] = all(
                np.array_equal(z_base[c][v], z[c][v]) for c in z_base for v in vids)
    for kind in res["base"]:
        names = {k for k in res["base"][kind] if "+" in k}
        kb = max(names, key=lambda k: k.count("+"))
        arch = json.loads((REF / f"{a.dataset}_{kind}.json").read_text())
        ka = max((k for k in arch if "+" in k), key=lambda k: k.count("+"))
        d = abs(res["base"][kind][kb]["causal"] - arch[ka]["causal"])
        out["checks"][f"{kind}_reproduces_archived"] = {"ours": res["base"][kind][kb]["causal"],
                                                         "archived": arch[ka]["causal"], "abs_diff": d}
        if d > 1e-6:
            Path(a.out).write_text(json.dumps(out, indent=1))
            raise SystemExit(f"{a.dataset} {kind}: base does not reproduce the archived result ({d:.3g})")
    if not out["checks"]["first_three_channels_identical"]:
        Path(a.out).write_text(json.dumps(out, indent=1))
        raise SystemExit("the first three channels of KISS-1 differ from base")
    out["results"] = res
    out["boot"] = {kind: paired_bootstrap(fused["kiss1"][kind], fused["base"][kind], gt, 4, 2000, 0)
                   for kind in fused["base"]}
    for kind, b in out["boot"].items():
        d = b["difference"]
        print(f"{a.dataset} {kind}: kiss1 {b['auc']['a']['point']:.4f} base {b['auc']['b']['point']:.4f} "
              f"diff {d['point']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("KISS1_DONE", flush=True)


if __name__ == "__main__":
    main()
