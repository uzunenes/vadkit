"""The single pre-registered test evaluation of HB-POT stage 1 (docs/tez_notlari/hiyerarsik_kalibrasyon_on_kayit_2026-11-05.md
§4-§5; gate committed before it in c91bfe4). One process per dataset, boot_grid.py pattern: the object state is
fitted once, the grid channels are calibrated by the arms T0 (fisher, the current method), T1 and T2 (fisher_hb) and
fused with the same object scores (sum, causal median 9, warmup 4).

Order: T0 first and its reproduction check (grid+obj and obj micro within 1e-4 of the pre-registered values); on a
failure the result is written with status T0_REPRODUCTION_FAILED and the process exits 2 before T1 / T2 are built.
Then: fisher_hb t0 must equal fisher bit for bit; T1, T2 (with one scene T2 must equal T1); micro / macro AUC of
obj, grid+obj, grid and each grid channel per arm; paired video bootstraps (2000 replicates, seed 0) of grid+obj
T2-T0 (H1), T2-T1 (H1b), T1-T0 (H4), T2/T1/T0 - obj (H2) and Fd alone T2-T0, T1-T0 (H3b); single-channel deltas
(H3a / H4 checks). --combine reads the per-dataset JSONs and writes the verdicts and the decision rule.

  python vad_bench/vast/runs_hb/boot_hb.py --dataset D --config C --features F --gt G [--scenes] \
      --extra-channels ... --extra-ref-ids ... --out results_D.json
  python vad_bench/vast/runs_hb/boot_hb.py --combine results_*.json --out results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))            # repo root (vadkit)

REPRO = {"shanghaitech": (0.7020, 0.7954), "ped1": (0.7641, 0.7086), "ped2": (0.9653, 0.9385),
         "avenue": (0.8738, 0.8278)}                                   # (grid+obj, obj) micro, binding doc §4
TOL = 1e-4
GRID = ("I", "Fs", "Fd")
ARMS = {"T0": "fisher", "T1": {"name": "fisher_hb", "args": {"mode": "t1"}},
        "T2": {"name": "fisher_hb", "args": {"mode": "t2"}}}
SINGLE = ("ped1", "ped2", "avenue")


def _jsonable(x):
    if isinstance(x, dict):
        return {("null" if k is None else str(k)): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def run(a) -> dict:
    from vadkit.calibration import hier
    from vadkit.eval.protocol import evaluate, paired_bootstrap
    from vadkit.grid.precomputed import PrecomputedChannels
    from vadkit.pipeline.offline import ObjectBranch, channel_order, load_config

    fits = []                                       # tail parameters of every fisher_hb fit, in call order
    orig = hier.HBGroupCalibrator.fit_groups

    def recording(self, refs, ids):
        out = orig(self, refs, ids)
        fits.append(self.fit_)
        return out

    hier.HBGroupCalibrator.fit_groups = recording

    import torch
    if torch.cuda.is_available():
        raise SystemExit("CPU only: run with CUDA_VISIBLE_DEVICES=''")
    t_start = time.time()
    log = lambda m: print(f"[{a.dataset} {time.time() - t_start:7.1f}s] {m}", flush=True)
    cfg = load_config(a.config)
    gcal = cfg.get("grid_calibrator", cfg.get("calibrator", "fisher"))
    if gcal != "fisher" or cfg.get("calibrator", "fisher") != "fisher":
        raise SystemExit(f"config calibrators must be fisher (T0), got {gcal!r}")
    gt = json.loads(Path(a.gt).read_text())
    vids = sorted(gt)
    warm = int(cfg.get("warmup", 4))
    branch = ObjectBranch(cfg, a.scenes, None)
    test = branch.load(a.features, ("testing",))[0]
    for v in vids:
        assert int(test[v]["n_frames"]) == len(gt[v]), v
    state = branch.fit(branch.load(a.features, ("training",))[0])
    log(f"object state fitted ({len(state.groups)} groups, channels {state.channels})")
    z_obj = branch.calibrate(branch.score({v: test[v] for v in vids}, state), state)
    fuse = lambda zg, zo: branch.fuse(channel_order(zg, zo, state.scenes))
    ev = lambda f: evaluate({v: f[v] for v in vids}, gt, warm)
    res = {"dataset": a.dataset, "config": a.config, "features": a.features, "gt": a.gt, "scenes": bool(a.scenes),
           "extra_channels": a.extra_channels, "extra_ref_ids": a.extra_ref_ids, "warmup": warm, "reps": a.reps,
           "seed": 0, "videos": len(vids), "device": "cpu", "torch_threads": torch.get_num_threads(),
           "preregistration": "docs/tez_notlari/hiyerarsik_kalibrasyon_on_kayit_2026-11-05.md"}

    # ---------------------------------------------------------------- T0 and its reproduction
    z = {"T0": PrecomputedChannels(a.extra_channels, ARMS["T0"]).calibrated(vids, state.scenes)}
    if tuple(z["T0"]) != GRID:
        raise SystemExit(f"grid channels {list(z['T0'])}, expected {GRID}")
    name_all, f_t0 = fuse(z["T0"], z_obj)
    name_obj, f_obj = fuse({}, z_obj)
    m_all, m_obj = ev(f_t0)["causal"], ev(f_obj)["causal"]
    want = REPRO[a.dataset]
    ok = abs(m_all - want[0]) <= TOL and abs(m_obj - want[1]) <= TOL
    res["t0_reproduction"] = {"fused": name_all, "grid+obj": m_all, "obj": m_obj, "preregistered": list(want),
                              "tolerance": TOL, "pass": bool(ok)}
    log(f"T0 reproduction: grid+obj {m_all:.6f} (want {want[0]}), obj {m_obj:.6f} (want {want[1]}) -> "
        f"{'OK' if ok else 'FAILED'}")
    if not ok:
        res["status"] = "T0_REPRODUCTION_FAILED"
        return res

    # ---------------------------------------------------------------- T1, T2
    z_hb0 = PrecomputedChannels(a.extra_channels, {"name": "fisher_hb", "args": {"mode": "t0"}},
                                a.extra_ref_ids).calibrated(vids, state.scenes)
    assert all(np.array_equal(z_hb0[c][v], z["T0"][c][v]) for c in GRID for v in vids), "fisher_hb t0 != fisher"
    fits.clear()
    params = {}
    for arm in ("T1", "T2"):
        n0 = len(fits)
        z[arm] = PrecomputedChannels(a.extra_channels, ARMS[arm], a.extra_ref_ids).calibrated(vids, state.scenes)
        assert tuple(z[arm]) == GRID and len(fits) - n0 == len(GRID)
        params[arm] = {c: {"mu": f["mu"], "tau": None if f["T"] is None else np.sqrt(np.diag(f["T"])),
                           "groups": {g: {k: p[k] for k in ("xi", "sigma", "theta_hat", "Sigma", "J", "u", "d",
                                                            "n_exc", "theta_star")} for g, p in f["groups"].items()}}
                       for c, f in zip(z[arm], fits[n0:])}
        log(f"{arm} calibrated")
    if not state.scenes:                                     # one scene: t2 is t1 by definition
        assert all(np.array_equal(z["T2"][c][v], z["T1"][c][v]) for c in GRID for v in vids), "single scene T2 != T1"
    res["tail_parameters"] = params

    # ---------------------------------------------------------------- AUCs
    f_all = {"T0": f_t0, **{arm: fuse(z[arm], z_obj)[1] for arm in ("T1", "T2")}}
    f_grid = {arm: fuse(z[arm], {})[1] for arm in ARMS}
    auc = {"obj": ev(f_obj)}
    for arm in ARMS:
        auc[arm] = {"grid+obj": ev(f_all[arm]), "grid": ev(f_grid[arm]), **{c: ev(z[arm][c]) for c in GRID}}
    res["auc"] = auc
    res["single_channel_delta"] = {f"{arm}-T0": {c: {"micro": auc[arm][c]["causal"] - auc["T0"][c]["causal"],
                                                     "macro": auc[arm][c]["causal_macro"] - auc["T0"][c]["causal_macro"]}
                                                 for c in GRID} for arm in ("T1", "T2")}
    log("AUCs: " + ", ".join(f"{arm} grid+obj {auc[arm]['grid+obj']['causal']:.4f}/{auc[arm]['grid+obj']['causal_macro']:.4f}"
                             for arm in ARMS) + f", obj {auc['obj']['causal']:.4f}/{auc['obj']['causal_macro']:.4f}")

    # ---------------------------------------------------------------- paired video bootstraps (2000, seed 0)
    pairs = {"grid+obj T2-T0 (H1)": (f_all["T2"], f_all["T0"]), "grid+obj T2-T1 (H1b)": (f_all["T2"], f_all["T1"]),
             "grid+obj T1-T0 (H4)": (f_all["T1"], f_all["T0"]), "grid+obj T2 - obj (H2)": (f_all["T2"], f_obj),
             "grid+obj T1 - obj": (f_all["T1"], f_obj), "grid+obj T0 - obj": (f_all["T0"], f_obj),
             "Fd T2-T0 (H3b)": (z["T2"]["Fd"], z["T0"]["Fd"]), "Fd T1-T0": (z["T1"]["Fd"], z["T0"]["Fd"])}
    boot = {}
    for k, (x, y) in pairs.items():
        boot[k] = paired_bootstrap(x, y, gt, warm, a.reps, 0)
        d = boot[k]["difference"]
        log(f"{k:24s} a {boot[k]['auc']['a']['point']:.4f} b {boot[k]['auc']['b']['point']:.4f} diff "
            f"{d['point']:+.4f} CI [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]")
    res["bootstrap"] = boot
    res["status"] = "OK"
    res["elapsed_s"] = time.time() - t_start
    return res


def combine(paths: list[str]) -> dict:
    """Verdicts of H1-H5 and the decision rule from the per-dataset results."""
    R = {}
    for p in paths:
        r = json.loads(Path(p).read_text())
        R[r["dataset"]] = r
    if any(r.get("status") != "OK" for r in R.values()) or set(R) != {"shanghaitech", *SINGLE}:
        return {"datasets": R, "status": "INCOMPLETE",
                "note": "a dataset is missing or failed (T0 reproduction); no verdicts"}
    sh = R["shanghaitech"]
    b = sh["bootstrap"]
    d = lambda k, r=sh: r["bootstrap"][k]["difference"]
    h1, h1b, h2 = d("grid+obj T2-T0 (H1)"), d("grid+obj T2-T1 (H1b)"), d("grid+obj T2 - obj (H2)")
    deltas = sh["single_channel_delta"]
    h3a = {f"{arm}/{c}": deltas[arm][c]["macro"] for arm in deltas for c in GRID}
    h3b = d("Fd T2-T0 (H3b)")
    h3c = deltas["T2-T0"]["I"]["micro"]
    h4 = {ds: d("grid+obj T1-T0 (H4)", R[ds]) for ds in SINGLE}
    h4_chk = {ds: {c: R[ds]["single_channel_delta"]["T1-T0"][c] for c in GRID} for ds in SINGLE}
    ped1_gain = d("grid+obj T1 - obj", R["ped1"])
    h5 = sh["auc"]["T2"]["grid+obj"]["causal_macro"]
    v = {
        "H1": {"rule": "ShT micro(obj+grid_T2) - micro(obj+grid_T0) > 0 and 95% CI lower bound > 0", **h1,
               "pass": bool(h1["point"] > 0 and h1["ci95"][0] > 0)},
        "H1b": {"rule": "ShT T2 - T1, as H1", **h1b, "pass": bool(h1b["point"] > 0 and h1b["ci95"][0] > 0)},
        "H2": {"rule": "ShT micro(obj+grid_T2) - micro(obj): weak CI upper >= 0, strong CI lower > 0", **h2,
               "weak": bool(h2["ci95"][1] >= 0), "strong": bool(h2["ci95"][0] > 0)},
        "H3a": {"rule": "ShT single-channel macro |delta| < 0.005 in every arm (else an implementation error)",
                "deltas": h3a, "pass": bool(all(abs(x) < 0.005 for x in h3a.values()))},
        "H3b": {"rule": "ShT Fd alone micro T2 - T0: 95% CI lower bound > 0", **h3b, "pass": bool(h3b["ci95"][0] > 0)},
        "H3c": {"rule": "prediction: ShT I alone micro |T2 - T0| < 0.02", "delta": h3c, "pass": bool(abs(h3c) < 0.02)},
        "H4": {"rule": "Ped1, Ped2, Avenue: micro(obj+grid_T1) - micro(obj+grid_T0) >= -0.01 (point)",
               "per_dataset": {ds: {**x, "pass": bool(x["point"] >= -0.01)} for ds, x in h4.items()},
               "pass": bool(all(x["point"] >= -0.01 for x in h4.values()))},
        "H4_checks": {"rule": "descriptive (TASLAK §3.6): single-channel micro and macro |delta T1-T0| < 0.005; "
                              "Ped1 grid gain over obj (T1) CI lower bound > 0",
                      "single_channel": h4_chk,
                      "single_channel_pass": bool(all(abs(x[m]) < 0.005 for ds in h4_chk for x in h4_chk[ds].values()
                                                      for m in ("micro", "macro"))),
                      "ped1_grid_gain_T1": {**ped1_gain, "pass": bool(ped1_gain["ci95"][0] > 0)}},
        "H5": {"rule": "ShT macro(obj+grid_T2) >= 0.8125 (point)", "macro": h5, "pass": bool(h5 >= 0.8125)},
    }
    adopt = v["H1"]["pass"] and v["H4"]["pass"]
    v["decision"] = {"rule": "T2 becomes the grid calibrator only if H1 and H4 both hold; otherwise T0 stays and "
                             "stage 1 is reported as a negative result",
                     "adopt_T2": bool(adopt), "grid_calibrator": "T2 (fisher_hb)" if adopt else "T0 (fisher)"}
    implementation_checks = {"H3a_pass": v["H3a"]["pass"], "H4_single_channel_pass": v["H4_checks"]["single_channel_pass"]}
    return {"status": "OK", "verdicts": v, "implementation_checks": implementation_checks, "datasets": R}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--combine", nargs="+", help="per-dataset result JSONs -> verdicts")
    p.add_argument("--dataset", choices=sorted(REPRO))
    p.add_argument("--config")
    p.add_argument("--features")
    p.add_argument("--gt")
    p.add_argument("--scenes", action="store_true")
    p.add_argument("--extra-channels", nargs="+")
    p.add_argument("--extra-ref-ids", nargs="+")
    p.add_argument("--reps", type=int, default=2000)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.combine:
        res = combine(a.combine)
    else:
        if a.scenes != (a.dataset == "shanghaitech"):
            raise SystemExit("--scenes is for shanghaitech only (and required there)")
        res = run(a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(_jsonable(res), indent=1))
    print(f"wrote {a.out} (status {res.get('status')})", flush=True)
    if res.get("status") != "OK":
        sys.exit(2)
    if a.combine:
        for k, x in res["verdicts"].items():
            print(f"{k}: " + json.dumps({kk: vv for kk, vv in x.items() if kk != "rule"}), flush=True)


if __name__ == "__main__":
    main()
