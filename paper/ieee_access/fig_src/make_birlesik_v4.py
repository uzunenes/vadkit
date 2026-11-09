"""Numbers and the supplementary table of the combined streaming measurement
(docs/tez_notlari/birlesik_hiz_on_kayit_2026-11-08.md) from kanit/birlesik_20261108/vast_out:
out/bench/<ds>_<backend>_<mode>_<i>.json (tools/bench_e2e.py), out/grid/check_<ds>.json (tools/grid_check.py),
out/stream/avenue/result.json (tools/stream_eval.py).

    python3 fig_src/make_birlesik_v4.py   # gen/numbers_birlesik_v4.tex, gen/supp_tab_birlesik_v4.tex, provenance
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
K = ROOT / "kanit/birlesik_20261108/vast_out/out"
GEN = ROOT / "gen"
DS = {"ped2": "PedTwo", "avenue": "Avenue"}
BE = {"trt": "Trt", "pt": "Pt"}
MODE = {"obj": "Obj", "grid": "Grid"}
MACROS, PROV = [], []


def put(name: str, value: str, src: str) -> None:
    MACROS.append((name, value))
    PROV.append(f"\\{name} = {value}    <- {src}")


def bench(ds: str, be: str, mode: str) -> dict:
    """The three runs of one cell: median ms over all 300 x 3 frames, p99, FPS = 1000 / median of the run medians."""
    files = sorted(K.glob(f"bench/{ds}_{be}_{mode}_*.json"))
    if len(files) != 3:
        raise SystemExit(f"{ds} {be} {mode}: {len(files)} runs, the pre-registration has 3")
    runs = [json.loads(f.read_text()) for f in files]
    for r in runs:
        if r["warmup_frames"] != 20 or r["frames_per_rep"] != 100 or len(r["reps"]) != 3 or not r["parallel_flow"]:
            raise SystemExit(f"{ds} {be} {mode}: protocol differs from the pre-registration")
        if (mode == "grid") != (r["mode"] == "object+grid"):
            raise SystemExit(f"{ds} {be} {mode}: mode {r['mode']}")
    frames = np.concatenate([r["timed_frames"]["total_ms"] for r in runs])
    rep_med = [rep["total"]["median_ms"] for r in runs for rep in r["reps"]]
    run_fps = [r["fps_median"] for r in runs]
    return {"ms50": float(np.median(frames)), "ms99": float(np.quantile(frames, 0.99)),
            "fps": 1000.0 / float(np.median(rep_med)), "fps_lo": min(run_fps), "fps_hi": max(run_fps),
            "idle": sum(r["idle_ok"] for r in runs), "n": len(runs),
            "load": [r["loadavg_before"][0] for r in runs], "env": runs[0]["env"],
            "files": [f.relative_to(ROOT).as_posix() for f in files]}


def main():
    cells = {(ds, be, m): bench(ds, be, m) for ds in DS for be in BE for m in MODE}
    env = next(iter(cells.values()))["env"]
    put("VbGpu", env["gpu"].replace("NVIDIA GeForce ", ""), "bench env.gpu")
    put("VbCpu", env["cpu"].replace("(R)", "").replace("(TM)", "").replace(" 16-Core Processor", "").strip(),
        "bench env.cpu")
    for (ds, be, m), c in cells.items():
        tag = f"{MODE[m]}{BE[be]}{DS[ds]}"
        src = ", ".join(c["files"])
        put(f"VbFps{tag}", f"{c['fps']:.1f}", src)
        put(f"VbMs{tag}", f"{c['ms50']:.1f}", src)
        put(f"VbMsNN{tag}", f"{c['ms99']:.1f}", src)
    for ds in DS:
        for be in BE:
            d = cells[(ds, be, "grid")]["ms50"] - cells[(ds, be, "obj")]["ms50"]
            put(f"VbGridAddMs{BE[be]}{DS[ds]}", f"{d:.1f}", "difference of the medians above")
    idle = sum(c["idle"] for c in cells.values())
    total = sum(c["n"] for c in cells.values())
    put("VbIdleRuns", str(idle), "sum of idle_ok over the bench JSONs")
    put("VbRuns", str(total), "number of bench JSONs")
    loads = [x for c in cells.values() for x in c["load"]]
    put("VbLoadMax", f"{max(loads):.1f}", "max of loadavg_before[0]")
    # grid refit against the archive
    dauc = []
    for ds in DS:
        g = json.loads((K / f"grid/check_{ds}.json").read_text())
        for ch, r in g["channels"].items():
            dauc.append(abs(r["causal_auc_refit"] - r["causal_auc_archived"]))
        dauc.append(abs(g["grid_fused_causal_auc"]["refit"] - g["grid_fused_causal_auc"]["archived"]))
    put("VbGridDaucMax", f"{100 * max(dauc):.2f}", "grid/check_{ped2,avenue}.json, |refit - archived| causal AUC x 100")
    put("VbGridCheckOk", "yes" if max(dauc) <= 0.005 else "no", "pre-registered tolerance 0.005")
    s = json.loads((K / "stream/avenue/result.json").read_text())
    put("VbStreamObjAvenue", f"{100 * s['object']['causal']:.1f}", "stream/avenue/result.json object.causal")
    put("VbStreamGridAvenue", f"{100 * s['object+grid']['causal']:.1f}", "stream/avenue/result.json object+grid.causal")

    fg_path = ROOT / "kanit/birlesik_20261108/flow_grid_posthoc.json"
    fg = json.loads(fg_path.read_text())["datasets"]
    fg_src = fg_path.relative_to(ROOT).as_posix()

    def sgn(x: float) -> str:
        v = 100.0 * x
        return ("$-$" if v < 0 else "$+$") + f"{abs(v):.1f}"

    put("VbShtFsOn", f"{100 * fg['shanghaitech']['channels']['Fs']['causal']:.1f}", f"{fg_src} shanghaitech.channels.Fs.causal")
    put("VbShtFdOn", f"{100 * fg['shanghaitech']['channels']['Fd']['causal']:.1f}", f"{fg_src} shanghaitech.channels.Fd.causal")
    for ds, tag in (("ped1", "PedOne"), ("ped2", "PedTwo"), ("avenue", "Avenue"), ("shanghaitech", "SHT")):
        d = fg[ds]
        put(f"VbGridOnlyFullOn{tag}", f"{100 * d['grid_only_full']['causal']:.1f}", f"{fg_src} {ds}.grid_only_full.causal")
        put(f"VbGridOnlyFlowOn{tag}", f"{100 * d['grid_only_flow']['causal']:.1f}", f"{fg_src} {ds}.grid_only_flow.causal")
        c1 = d["obj_plus_flow_grid"]
        b1 = c1["boot_vs_full_grid"]
        put(f"VbFlowGridOn{tag}", f"{100 * c1['eval']['causal']:.1f}", f"{fg_src} {ds}.obj_plus_flow_grid.eval.causal")
        put(f"VbFlowGridStd{tag}", f"{100 * c1['eval']['offline']:.1f}", f"{fg_src} {ds}.obj_plus_flow_grid.eval.offline")
        put(f"VbFlowGridMacro{tag}", f"{100 * c1['eval']['causal_macro']:.1f}", f"{fg_src} {ds}.obj_plus_flow_grid.eval.causal_macro")
        put(f"VbFlowGridDiff{tag}", sgn(b1["point"]), f"{fg_src} {ds}.obj_plus_flow_grid.boot_vs_full_grid.point")
        put(f"VbFlowGridCI{tag}", f"[{sgn(b1['ci95'][0])}, {sgn(b1['ci95'][1])}]", f"{fg_src} {ds}.obj_plus_flow_grid.boot_vs_full_grid.ci95")
        c3 = d["pose_gate_plus_flow_grid"]
        b3 = c3["boot_vs_full_grid"]
        put(f"VbRuleFlowGridOn{tag}", f"{100 * c3['eval']['causal']:.1f}", f"{fg_src} {ds}.pose_gate_plus_flow_grid.eval.causal")
        put(f"VbRuleFlowGridDiff{tag}", sgn(b3["point"]), f"{fg_src} {ds}.pose_gate_plus_flow_grid.boot_vs_full_grid.point")
        put(f"VbRuleFlowGridCI{tag}", f"[{sgn(b3['ci95'][0])}, {sgn(b3['ci95'][1])}]", f"{fg_src} {ds}.pose_gate_plus_flow_grid.boot_vs_full_grid.ci95")

    # supplementary table
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{End-to-end latency per frame of the combined streaming detector, from reading the JPEG to the "
         r"fused online score, batch size one, in one session (" + MACROS[0][1] + ", " + MACROS[1][1] + r"); "
         r"object-branch flow in a CPU thread, grid branch in a second thread with its own DIS flow. Median and 99th "
         r"percentile over $3\times300$ frames; FPS: 1000 / median of the per-repetition medians (range over the "
         r"three runs). Idle: runs that started on an idle GPU (pre-registered rule).}",
         r"\label{stab:combined}", r"\small", r"\begin{tabular}{llrrrrc}", r"\toprule",
         r"Dataset & Configuration & Back-end & Median (ms) & p99 (ms) & FPS (range) & Idle \\", r"\midrule"]
    for ds, dn in (("avenue", "Avenue"), ("ped2", "Ped2")):
        for be, bn in (("trt", "TensorRT"), ("pt", "PyTorch")):
            for m, mn in (("obj", "Object"), ("grid", "Object + grid")):
                c = cells[(ds, be, m)]
                L.append(f"{dn} & {mn} & {bn} & {c['ms50']:.1f} & {c['ms99']:.1f} & {c['fps']:.1f} "
                         f"({c['fps_lo']:.1f}--{c['fps_hi']:.1f}) & {c['idle']}/{c['n']} \\\\")
        if ds == "avenue":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (GEN / "supp_tab_birlesik_v4.tex").write_text("% generated by fig_src/make_birlesik_v4.py\n" + "\n".join(L) + "\n")
    (GEN / "numbers_birlesik_v4.tex").write_text(
        "% generated by fig_src/make_birlesik_v4.py from kanit/birlesik_20261108/vast_out\n"
        + "".join(f"\\newcommand{{\\{n}}}{{{v}}}\n" for n, v in MACROS))
    (GEN / "birlesik_v4_provenance.txt").write_text("\n".join(PROV) + "\n")
    print("\n".join(PROV))


if __name__ == "__main__":
    main()
