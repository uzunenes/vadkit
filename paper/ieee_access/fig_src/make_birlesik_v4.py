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
    put("VbCpu", env["cpu"].replace("(R)", "").replace("(TM)", "").replace(" 16-Core Processor", "")
        .replace(" 32-Core Processor", "").strip(),
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
    gs_mb = [sum(p.stat().st_size for p in (K / f"grid/state_{ds}").glob("*")) / 1e6 for ds in ("avenue", "ped2")]
    put("VbGridStateMBMin", f"{min(gs_mb):.2f}", "grid/state_{avenue,ped2}/grid.{npz,json} byte sum / 1e6")
    put("VbGridStateMBMax", f"{max(gs_mb):.2f}", "grid/state_{avenue,ped2}/grid.{npz,json} byte sum / 1e6")
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
    put("VbPedTwoGeEightyVids", str(fg["ped2"]["gt_video_summary"]["videos_ge_80pct_anomalous"]),
        f"{fg_src} ped2.gt_video_summary.videos_ge_80pct_anomalous")
    put("VbPedTwoAllAnomVids", str(fg["ped2"]["gt_video_summary"]["videos_100pct_anomalous"]),
        f"{fg_src} ped2.gt_video_summary.videos_100pct_anomalous")
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

    # Stage 2: directly measured setup & fitting timings on the single reference host
    t2_path = K / "stage2_timings.json"
    if t2_path.exists():
        t2 = json.loads(t2_path.read_text())
        t2_src = t2_path.relative_to(ROOT).as_posix()
        # TRT engine builds
        put("VbTrtBuildMin", f"{t2['export_trt']['total_trt_build_min']:.1f}", f"{t2_src} export_trt.total_trt_build_min")
        put("VbTrtBuildYoloS", f"{t2['export_trt']['yolo11n_engine_s']:.1f}", f"{t2_src} export_trt.yolo11n_engine_s")
        put("VbTrtBuildPoseS", f"{t2['export_trt']['yolo11n_pose_engine_s']:.1f}", f"{t2_src} export_trt.yolo11n_pose_engine_s")
        # Feature extraction timings (sequential PyTorch vs TensorRT + parallel flow)
        put("VbExtractPtPedTwoTrainMin", f"{t2['extract']['ped2_pt_train_min']:.1f}", f"{t2_src} extract.ped2_pt_train_min")
        put("VbExtractTrtPedTwoTrainMin", f"{t2['extract']['ped2_trt_train_min']:.1f}", f"{t2_src} extract.ped2_trt_train_min")
        put("VbExtractPtAvenueTrainMin", f"{t2['extract']['avenue_pt_train_min']:.1f}", f"{t2_src} extract.avenue_pt_train_min")
        put("VbExtractTrtAvenueTrainMin", f"{t2['extract']['avenue_trt_train_min']:.1f}", f"{t2_src} extract.avenue_trt_train_min")
        # Model fitting timings (GPU vs AMD EPYC 7452 CPU)
        put("VbFitGpuPedTwoS", f"{t2['fit']['ped2_gpu_s']:.1f}", f"{t2_src} fit.ped2_gpu_s")
        put("VbFitCpuPedTwoMin", f"{t2['fit']['ped2_cpu_min']:.1f}", f"{t2_src} fit.ped2_cpu_min")
        put("VbFitGpuAvenueS", f"{t2['fit']['avenue_gpu_s']:.1f}", f"{t2_src} fit.avenue_gpu_s")
        put("VbFitCpuAvenueMin", f"{t2['fit']['avenue_cpu_min']:.1f}", f"{t2_src} fit.avenue_cpu_min")
        # Total setup time on reference host (extraction + GPU fit)
        tot_pt_ped2 = t2['extract']['ped2_pt_train_min'] + t2['fit']['ped2_gpu_s'] / 60.0
        tot_pt_av = t2['extract']['avenue_pt_train_min'] + t2['fit']['avenue_gpu_s'] / 60.0
        tot_trt_ped2 = t2['extract']['ped2_trt_train_min'] + t2['fit']['ped2_gpu_s'] / 60.0
        tot_trt_av = t2['extract']['avenue_trt_train_min'] + t2['fit']['avenue_gpu_s'] / 60.0
        put("VbSetupPtPedTwoMin", f"{tot_pt_ped2:.1f}", "extract.ped2_pt_train_min + fit.ped2_gpu_s")
        put("VbSetupPtAvenueMin", f"{tot_pt_av:.1f}", "extract.avenue_pt_train_min + fit.avenue_gpu_s")
        put("VbSetupTrtPedTwoMin", f"{tot_trt_ped2:.1f}", "extract.ped2_trt_train_min + fit.ped2_gpu_s")
        put("VbSetupTrtAvenueMin", f"{tot_trt_av:.1f}", "extract.avenue_trt_train_min + fit.avenue_gpu_s")
        sp_ped2 = t2["extract"]["ped2_pt_train_s"] / t2["extract"]["ped2_trt_train_s"]
        sp_av = t2["extract"]["avenue_pt_train_s"] / t2["extract"]["avenue_trt_train_s"]
        put("VbExtractSpeedupPedTwo", f"{sp_ped2:.2f}", f"{t2_src} extract.ped2_pt_train_s / ped2_trt_train_s")
        put("VbExtractSpeedupAvenue", f"{sp_av:.2f}", f"{t2_src} extract.avenue_pt_train_s / avenue_trt_train_s")

    # Stage 3: Prior regularization & Cross-scene transfer evidence
    s3_path = ROOT / "kanit/birlesik_20261108/stage3_prior_dataeff.json"
    if s3_path.exists():
        s3 = json.loads(s3_path.read_text())
        s3_src = s3_path.relative_to(ROOT).as_posix()
        # SHT scene overlap audit
        put("VbShtTotalScenes", str(s3["sht_scene_overlap"]["total_scenes"]), f"{s3_src} sht_scene_overlap.total_scenes")
        put("VbShtOverlapScenes", str(s3["sht_scene_overlap"]["n_overlap_scenes"]), f"{s3_src} sht_scene_overlap.n_overlap_scenes")
        put("VbShtOverlapTrainVids", str(s3["sht_scene_overlap"]["overlap_train_videos"]), f"{s3_src} sht_scene_overlap.overlap_train_videos")
        put("VbShtTotalTestVids", str(s3["sht_scene_overlap"]["total_test_videos"]), f"{s3_src} sht_scene_overlap.total_test_videos")
        sht_de = json.loads((ROOT / "kanit/kurulum_20261105/dataeff_shanghaitech_1.0.json").read_text())
        put("VbShtTrainMin", f"{sht_de['rows'][0]['minutes']:.0f}",
            "kanit/kurulum_20261105/dataeff_shanghaitech_1.0.json rows[0].minutes")
        put("VbShtSubsampleSec", f"{5.0 / 24.0:.1f}", "5 frames / 24 FPS on ShanghaiTech")
        # f=0.10 NIW vs ML-GMM gain per dataset
        f010 = s3["f010_ablation"]
        put("VbFZeroTenNiwPedOne", f"{100 * f010['ped1']['summary']['D1']['causal_mean']:.1f}", f"{s3_src} ped1.D1.causal_mean")
        put("VbFZeroTenGmmPedOne", f"{100 * f010['ped1']['summary']['D2']['causal_mean']:.1f}", f"{s3_src} ped1.D2.causal_mean")
        put("VbFZeroTenGainPedOne", f"{100 * f010['ped1']['summary']['D1_minus_D2_causal_mean']:.1f}", f"{s3_src} ped1.D1_minus_D2")
        put("VbFZeroTenNiwPedTwo", f"{100 * f010['ped2']['summary']['D1']['causal_mean']:.1f}", f"{s3_src} ped2.D1.causal_mean")
        put("VbFZeroTenGmmPedTwo", f"{100 * f010['ped2']['summary']['D2']['causal_mean']:.1f}", f"{s3_src} ped2.D2.causal_mean")
        put("VbFZeroTenGainPedTwo", f"{100 * f010['ped2']['summary']['D1_minus_D2_causal_mean']:.1f}", f"{s3_src} ped2.D1_minus_D2")
        put("VbFZeroTenNiwAvenue", f"{100 * f010['avenue']['summary']['D1']['causal_mean']:.1f}", f"{s3_src} avenue.D1.causal_mean")
        put("VbFZeroTenGmmAvenue", f"{100 * f010['avenue']['summary']['D2']['causal_mean']:.1f}", f"{s3_src} avenue.D2.causal_mean")
        put("VbFZeroTenGainAvenue", f"{100 * f010['avenue']['summary']['D1_minus_D2_causal_mean']:.1f}", f"{s3_src} avenue.D1_minus_D2")
        put("VbFZeroTenNiwSHT", f"{100 * f010['shanghaitech']['summary']['D1']['causal_mean']:.1f}", f"{s3_src} sht.D1.causal_mean")
        put("VbFZeroTenGmmSHT", f"{100 * f010['shanghaitech']['summary']['D2']['causal_mean']:.1f}", f"{s3_src} sht.D2.causal_mean")
        put("VbFZeroTenGainSHT", f"{100 * f010['shanghaitech']['summary']['D1_minus_D2_causal_mean']:.1f}", f"{s3_src} sht.D1_minus_D2")
        # Leave-one-scene-out transfer speedup and sparse data gain
        loto = s3["sht_cross_scene_loo"]
        put("VbLotoFitSpeedup", f"{loto['full_data']['speedup_factor']:.1f}", f"{s3_src} full_data.speedup_factor")
        put("VbLotoFullScratch", f"{100 * loto['full_data']['scene_cv']['causal']:.1f}", f"{s3_src} full_data.scene_cv.causal")
        put("VbLotoFullTrans", f"{100 * loto['full_data']['loo_hyper']['causal']:.1f}", f"{s3_src} full_data.loo_hyper.causal")
        put("VbLotoFullDiff", f"{100 * abs(loto['full_data']['boot_loo_vs_cv']['point']):.1f}",
            f"{s3_src} |full_data.boot_loo_vs_cv.point|")
        lci = loto["full_data"]["boot_loo_vs_cv"]["ci95"]
        put("VbLotoFullCI", f"[{sgn(lci[0])}, {sgn(lci[1])}]", f"{s3_src} full_data.boot_loo_vs_cv.ci95")
        f010_diff = loto["fractions"]["0.1"]["loo_prior_blend_causal_mean"] - loto["fractions"]["0.1"]["scene_cv_causal_mean"]
        f025_diff = loto["fractions"]["0.25"]["loo_hyper_causal_mean"] - loto["fractions"]["0.25"]["scene_cv_causal_mean"]
        put("VbLotoFZeroTenGain", f"{100 * f010_diff:.2f}", f"{s3_src} fractions.0.1 loo_prior_blend vs scene_cv")
        put("VbLotoFTwoFiveGain", f"{100 * f025_diff:.2f}", f"{s3_src} fractions.0.25 loo_hyper vs scene_cv")

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
