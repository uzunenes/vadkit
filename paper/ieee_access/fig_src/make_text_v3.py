#!/usr/bin/env python3
"""Write the numbers used in the running text of main_v3.tex that are not already in gen/numbers_v3.tex, the
ablation table gen/tab_ablation_v3.tex, and the provenance list gen/text_v3_provenance.txt.

    python3 paper/ieee_access/fig_src/make_text_v3.py

Run it after make_tables_v3.py (it reads gen/tables_v3_provenance.txt for two values that script derives).
Standard library only; every number is read from kanit/ (or gen/) and a missing input stops the script.

Sources:
  grid-branch bootstrap     kanit/r2_20261105/boot/<ds>.json            ('grid+obj_vs_obj', paired video bootstrap)
  component study           kanit/r2_20261105/res/<cfg>/<ds>_obj.json   (fused 'causal' = online micro AUC)
                            kanit/r2_20261105/bench/<cfg>.json          (median of the per-repetition median totals)
  per-channel AUCs          kanit/r2_20261105/res/base/<ds>_{obj,grid}.json
  HB-POT stage 1            kanit/hb_20261105/{results,gate}.json
  label-free threshold      kanit/kurulum_20261105/far_<ds>.json        ('empirical' / 'smoothed')
  data efficiency           kanit/kurulum_20261105/dataeff_<ds>_<f>.json
  dataset sizes             kanit/r2_20261105/boot/<ds>.json ('videos'), kanit/kurulum_20261105/far_<ds>.json
  Avenue labels             kanit/avenue_etiket_20261105/diff.json
  online vs standard        kanit/r2_20261105/res/base/<ds>_{obj,grid}.json (fused 'causal' - 'offline')
  competitor AUCs (gaps)    kanit/literature/dogrulanmis_literatur_2026-11-04.json (sota list, as make_tables_v3.py);
                            blocks from fig_src/tables_v3_inputs.json, checked against denetim_turu_2026-11-05.json
  grid AE training time     gen/tables_v3_provenance.txt (make_tables_v3.py, from results/runs_ped{1,2}_2d_noadv.log)
  median delay              vadkit/configs/r2/base.yaml (temporal causal_median k) and the frame rates of
                            kanit/kurulum_20261105/dataeff_<ds>_1.0.json (frames / minutes / 60)
  FastAno, Ouyang et al.    kanit/literature/dogrulanmis_literatur_2026-11-04.json (lightweight list) and
                            kanit/literature/online_rakipler_2026-11-05.json (published values with quotes)
  NIW seeds (Method)        vadkit/configs/r2/base.yaml (seed), vadkit/pipeline/offline.py, vadkit/density/niw.py,
                            aean/niw.py (_kmeans, NIWMixture.fit defaults); checked, recorded in the provenance file
  pre-registration          git show of the four commits cited in Section prereg (checked against main_v3.tex);
                            budgets and slot order from docs/tez_notlari/tur2_on_kayit_2026-11-05.md; round-2
                            hypothesis outcomes from docs/tez_notlari/sonuclar_tur2_hb_trt_2026-11-05.md (H1 recomputed)
"""
import json
import re
import statistics
import subprocess
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # paper/ieee_access
REPO = ROOT.parents[1]                  # cctv-aean
K = ROOT / "kanit"
GEN = ROOT / "gen"
R2 = K / "r2_20261105"

DS4 = ["ped1", "ped2", "avenue", "shanghaitech"]
DSN = {"ped1": "Ped1", "ped2": "Ped2", "avenue": "Avenue", "shanghaitech": "ShT"}
MAC = {"ped1": "PedOne", "ped2": "PedTwo", "avenue": "Avenue", "shanghaitech": "SHT"}
OBJ = ("ovel", "opose", "oapp")

# component study: config directory -> label; order as in the round-2 pre-registration (slot by slot), B3 last
CONFIGS = [
    ("base", r"Base$^{e}$"),
    ("det_y11s", r"Detector YOLO11s"),
    ("det_y11m", r"Detector YOLO11m"),
    ("pose_y11s", r"Pose YOLO11s-pose"),
    ("enc_mc2s2", r"MobileCLIP2-S2"),
    ("enc_mc2b", r"MobileCLIP2-B"),
    ("enc_clipb16", r"CLIP ViT-B/16"),
    ("enc_dinov2s", r"DINOv2 ViT-S/14"),
    ("b3", r"B3 rule$^{c}$"),
]
# what each config changes, checked against vadkit/configs/r2/<cfg>.yaml so that the labels stay true
CONFIG_CHECK = {
    "det_y11s": ["yolo11s.pt"], "det_y11m": ["yolo11m.pt"], "pose_y11s": ["yolo11s-pose.pt"],
    "enc_mc2s2": ["MobileCLIP2-S2"], "enc_mc2b": ["MobileCLIP2-B"], "enc_clipb16": ["ViT-B-16", "openai"],
    "enc_dinov2s": ["vit_small_patch14_dinov2"], "b3": ["yolo11m.pt", "MobileCLIP2-S2", "yolo11n-pose.pt"],
    "base": ["yolo11n.pt", "yolo11n-pose.pt", "MobileCLIP2-S0"],
}
# result files without a pose channel (no pose model fitted for that configuration and dataset); checked in the data
MISSING_POSE = {("pose_y11s", "ped2")}
LIT_KEY = {"ped2": "ped2_auc", "avenue": "avenue_auc", "shanghaitech": "shtech_auc"}

PROV = []
MACROS = []
CALIB = {}
# v4 outputs (gen/tab_online_v4.tex, gen/supp_tab_component_v4.tex) and the check of supp_v4.tex: provenance in
# gen/text_v4_provenance.txt, so that every v3 output stays byte-identical
PROV4 = []


def rel(p: Path) -> str:
    return str(p.relative_to(REPO))


def jload(p: Path):
    if not p.exists():
        raise SystemExit(f"missing input: {p}")
    return json.loads(p.read_text())


def fused(d: dict, members) -> dict:
    """The fused entry of a result file: its key is the '+'-joined channel set (order differs on ShT)."""
    for k, v in d.items():
        if "+" in k and set(k.split("+")) == set(members):
            return v
    raise SystemExit(f"no fused entry over {sorted(members)} in {list(d)}")


def pts(x, d=1):
    """AUC difference in points, signed, typographic minus."""
    v = 100 * x
    return ("$-$" if v < 0 else "$+$") + f"{abs(v):.{d}f}"


def pc(x, d=1):
    return f"{100 * x:.{d}f}"


def half_up(x, d=1):
    """Round half up on the shortest decimal form (minutes are frames / frame rate: 1.15 -> 1.2, not 1.1)."""
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def macro(name, value, source, label=None):
    if not re.fullmatch(r"Vt[A-Za-z]+", name):
        raise SystemExit(f"bad macro name {name}")
    MACROS.append((name, value))
    PROV.append(("text", label or name, value, source))


def ci(c, d=1):
    return f"[{pts(c[0], d)}, {pts(c[1], d)}]"


# --------------------------------------------------------------------------------------------- grid-branch bootstrap
def grid_bootstrap():
    rows = []
    for ds in DS4:
        p = R2 / "boot" / f"{ds}.json"
        b = jload(p)["grid+obj_vs_obj"]
        if b["skipped"] != 0:
            raise SystemExit(f"{p}: skipped replicates")
        # the bootstrap point values must equal the base result files (same scores)
        g = fused(jload(R2 / "res" / "base" / f"{ds}_grid.json"), OBJ + ("I", "Fs", "Fd"))["causal"]
        o = fused(jload(R2 / "res" / "base" / f"{ds}_obj.json"), OBJ)["causal"]
        if abs(g - b["auc"]["a"]["point"]) > 1e-9 or abs(o - b["auc"]["b"]["point"]) > 1e-9:
            raise SystemExit(f"{p}: bootstrap point values differ from res/base")
        dpt, dci = b["difference"]["point"], b["difference"]["ci95"]
        sig = dci[0] > 0 or dci[1] < 0
        rows.append((ds, o, g, dpt, dci, sig, b["videos"], b["replicates"]))
        m = MAC[ds]
        src = f"{rel(p)} : grid+obj_vs_obj"
        macro(f"VtGridGain{m}", pts(dpt), src + " / difference.point")
        macro(f"VtGridGainCI{m}", ci(dci), src + " / difference.ci95")
        macro(f"VtTestVideos{m}", str(b["videos"]), src + " / videos")
    macro("VtBootReps", f"{rows[0][7]:,}", f"{rel(R2 / 'boot' / 'ped1.json')} : replicates")
    seeds = {jload(R2 / "boot" / f"{ds}.json")["grid+obj_vs_obj"]["seed"] for ds in DS4}
    if len(seeds) != 1:
        raise SystemExit("bootstrap seeds differ between datasets")
    macro("VtBootSeed", str(seeds.pop()), f"{rel(R2 / 'boot')}/<ds>.json : grid+obj_vs_obj.seed (all datasets)")
    return rows


# --------------------------------------------------------------------------------------------- component study
def component_study():
    out = []
    for cfg, label in CONFIGS:
        y = REPO / "vadkit" / "configs" / "r2" / f"{cfg}.yaml"
        if not y.exists():
            raise SystemExit(f"missing config {y}")
        txt = y.read_text()
        for s in CONFIG_CHECK[cfg]:
            if s not in txt:
                raise SystemExit(f"{y}: expected {s!r}")
        bp = R2 / "bench" / f"{cfg}.json"
        bench = jload(bp)
        if bench.get("parallel_flow"):
            raise SystemExit(f"{bp}: expected the sequential pipeline")
        if set(bench["reps"][0]) != {"decode", "detect", "pose", "flow", "embed", "total"}:
            raise SystemExit(f"{bp}: stages changed; footnote b of the ablation table says scoring is not timed")
        tot = statistics.median(r["total"]["median_ms"] for r in bench["reps"])
        PROV.append(("ablation", f"{cfg} latency (ms)", f"{tot:.1f}",
                     f"{rel(bp)} : median of {len(bench['reps'])} per-repetition median totals"))
        aucs = {}
        for ds in DS4:
            rp = R2 / "res" / cfg / f"{ds}_obj.json"
            d = jload(rp)
            if (cfg, ds) in MISSING_POSE:          # no pose model was fitted: fused over velocity and appearance
                if "opose" in d:
                    raise SystemExit(f"{rp}: pose channel present, MISSING_POSE is stale")
                v = fused(d, ("ovel", "oapp"))["causal"]
            else:
                v = fused(d, OBJ)["causal"]
            aucs[ds] = v
            PROV.append(("ablation", f"{cfg} {ds} object branch online", pc(v), f"{rel(rp)} : fused causal"))
        out.append((cfg, label, tot, aucs, bench["frame_hw"], bench["device"]))
    lat = [r[2] for r in out]
    single = [r for r in out if r[0] not in ("base", "b3")]
    others = [r[2] for r in out if r[0] not in ("enc_mc2s2", "b3")]
    macro("VtCompLatMin", f"{min(others):.0f}", "ablation latencies of every config except MobileCLIP2-S2 and B3: min")
    macro("VtCompLatMax", f"{max(others):.0f}", "ablation latencies of every config except MobileCLIP2-S2 and B3: max")
    s2 = [r for r in out if r[0] == "enc_mc2s2"][0][2]
    b3 = [r for r in out if r[0] == "b3"][0][2]
    macro("VtCompLatSTwo", f"{s2:.0f}", f"{rel(R2 / 'bench' / 'enc_mc2s2.json')} : median total")
    macro("VtCompLatBThree", f"{b3:.0f}", f"{rel(R2 / 'bench' / 'b3.json')} : median total")
    # largest gain and loss of a single swap (B3 excluded) over all datasets, from the AUCs as rounded in the table
    # (one decimal, so that the reader can recompute them from Table ablation b); ShT: no config improves it (checked)
    base = out[0][3]
    sw = [((float(pc(r[3][ds])) - float(pc(base[ds]))) / 100, ds, r[0]) for r in single for ds in DS4]
    g = max(sw)
    macro("VtCompMaxGain", pts(g[0]), f"max over single swaps and datasets of res/<cfg>/<ds>_obj - res/base, "
                                      f"rounded table values ({g[2]}, {g[1]})")
    macro("VtCompMaxGainDs", {"ped1": "Ped1", "ped2": "Ped2", "avenue": "Avenue", "shanghaitech": "ShanghaiTech"}[g[1]],
          "dataset of VtCompMaxGain")
    lo = min(sw)
    macro("VtCompMaxLoss", f"{-100 * lo[0]:.1f}", f"-min over single swaps and datasets, rounded table values "
                                                  f"({lo[2]}, {lo[1]})")
    if max(r[3]["shanghaitech"] - base["shanghaitech"] for r in out[1:]) >= 0:
        raise SystemExit("a configuration improves ShT: the text says none does")
    for ds in DS4:
        # from the AUCs as rounded in the table, like VtCompMaxGain / VtCompMaxLoss
        diffs = [(float(pc(r[3][ds])) - float(pc(base[ds]))) / 100 for r in out[1:]]
        macro(f"VtCompBest{MAC[ds]}", pts(max(diffs)), f"max over configs of res/<cfg>/{ds}_obj.json - res/base ({ds}), "
                                                       f"rounded table values")
        macro(f"VtCompWorst{MAC[ds]}", pts(min(diffs)), f"min over configs of res/<cfg>/{ds}_obj.json - res/base ({ds}), "
                                                        f"rounded table values")
    # appearance channel on Avenue: base encoder and best encoder
    app = {}
    for cfg, *_ in out:
        rp = R2 / "res" / cfg / "avenue_obj.json"
        app[cfg] = jload(rp)["oapp"]["causal"]
    best = max((c for c in app if c.startswith("enc_")), key=app.get)
    macro("VtAppAvenueBase", pc(app["base"]), f"{rel(R2 / 'res/base/avenue_obj.json')} : oapp causal")
    macro("VtAppAvenueBest", pc(app[best]), f"{rel(R2 / 'res' / best / 'avenue_obj.json')} : oapp causal ({best})")
    names = {"enc_mc2b": "MobileCLIP2-B", "enc_mc2s2": "MobileCLIP2-S2", "enc_clipb16": "CLIP ViT-B/16",
             "enc_dinov2s": "DINOv2 ViT-S/14"}
    macro("VtAppAvenueBestName", names[best], "argmax of the Avenue oapp causal AUC over the encoder configs")
    fb = (float(pc([r for r in out if r[0] == best][0][3]["avenue"])) - float(pc(out[0][3]["avenue"]))) / 100
    macro("VtAppAvenueBestFused", pts(fb), f"res/{best}/avenue_obj.json fused causal - res/base/avenue_obj.json fused "
                                           f"causal, rounded table values")
    return out


# --------------------------------------------------------------------------------------------- per-channel numbers
def channels():
    for ds in DS4:
        p = R2 / "res" / "base" / f"{ds}_grid.json"
        d = jload(p)
        g = fused(d, OBJ + ("I", "Fs", "Fd"))
        macro(f"VtGridStd{MAC[ds]}", pc(g["offline"]), f"{rel(p)} : fused offline")
        macro(f"VtGridOn{MAC[ds]}", pc(g["causal"]), f"{rel(p)} : fused causal")
        p2 = R2 / "res" / "base" / f"{ds}_obj.json"
        o = fused(jload(p2), OBJ)
        macro(f"VtObjStd{MAC[ds]}", pc(o["offline"]), f"{rel(p2)} : fused offline")
        macro(f"VtObjOn{MAC[ds]}", pc(o["causal"]), f"{rel(p2)} : fused causal")
    d = jload(R2 / "res" / "base" / "shanghaitech_grid.json")
    macro("VtShtIOn", pc(d["I"]["causal"]), f"{rel(R2 / 'res/base/shanghaitech_grid.json')} : I causal")
    macro("VtShtGridMacro", pc(fused(d, OBJ + ("I", "Fs", "Fd"))["causal_macro"]),
          f"{rel(R2 / 'res/base/shanghaitech_grid.json')} : fused causal_macro")
    o = jload(R2 / "res" / "base" / "shanghaitech_obj.json")
    macro("VtShtObjMacro", pc(fused(o, OBJ)["causal_macro"]),
          f"{rel(R2 / 'res/base/shanghaitech_obj.json')} : fused causal_macro")


# --------------------------------------------------------------------------------------------- HB-POT
def hbpot():
    p, q = K / "hb_20261105" / "results.json", K / "hb_20261105" / "gate.json"
    r, g = jload(p)["verdicts"], jload(q)["gates"]
    macro("VtHbGates", str(len(g)), f"{rel(q)} : number of gates")
    macro("VtHbGatesPassed", str(sum(bool(v["pass"]) for v in g.values())), f"{rel(q)} : gates passed")
    macro("VtHbHOne", pts(r["H1"]["point"], 2), f"{rel(p)} : verdicts.H1.point")
    macro("VtHbHOneCI", ci(r["H1"]["ci95"], 2), f"{rel(p)} : verdicts.H1.ci95")
    macro("VtHbHTwo", pts(r["H2"]["point"]), f"{rel(p)} : verdicts.H2.point")
    macro("VtHbHTwoCI", ci(r["H2"]["ci95"]), f"{rel(p)} : verdicts.H2.ci95")
    macro("VtHbZTenBefore", str(g["G2"]["n_z10_T0"]), f"{rel(q)} : gates.G2.n_z10_T0")
    macro("VtHbZTenAfter", str(g["G2"]["n_z10_T2"]), f"{rel(q)} : gates.G2.n_z10_T2")
    if r["decision"]["adopt_T2"]:
        raise SystemExit("HB-POT T2 adopted: the text assumes it was not")


# --------------------------------------------------------------------------------------------- label-free threshold
def far():
    for ds in DS4:
        p = K / "kurulum_20261105" / f"far_{ds}.json"
        d = jload(p)
        rows = {round(r["alpha"], 4): r for r in d["empirical"]["smoothed"]}
        for a, tag in ((0.05, "Five"), (0.01, "One")):
            macro(f"VtFar{tag}{MAC[ds]}", pc(rows[a]["far"]), f"{rel(p)} : empirical.smoothed alpha={a} far")
            macro(f"VtFarRatio{tag}{MAC[ds]}", f"{rows[a]['far_over_alpha']:.2f}",
                  f"{rel(p)} : empirical.smoothed alpha={a} far_over_alpha")
            CALIB[(ds, a)] = bool(rows[a]["calibrated_0.5_2"])
            if CALIB[(ds, a)] != (0.5 <= rows[a]["far_over_alpha"] <= 2):
                raise SystemExit(f"{p}: calibrated_0.5_2 disagrees with far_over_alpha")
        macro(f"VtTpr{MAC[ds]}", pc(rows[0.05]["tpr"], 0), f"{rel(p)} : empirical.smoothed alpha=0.05 tpr")
        tv = sum(g["train_videos"] for g in d["groups"].values())
        macro(f"VtTrainVideos{MAC[ds]}", str(tv), f"{rel(p)} : sum of groups.*.train_videos")
        macro(f"VtTestNormal{MAC[ds]}", f"{rows[0.05]['normal_frames']:,}", f"{rel(p)} : normal_frames")
    # pre-registered criterion (kurulum_olcumleri_on_kayit M1): FAR / alpha in [0.5, 2] at alpha = 5% and 1%; the
    # Limitations text describes this pattern, so it is checked here
    both = [ds for ds in ("ped1", "ped2", "avenue") if CALIB[(ds, 0.05)] and CALIB[(ds, 0.01)]]
    macro("VtFarCalibMain", str(len(both)), "far_<ds>.json : main datasets calibrated at both alpha levels")
    expect = {("ped1", 0.05): True, ("ped2", 0.05): True, ("avenue", 0.05): False, ("ped1", 0.01): False,
              ("ped2", 0.01): False, ("avenue", 0.01): False, ("shanghaitech", 0.05): True,
              ("shanghaitech", 0.01): True}
    if any(CALIB[k] != v for k, v in expect.items()):
        raise SystemExit("threshold calibration pattern changed: revise the Limitations text")
    # further pre-registered items of M1 (kurulum_olcumleri_on_kayit): bootstrap intervals of the false-alarm rate
    # and the analytic threshold (independent channels, Gamma(k, 1)); the text states their pattern, checked here
    p = K / "kurulum_20261105" / "far_ped2.json"
    r = {round(x["alpha"], 4): x for x in jload(p)["empirical"]["smoothed"]}[0.05]
    macro("VtFarCiLoFivePedTwo", pc(r["far_ci95"][0]), f"{rel(p)} : empirical.smoothed alpha=0.05 far_ci95[0]")
    macro("VtFarCiHiFivePedTwo", pc(r["far_ci95"][1]), f"{rel(p)} : empirical.smoothed alpha=0.05 far_ci95[1]")
    for br, want_same in (("obj", True), ("grid+obj", False)):
        inside = {}
        for ds in DS4:
            q = K / "kurulum_20261105" / f"far_{ds}.json"
            for x in jload(q)["analytic"][br]["rows"]:
                if round(x["alpha"], 4) in (0.05, 0.01):
                    inside[(ds, round(x["alpha"], 4))] = 0.5 <= x["far"] / x["alpha"] <= 2
        same = all(inside[k] == CALIB[k] for k in inside)
        if want_same and not same or not want_same and any(inside.values()):
            raise SystemExit(f"analytic threshold ({br}) pattern changed: revise the Limitations text")


# --------------------------------------------------------------------------------------------- data efficiency
def dataeff():
    for ds in DS4:
        a = jload(K / "kurulum_20261105" / f"dataeff_{ds}_0.25.json")["rows"]
        b = jload(K / "kurulum_20261105" / f"dataeff_{ds}_1.0.json")["rows"]
        d = statistics.mean(r["causal"] for r in a) - statistics.mean(r["causal"] for r in b)
        src = f"kanit/kurulum_20261105/dataeff_{ds}_0.25.json mean causal - dataeff_{ds}_1.0.json causal"
        macro(f"VtEffQuarter{MAC[ds]}", pts(d), src)
        mins = [r["minutes"] for r in a]
        lo, hi = min(mins), max(mins)
        macro(f"VtEffQuarterMin{MAC[ds]}",
              half_up(lo) if half_up(lo) == half_up(hi) else f"{half_up(lo)}--{half_up(hi)}",
              f"kanit/kurulum_20261105/dataeff_{ds}_0.25.json : minutes range over draws (rounded half up)")
    allm = [r["minutes"] for ds in ("ped1", "ped2", "avenue")
            for r in jload(K / "kurulum_20261105" / f"dataeff_{ds}_0.25.json")["rows"]]
    macro("VtEffQuarterMinRange", f"{half_up(min(allm))}--{half_up(max(allm))}",
          "kanit/kurulum_20261105/dataeff_{ped1,ped2,avenue}_0.25.json : minutes, min and max over all draws")
    a = jload(K / "kurulum_20261105" / "dataeff_shanghaitech_0.25.json")["rows"]
    b = jload(K / "kurulum_20261105" / "dataeff_shanghaitech_1.0.json")["rows"]
    d = statistics.mean(r["causal_macro"] for r in a) - statistics.mean(r["causal_macro"] for r in b)
    macro("VtEffQuarterMacroSHT", pts(d), "dataeff_shanghaitech_{0.25,1.0}.json causal_macro (mean)")


# --------------------------------------------------------------------------------------------- Avenue labels
def avenue_labels():
    p = K / "avenue_etiket_20261105" / "diff.json"
    d = jload(p)
    pos = d["total_positive_frames"]
    if pos["ours_gt_json"] != pos["cuhk_mask_anypixel"] or pos["vpc_avenue_mat"] != pos["aedmae_avenue_gt_txt"]:
        raise SystemExit("Avenue label groups changed")
    macro("VtAvePosA", f"{pos['cuhk_mask_anypixel']:,}", f"{rel(p)} : total_positive_frames.cuhk_mask_anypixel")
    macro("VtAvePosB", f"{pos['vpc_avenue_mat']:,}", f"{rel(p)} : total_positive_frames.vpc_avenue_mat")
    s = d["cuhk_vs_vpc_summary"]
    macro("VtAveOnlyA", str(s["only_cuhk_frames"]), f"{rel(p)} : cuhk_vs_vpc_summary.only_cuhk_frames")
    macro("VtAveOnlyB", str(s["only_vpc_frames"]), f"{rel(p)} : cuhk_vs_vpc_summary.only_vpc_frames")
    macro("VtAveEventsB", str(s["n_events_vpc"]), f"{rel(p)} : cuhk_vs_vpc_summary.n_events_vpc")
    macro("VtAveRunsA", str(s["n_runs_cuhk"]), f"{rel(p)} : cuhk_vs_vpc_summary.n_runs_cuhk")


# --------------------------------------------------------------------------------------------- accuracy gaps
def gaps():
    """Accuracy gaps of our two configurations within their supervision blocks (placement by the block definitions:
    object branch -> training-free, object + grid -> one-class, both on every dataset; written down in
    kurulum_olcumleri_on_kayit at 11:41 on 5 Nov, after the round-2 test results, so not a pre-registration),
    and their comparison with the synthetic-anomaly block. Competitors and blocks are those of the main table
    (tables_v3_inputs.json); standard protocol, values rounded as shown in the table. Every comparison the Accuracy
    and Size sections state in words is checked here."""
    p = K / "literature" / "dogrulanmis_literatur_2026-11-04.json"
    sota = jload(p)["sota"]["papers"]
    aud = jload(K / "literature" / "denetim_turu_2026-11-05.json")
    comps = jload(HERE / "tables_v3_inputs.json")["competitors"]
    block_of = {"grid": "one_class_trained", "obj": "training_free"}
    DS3 = ("ped2", "avenue", "shanghaitech")

    def published(c, ds):
        a = [x for x in aud if x["title"].startswith(c["audit"])]
        if len(a) != 1 or a[0]["supervision"] != c["block"]:
            raise SystemExit(f"{c['name']}: block not confirmed by the audit file")
        hits = [x for x in sota if x["title"].startswith(c["lit"][1])]
        if len(hits) != 1:
            raise SystemExit(f"{p}: {len(hits)} entries for {c['lit'][1]!r}")
        v = hits[0][LIT_KEY[ds]]
        return None if v is None else round(float(v), 1)

    ours = {}
    for ds in DS3:
        for br in ("obj", "grid"):
            r = jload(R2 / "res" / "base" / f"{ds}_{br}.json")
            ours[(ds, br)] = float(pc(fused(r, OBJ + (("I", "Fs", "Fd") if br == "grid" else ()))["offline"]))
    for br in ("obj", "grid"):
        for ds in DS3:
            g = [(round(v - ours[(ds, br)], 1), c["name"]) for c in comps if c["block"] == block_of[br]
                 for v in [published(c, ds)] if v is not None]
            src = (f"{rel(p)} sota {LIT_KEY[ds]} of the {block_of[br]} block minus ours standard "
                   f"({br}, {ours[(ds, br)]})")
            above = [(x, n) for x, n in g if x > 0]
            below = [(x, n) for x, n in g if x <= 0]
            if br == "obj":
                # text: on Ped2 and ShT every training-free method is more accurate; on Avenue all but AnyAnomaly
                want_below = ["AnyAnomaly"] if ds == "avenue" else []
                if sorted(n for _, n in below) != want_below or not above:
                    raise SystemExit(f"{ds}: training-free methods below the object branch: {below}; revise the text")
                macro(f"VtGapTfMin{MAC[ds]}", f"{min(above)[0]:.1f}", src + ": min over the more accurate methods")
                macro(f"VtGapTfMax{MAC[ds]}", f"{max(above)[0]:.1f}", src + ": max over the more accurate methods")
                if below:
                    macro(f"VtTfBelow{MAC[ds]}", f"{-below[0][0]:.1f}", src + f": {below[0][1]} (less accurate)")
            else:
                # text: MULDE on every dataset, STG-NF on ShT only, both more accurate than object + grid
                names = sorted(n for _, n in g)
                if names != (["MULDE", "STG-NF"] if ds == "shanghaitech" else ["MULDE"]) or below:
                    raise SystemExit(f"{ds}: one-class methods {g}; revise the text")
                macro(f"VtGapOc{MAC[ds]}", f"{dict((n, x) for x, n in g)['MULDE']:.1f}", src + ": MULDE")
                if ds == "shanghaitech":
                    macro("VtGapOcStgnfSHT", f"{dict((n, x) for x, n in g)['STG-NF']:.1f}", src + ": STG-NF")
    # synthetic-anomaly block (AED-MAE, Fast-AED): the pattern stated in the Accuracy and Size sections
    syn = [c for c in comps if c["block"] == "synthetic_or_pseudo_anomaly"]
    if sorted(c["name"] for c in syn) != ["AED-MAE", "Fast-AED"]:
        raise SystemExit("synthetic-anomaly block changed; revise the text")
    sign = {(br, ds, c["name"]): (published(c, ds) > ours[(ds, br)]) - (published(c, ds) < ours[(ds, br)])
            for br in ("obj", "grid") for ds in DS3 for c in syn}
    want = {**{("obj", ds, n): 1 for ds in DS3 for n in ("AED-MAE", "Fast-AED")},
            ("obj", "shanghaitech", "Fast-AED"): 0,
            ("grid", "ped2", "AED-MAE"): -1, ("grid", "ped2", "Fast-AED"): -1,
            ("grid", "avenue", "AED-MAE"): 1, ("grid", "avenue", "Fast-AED"): -1,
            ("grid", "shanghaitech", "AED-MAE"): 1, ("grid", "shanghaitech", "Fast-AED"): 1}
    if sign != want:
        raise SystemExit(f"synthetic-anomaly comparison changed: {sign}; revise the Accuracy and Size sections")
    PROV.append(("text", "synthetic-anomaly vs ours (sign of published - ours)", str(sign),
                 f"{rel(p)} sota and res/base offline, rounded to one decimal"))


def protocols():
    """Online vs standard micro AUC of our two configurations (Table tab:online): range of the differences on the
    three main datasets, and the configuration ranking under both protocols, as stated in the Accuracy section."""
    diffs = []
    for ds in DS4:
        v = {}
        for br, members in (("obj", OBJ), ("grid", OBJ + ("I", "Fs", "Fd"))):
            f = fused(jload(R2 / "res" / "base" / f"{ds}_{br}.json"), members)
            v[br] = {k: float(pc(f[k])) for k in ("offline", "causal")}
            if ds != "ped1":
                diffs.append(round(v[br]["causal"] - v[br]["offline"], 1))
        better = {k: "grid" if v["grid"][k] > v["obj"][k] else "obj" for k in ("offline", "causal")}
        want = "obj" if ds == "shanghaitech" else "grid"
        if better != {"offline": want, "causal": want}:
            raise SystemExit(f"{ds}: configuration ranking {better}; the text says grid helps on Ped1, Ped2, Avenue "
                             f"and harms ShT under both protocols")
    src = "kanit/r2_20261105/res/base/<ped2,avenue,shanghaitech>_<obj,grid>.json : fused causal - offline (rounded)"
    if min(diffs) >= 0 or max(diffs) <= 0:
        raise SystemExit("online minus standard no longer has both signs; revise the text")
    macro("VtOnStdDiffMin", pts(min(diffs) / 100), src + ": min")
    macro("VtOnStdDiffMax", pts(max(diffs) / 100), src + ": max")


# --------------------------------------------------------------------------------------------- median delay
def median_delay():
    """Delay of the causal median (Section fusion): a median of the last k scores follows a step after (k-1)/2
    frames and removes a rise shorter than (k+1)/2 frames; seconds at the frame rate of each dataset."""
    y = REPO / "vadkit" / "configs" / "r2" / "base.yaml"
    if not y.exists():
        raise SystemExit(f"missing config {y}")
    m = re.search(r"temporal:\s*\n\s*name:\s*causal_median\s*\n\s*args:\s*\n\s*k:\s*(\d+)", y.read_text())
    if not m:
        raise SystemExit(f"{y}: causal_median k not found")
    k = int(m.group(1))
    if k != 9:
        raise SystemExit(f"{y}: median length {k}; the text says nine scores, four frames of delay")
    lag = (k - 1) // 2
    fps = {}
    for ds in DS4:
        q = K / "kurulum_20261105" / f"dataeff_{ds}_1.0.json"
        r = jload(q)["rows"][0]
        fps[ds] = r["frames"] / r["minutes"] / 60
        if abs(fps[ds] - round(fps[ds])) > 1e-6:
            raise SystemExit(f"{q}: frame rate {fps[ds]} is not an integer")
        PROV.append(("text", f"frame rate {ds}", f"{round(fps[ds])}", f"{rel(q)} : frames / minutes / 60"))
    if round(fps["ped1"]) != round(fps["ped2"]):
        raise SystemExit("Ped1 and Ped2 frame rates differ; the text gives one UCSD rate")
    for ds, name in (("ped2", "UCSD"), ("avenue", "Avenue"), ("shanghaitech", "SHT")):
        f = round(fps[ds])
        macro(f"VtFps{name}", str(f), f"kanit/kurulum_20261105/dataeff_{ds}_1.0.json : frames / minutes / 60")
        sec = lag / f
        txt = f"{sec:.2f}".rstrip("0").rstrip(".")
        if abs(float(txt) - sec) > 1e-9 and ds != "shanghaitech":
            raise SystemExit(f"{ds}: delay {sec} is not exact at two decimals; the text says 'about' only for ShT")
        macro(f"VtMedDelay{name}", txt, f"({k} - 1) / 2 = {lag} frames (vadkit/configs/r2/base.yaml k) / {f} frames/s")


# --------------------------------------------------------------------------------------------- literature values
def literature():
    """Published values of FastAno (Related Work) and Ouyang et al. (Related Work), from the verified literature
    files; the comparison with our results stated in the text is checked here."""
    p = K / "literature" / "dogrulanmis_literatur_2026-11-04.json"
    fa = [x for x in jload(p)["lightweight"]["papers"] if x["title"].startswith("FastAno")]
    if len(fa) != 1 or fa[0]["status"] != "confirmed":
        raise SystemExit(f"{p}: FastAno entry missing or not confirmed")
    fa = fa[0]
    if "'Its value for our model is 2.15 M'" not in fa["evidence"] or "FPS 195" not in fa["evidence"]:
        raise SystemExit(f"{p}: FastAno quotes changed")
    macro("VtFastAnoFps", str(fa["fps"]), f"{rel(p)} : lightweight / FastAno / fps (paper Table 1, 'FPS 195')")
    m = re.match(r"([\d.]+)M trainable params \(self-reported\)", fa["params_or_flops"])
    macro("VtFastAnoParams", m.group(1), f"{rel(p)} : lightweight / FastAno / params_or_flops (paper: 'Its value for "
                                         f"our model is 2.15 M')")
    m = re.search(r"lists FastAno at [\d.]+ GFLOPs / (\d+)M params", fa["params_or_flops"])
    macro("VtFastAnoParamsSdmae", m.group(1), f"{rel(p)} : lightweight / FastAno / params_or_flops (SD-MAE Table 5)")
    q = K / "literature" / "online_rakipler_2026-11-05.json"
    oy = [x for x in jload(q)["rows"] if x["title"].startswith("Look at Adjacent Frames")]
    if len(oy) != 1 or oy[0]["verdict"] != "online_confirmed" or oy[0]["supervision"] != "training_free":
        raise SystemExit(f"{q}: Ouyang et al. entry changed")
    oy = oy[0]
    if "per-video min-max for the AUC cannot be ruled out" not in oy["causality_evidence"]:
        raise SystemExit(f"{q}: Ouyang normalisation caveat changed; revise the Related Work sentence")
    # paper Sec. 3.1 / 4: k is the maximum number of GD iterations (the adapter also stops when the loss is small), 500
    # on the first 5 frames of each video and 100 afterwards; the Related Work sentence says "up to"
    it = re.search(r"up to k=(\d+) gradient-descent iterations \(k=(\d+) for the first (\d+) frames of each video\)",
                   oy["fps_or_latency"])
    if not it:
        raise SystemExit(f"{q}: Ouyang iteration count changed; revise the Related Work sentence")
    words = {5: "five"}
    if int(it.group(3)) not in words:
        raise SystemExit(f"{q}: Ouyang first-frame count {it.group(3)} has no number word; revise literature()")
    for ds, key in (("ped2", "ped2_auc"), ("avenue", "avenue_auc"), ("shanghaitech", "shtech_auc")):
        v = float(re.match(r"([\d.]+)", oy[key]).group(1))
        macro(f"VtOuyang{MAC[ds]}", f"{v:.1f}", f"{rel(q)} : {key} (paper Table 1)")
        # text: higher than our object branch (the training-free configuration) under both protocols
        f = fused(jload(R2 / "res" / "base" / f"{ds}_obj.json"), OBJ)
        if v <= max(float(pc(f["offline"])), float(pc(f["causal"]))):
            raise SystemExit(f"{ds}: Ouyang et al. not above our object branch; revise the Related Work sentence")
    src = f"{rel(q)} : fps_or_latency ({it.group(0)!r})"
    macro("VtOuyangIters", it.group(1), src + " : k after the first frames")
    macro("VtOuyangItersFirst", it.group(2), src + " : k on the first frames")
    macro("VtOuyangFirstFrames", words[int(it.group(3))], src + f" : number of first frames ({it.group(3)})")


# --------------------------------------------------------------------------------------------- values of make_tables_v3
def from_tables_provenance():
    p = GEN / "tables_v3_provenance.txt"
    if not p.exists():
        raise SystemExit("run make_tables_v3.py first")
    secs = []
    for line in p.read_text().splitlines():
        m = re.match(r"size\s*\|\s*grid AE training \(results/runs_(ped\d)_2d_noadv\.log\)\s*\|\s*([\d.]+) s for (\d+) steps",
                     line)
        if m:
            secs.append((float(m.group(2)), int(m.group(3))))
    if len(secs) != 2:
        raise SystemExit(f"{p}: grid AE training lines not found")
    lo, hi = min(s for s, _ in secs) / 60, max(s for s, _ in secs) / 60
    macro("VtAeTrainMin", f"{lo:.1f}--{hi:.1f}", f"{rel(p)} : grid AE training (s / 60)")
    # parallel flow vs sequential: identical objects and attributes
    q = K / "hiz_20261105" / "equivalence.log"
    if not q.exists():
        raise SystemExit(f"missing input: {q}")
    m = re.search(r"frames (\d+) mismatching frames (\d+)", q.read_text())
    if not m or int(m.group(2)) != 0:
        raise SystemExit(f"{q}: no equivalence line or mismatching frames")
    macro("VtParEquivFrames", m.group(1), f"{rel(q)} : frames (mismatching 0)")
    macro("VtAeSteps", f"{secs[0][1]:,}", f"{rel(p)} : grid AE training steps")


# --------------------------------------------------------------------------------------------- code constants
def code_constants():
    """Seeds stated in the Method section ('a NumPy generator seeded with 0', 'k-means ... with seed 0'), read from
    the code and the configuration that produced the results; a change stops the script."""
    checks = [
        (REPO / "vadkit" / "configs" / "r2" / "base.yaml", r"(?m)^seed:\s*(\d+)\s*$",
         "NIW held-out subsample: config seed (vadkit/pipeline/offline.py fit(): default_rng(cfg seed) per group)"),
        (REPO / "vadkit" / "pipeline" / "offline.py", r"rng = np\.random\.default_rng\(int\(cfg\.get\(\"seed\", (\d+)\)\)\)",
         "default seed of the per-group NumPy generator passed to select_hyper"),
        (REPO / "vadkit" / "density" / "niw.py", r"self\.rng if self\.rng is not None else np\.random\.default_rng\((\d+)\)",
         "NIWDensity.fit(): fallback NumPy generator of select_hyper"),
        (REPO / "aean" / "niw.py", r"def _kmeans\(X: torch\.Tensor, K: int, iters: int = 30, seed: int = (\d+)\)",
         "k-means initial centres: K training vectors by torch.randperm with this seed"),
        (REPO / "aean" / "niw.py", r"class NIWMixture:.*?def fit\(self, X: torch\.Tensor, seed: int = (\d+)",
         "NIWMixture.fit() default seed (vadkit NIWDensity.fit calls it without a seed)"),
    ]
    for p, pat, what in checks:
        if not p.exists():
            raise SystemExit(f"missing input: {p}")
        m = re.search(pat, p.read_text(), re.S)
        if not m:
            raise SystemExit(f"{p}: pattern for '{what}' not found")
        if int(m.group(1)) != 0:
            raise SystemExit(f"{p}: {what} is {m.group(1)}; the Method section says seed 0")
        PROV.append(("text", "seed 0 (Method, NIW selection)", m.group(1), f"{rel(p)} : {what}"))
    # select_mixture runs k-means without a seed (default 0), and NIWDensity.fit calls NIWMixture.fit(X) without one
    a, v = (REPO / "aean" / "niw.py").read_text(), (REPO / "vadkit" / "density" / "niw.py").read_text()
    if "_kmeans(Xf, K) if K > 1" not in a or "NIWMixture(self.K, self.kappa0, self.nu0, device=self.device).fit(X)" not in v:
        raise SystemExit("k-means / NIWMixture.fit calls changed: check the seed sentence of the Method section")


def preregistration():
    """Pre-registration documents and budgets (Section prereg, component study, Table ablation note c). The commit
    hashes, dates and times are written in main_v3.tex; they are checked against git and recorded here."""
    commits = [("3dffc5a", "docs/tez_notlari/tur2_on_kayit_2026-11-05.md"),
               ("e4031e4", "docs/tez_notlari/kurulum_olcumleri_on_kayit_2026-11-05.md"),
               ("2921bc5", "docs/tez_notlari/hiz_on_kayit_2026-11-05.md"),
               ("6b08c14", "docs/tez_notlari/hiyerarsik_kalibrasyon_on_kayit_2026-11-05.md")]
    # The citation check needs the manuscript sources and the dated (private) git history; the public code release
    # has neither, so there the check is skipped and the generated macros are unaffected.
    have_git = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", commits[0][0] + "^{commit}"],
                              capture_output=True).returncode == 0
    if not ((ROOT / "main_v3.tex").exists() and (ROOT / "supp_v4.tex").exists() and have_git):
        print("note: manuscript sources or the dated git history are not available; pre-registration citation check "
              "skipped")
        PROV.append(("text", "pre-registration citation check", "skipped",
                     "manuscript sources or the private git history not available (public release)"))
        commits_checked = []
    else:
        commits_checked = commits
    if commits_checked:
        tex = (ROOT / "main_v3.tex").read_text()
        # supp_v4.tex holds the pre-registration paragraph of main_v3.tex verbatim (Supplementary Section S1); main_v4.tex
        # points to it and gives no commit
        sp = ROOT / "supp_v4.tex"
        if not sp.exists():
            raise SystemExit("supp_v4.tex not found: the pre-registration record of the v4 paper is checked against it")
        supp = sp.read_text()
        months = {11: "Nov."}
        for h, doc in commits_checked:
            def git(*args):
                r = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
                if r.returncode:
                    raise SystemExit(f"git {' '.join(args)}: {r.stderr.strip()}")
                return r.stdout.strip()
            full, ci, ai = git("show", "-s", "--format=%H|%ci|%ai", h).split("|")
            if ci != ai:
                raise SystemExit(f"{h}: author and committer times differ ({ai} / {ci}); the text gives one time")
            m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2}) (\d{2}:\d{2}):\d{2} \+0300", ci)
            if not m:
                raise SystemExit(f"{h}: commit time {ci} is not in UTC+3 as the text says")
            if f"A\t{doc}" not in git("show", "--name-status", "--format=", h).splitlines():
                raise SystemExit(f"{h}: does not add {doc}")
            if git("diff", "--name-only", h, "HEAD", "--", doc) or git("diff", "--name-only", "--", doc):
                raise SystemExit(f"{doc} changed after its pre-registration commit {h}")
            y, mo, d, t = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
            cite = rf"\texttt{{{h}}}, {d}~{months[mo]}~{y}, {t}"
            if cite not in tex:
                raise SystemExit(f"main_v3.tex: pre-registration citation {cite!r} not found")
            if cite not in supp:
                raise SystemExit(f"supp_v4.tex: pre-registration citation {cite!r} not found")
            PROV4.append(("supp_v4.tex", f"pre-registration {h}", cite, f"git show -s {h}; found in supp_v4.tex"))
            PROV.append(("text", f"pre-registration {h}", f"{full} {ci}",
                         f"git show -s {h} (adds {doc}; unchanged since, also in the working tree)"))
    # latency budgets and slot order of the round-2 pre-registration
    p = REPO / commits[0][1]
    txt = p.read_text()
    bud = {b: int(v) for b, v in re.findall(r"\|\s*(B[123])\s*\|\s*≤\s*(\d+)\s*ms\s*\|", txt)}
    if sorted(bud) != ["B1", "B2", "B3"]:
        raise SystemExit(f"{p}: budgets not found")
    if "Sıra dedektör, akış, kodlayıcı, poz" not in txt:
        raise SystemExit(f"{p}: slot order changed; revise note c of the ablation table")
    for b, v in bud.items():
        PROV.append(("text", f"latency budget {b} (ms)", str(v), f"{rel(p)} : budget table"))
    macro("VtBudgetBTwo", str(bud["B2"]), f"{rel(p)} : budget table, B2")
    # outcomes of the round-2 hypotheses on our own system, as stated in the component study
    q = REPO / "docs" / "tez_notlari" / "sonuclar_tur2_hb_trt_2026-11-05.md"
    if not q.exists():
        raise SystemExit(f"missing input: {q}")
    res = q.read_text()
    want = {"H1": "**desteklenmedi.**", "H2": "**sınanamadı**", "H3": "**desteklenmedi.**"}
    for hyp, verdict in want.items():
        line = [x for x in res.splitlines() if x.startswith(f"- **{hyp}**")]
        if len(line) != 1 or verdict not in line[0]:
            raise SystemExit(f"{q}: round-2 {hyp} verdict changed; revise the component study")
        PROV.append(("text", f"round-2 {hyp} outcome", verdict.strip("*."), f"{rel(q)} : section 1, {hyp}"))
    # H1 recomputed: share of the Avenue appearance gap (base encoder to the heavy appearance of the pre-registration)
    # closed by the best larger encoder; supported only at >= 0.5
    heavy = float(re.search(r"ağır görünüm \((0\.\d+)\)", txt).group(1))
    app = {c: jload(R2 / "res" / c / "avenue_obj.json")["oapp"]["causal"] for c, _ in CONFIGS}
    best = max(app[c] for c in app if c.startswith("enc_"))
    share = (best - app["base"]) / (heavy - app["base"])
    if share >= 0.5:
        raise SystemExit("H1 would be supported; revise the component study")
    PROV.append(("text", "round-2 H1 share of the appearance gap closed", f"{share:.2f}",
                 f"(best enc_* oapp causal - base) / ({heavy} - base), res/<cfg>/avenue_obj.json, heavy value from "
                 f"{rel(p)} H1"))
    return bud


# --------------------------------------------------------------------------------------------- table
def table(boot, comp, bud):
    L = []
    L.append("% Generated by fig_src/make_text_v3.py -- do not edit by hand. Sources: gen/text_v3_provenance.txt")
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    L.append(r"\caption{Ablations, online micro AUC (\%). (a) Contribution of the grid branch. (b) Larger components "
             r"in the object branch, one slot changed at a time.}")
    L.append(r"\label{tab:ablation}")
    L.append(r"\footnotesize")
    L.append(r"\setlength{\tabcolsep}{3pt}")
    L.append(r"\begin{tabular}{lccc}")
    L.append(r"\toprule")
    L.append(r"\multicolumn{4}{l}{\textit{(a) Grid branch}} \\")
    L.append(r"Dataset & Object & Object + grid & Difference [95\% CI] \\")
    L.append(r"\midrule")
    for ds, o, g, d, dci, sig, nv, reps in boot:
        name = DSN[ds] + ("$^{a}$" if ds == "ped1" else "")
        diff = pts(d) + " " + ci(dci)
        if sig:
            diff = r"\textbf{" + diff + "}"
        L.append(f"{name} & {pc(o)} & {pc(g)} & {diff} \\\\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\par\vspace{6pt}")
    L.append(r"\begin{tabular}{lccccc}")
    L.append(r"\toprule")
    L.append(r"\multicolumn{6}{l}{\textit{(b) Object branch, component study}} \\")
    L.append(r"Configuration & ms$^{b}$ & Ped1$^{a}$ & Ped2 & Avenue & ShT \\")
    L.append(r"\midrule")
    for cfg, label, tot, aucs, hw, dev in comp:
        cells = [pc(aucs[ds]) + ("$^{d}$" if (cfg, ds) in MISSING_POSE else "") for ds in DS4]
        L.append(f"{label} & {tot:.1f} & " + " & ".join(cells) + r" \\")
        if cfg == "base":
            L.append(r"\midrule")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\par\vspace{3pt}")
    hw = comp[0][4]
    L.append(r"\parbox{\columnwidth}{\scriptsize (a) Paired video bootstrap of the difference (object + grid minus "
             rf"object; {boot[0][7]:,} replicates, percentile intervals), computed from the unrounded AUCs; bold: the "
             r"interval excludes zero. "
             r"(b) Every row uses the same scoring (NIW densities selected on normal data, Fisher calibration, sum, "
             r"causal median); only the named network changes. $^{a}$Development set. $^{b}$Latency per frame of the "
             r"sequential PyTorch perception pipeline (decoding, detection, pose, flow, embedding and attributes; "
             rf"density scoring and fusion not timed), batch 1, Avenue frames at {hw[1]}$\times${hw[0]}, median of "
             r"three repetitions, measured during the component study in a session whose benchmark files do not "
             r"record the GPU model; compare these values only with each other. $^{c}$Pre-registered rule: fill the "
             r"detector, flow, encoder and pose slots in turn with the most expensive option that keeps the latency "
             rf"within {bud['B3']}~ms; it chose YOLO11m, MobileCLIP2-S2 and YOLO11n-pose. The flow stayed DIS because the pre-registered flow "
             rf"alternatives, NeuFlow v2 and RAFT-large, were not run in this study; neither was the pre-registered extra "
             rf"velocity channel from ByteTrack box trajectories. No configuration met the "
             rf"stricter pre-registered budgets of {bud['B1']} and {bud['B2']}~ms. $^{{d}}$No pose model was fitted for this configuration "
             r"and dataset; velocity and appearance only. $^{e}$YOLO11n, YOLO11n-pose, DIS flow, MobileCLIP2-S0; "
             r"encoder rows replace MobileCLIP2-S0.}")
    L.append(r"\end{table}")
    (GEN / "tab_ablation_v3.tex").write_text("\n".join(L) + "\n")
    print("wrote", rel(GEN / "tab_ablation_v3.tex"))

    # supplement (supp_v4.tex, Section S8): part (b) only, relabelled; notes a-e verbatim
    S = ["% Generated by fig_src/make_text_v3.py -- do not edit by hand. Sources: gen/text_v4_provenance.txt",
         r"\begin{table}[t]", r"\centering",
         r"\caption{Component study of the object branch, online micro AUC (\%): larger networks, one slot changed at a "
         r"time.}",
         r"\label{stab:component}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{lccccc}", r"\toprule",
         r"Configuration & ms$^{b}$ & Ped1$^{a}$ & Ped2 & Avenue & ShT \\", r"\midrule"]
    for cfg, label, tot, aucs, hw_, dev in comp:
        cells = [pc(aucs[ds]) + ("$^{d}$" if (cfg, ds) in MISSING_POSE else "") for ds in DS4]
        S.append(f"{label} & {tot:.1f} & " + " & ".join(cells) + r" \\")
        if cfg == "base":
            S.append(r"\midrule")
    S += [r"\bottomrule", r"\end{tabular}", r"\par\vspace{3pt}"]
    S.append(r"\parbox{\columnwidth}{\scriptsize "
             r"Every row uses the same scoring (NIW densities selected on normal data, Fisher calibration, sum, "
             r"causal median); only the named network changes. $^{a}$Development set. $^{b}$Latency per frame of the "
             r"sequential PyTorch perception pipeline (decoding, detection, pose, flow, embedding and attributes; "
             rf"density scoring and fusion not timed), batch 1, Avenue frames at {hw[1]}$\times${hw[0]}, median of "
             r"three repetitions, measured during the component study in a session whose benchmark files do not "
             r"record the GPU model; compare these values only with each other. $^{c}$Pre-registered rule: fill the "
             r"detector, flow, encoder and pose slots in turn with the most expensive option that keeps the latency "
             rf"within {bud['B3']}~ms; it chose YOLO11m, MobileCLIP2-S2 and YOLO11n-pose. The flow stayed DIS because the pre-registered flow "
             rf"alternatives, NeuFlow v2 and RAFT-large, were not run in this study; neither was the pre-registered extra "
             rf"velocity channel from ByteTrack box trajectories. No configuration met the "
             rf"stricter pre-registered budgets of {bud['B1']} and {bud['B2']}~ms. $^{{d}}$No pose model was fitted for this configuration "
             r"and dataset; velocity and appearance only. $^{e}$YOLO11n, YOLO11n-pose, DIS flow, MobileCLIP2-S0; "
             r"encoder rows replace MobileCLIP2-S0.}")
    S.append(r"\end{table}")
    (GEN / "supp_tab_component_v4.tex").write_text("\n".join(S) + "\n")
    print("wrote", rel(GEN / "supp_tab_component_v4.tex"))
    PROV4.append(("supp_tab_component_v4.tex", "rows and notes", "identical to part (b) of tab_ablation_v3.tex",
                  "same variables (text_v3_provenance.txt, 'ablation' entries)"))


def table_online_v4(boot):
    """gen/tab_online_v4.tex (main_v4.tex): Table tab:online of v3 (cells from gen/online_cells_v4.json, written by
    make_tables_v3.py) merged with the grid-branch ablation, part (a) of tab_ablation_v3.tex (grid_bootstrap())."""
    p = GEN / "online_cells_v4.json"
    if not p.exists():
        raise SystemExit("gen/online_cells_v4.json not found: run make_tables_v3.py first")
    cells = {(c["config"], c["protocol"], c["dataset"]): c["value"] for c in jload(p)}
    if len(cells) != 24:
        raise SystemExit(f"{p}: expected 24 cells")
    # the online micro cells must be the bootstrap point values, rounded to one decimal
    for ds, o, g, *_ in boot:
        if cells[("obj", "online", ds)] != pc(o) or cells[("grid", "online", ds)] != pc(g):
            raise SystemExit(f"{ds}: online micro cells of {p.name} differ from the bootstrap point values")
    L = ["% Generated by fig_src/make_text_v3.py -- do not edit by hand. Sources: gen/text_v4_provenance.txt",
         r"\begin{table}[t]", r"\centering",
         r"\caption{Our two configurations under the standard and the online protocol, frame AUC (\%), and the effect of "
         r"the grid branch. Online: strictly causal scoring (no future frames, no per-video normalisation, median of the "
         r"last 9 frame scores; the first 4 frames of each video are not scored).}",
         # 2pt: the interval row is the widest; at 4pt the table was 15.6pt wider than the column
         r"\label{tab:online}", r"\footnotesize", r"\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{lcccc}",
         r"\toprule", r"Protocol & Ped1$^{a}$ & Ped2 & Avenue & ShT \\", r"\midrule"]
    for br, lab in (("obj", "Object branch"), ("grid", "Object + grid")):
        L.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{lab}}}}} \\")
        for proto, name in (("standard", "Standard, micro"), ("online", "Online, micro"),
                            ("online_macro", "Online, macro")):
            L.append(" & ".join([r"\quad " + name] + [cells[(br, proto, ds)] for ds in DS4]) + r" \\")
        L.append(r"\midrule")
    L.append(r"\multicolumn{5}{l}{\textit{Grid effect, online micro (object + grid $-$ object)}} \\")
    L.append(" & ".join([r"\quad Difference"] + [(r"\textbf{" + pts(d) + "}") if sig else pts(d)
                                                   for ds, o, g, d, dci, sig, nv, reps in boot]) + r" \\")
    L.append(" & ".join([r"\quad 95\% CI"] + [r"{\scriptsize " + ci(dci) + "}"
                                              for ds, o, g, d, dci, sig, nv, reps in boot]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\par\vspace{3pt}",
          r"\parbox{\columnwidth}{\scriptsize Micro: AUC over all test frames pooled; macro: mean of the per-video AUCs "
          r"over the test videos that contain normal and anomalous frames (online protocol only). Protocols: "
          r"Section~\ref{sec:protocols}. PyTorch detectors. Grid effect: paired video bootstrap of the difference "
          rf"({boot[0][7]:,} replicates, percentile intervals), computed from the unrounded AUCs; bold: the interval "
          r"excludes zero. $^{a}$Development set.}", r"\end{table}"]
    # the caption constants (median length 9, 4 unscored frames) are those of tab_online_v3.tex
    v3 = (GEN / "tab_online_v3.tex").read_text()
    if "median of the last 9 frame scores; the first 4 frames of each video are not scored" not in v3:
        raise SystemExit("tab_online_v3.tex caption changed: revise the v4 caption")
    if [ds for ds, *_ in boot] != DS4:
        raise SystemExit("bootstrap rows not in dataset order")
    (GEN / "tab_online_v4.tex").write_text("\n".join(L) + "\n")
    print("wrote", rel(GEN / "tab_online_v4.tex"))
    PROV4.append(("tab_online_v4.tex", "24 AUC cells", "as tab_online_v3.tex", f"{rel(p)} (make_tables_v3.py)"))
    for ds, o, g, d, dci, sig, nv, reps in boot:
        PROV4.append(("tab_online_v4.tex", f"grid effect {ds}", f"{pts(d)} {ci(dci)}{' (bold)' if sig else ''}",
                      f"kanit/r2_20261105/boot/{ds}.json : grid+obj_vs_obj (as tab_ablation_v3.tex part a)"))
    PROV4.append(("tab_online_v4.tex", "median length, unscored frames", "9, 4",
                  "caption of gen/tab_online_v3.tex (make_tables_v3.py protocol_constants())"))


def main():
    boot = grid_bootstrap()
    comp = component_study()
    channels()
    hbpot()
    far()
    dataeff()
    avenue_labels()
    gaps()
    protocols()
    from_tables_provenance()
    median_delay()
    literature()
    code_constants()
    bud = preregistration()
    table(boot, comp, bud)
    # budgets of the component study: B1/B2 had no configuration (stated in the table note and, for B2 / H2, in the
    # component study), checked here
    lat = [r[2] for r in comp]
    if min(lat) <= bud["B2"] or bud["B1"] > bud["B2"]:
        raise SystemExit("a configuration fits the B2 budget: the table note and the H2 sentence are wrong")
    GEN.mkdir(exist_ok=True)
    head = "% Generated by fig_src/make_text_v3.py -- do not edit by hand. Sources: gen/text_v3_provenance.txt\n"
    (GEN / "numbers_text_v3.tex").write_text(head + "".join(f"\\newcommand{{\\{n}}}{{{v}}}\n" for n, v in MACROS))
    print("wrote", rel(GEN / "numbers_text_v3.tex"))
    lines = ["# Provenance of every number in gen/numbers_text_v3.tex and gen/tab_ablation_v3.tex "
             "(written by fig_src/make_text_v3.py)", "# table | label | value | source"]
    lines += [f"{t:7} | {l:40} | {v} | {s}" for t, l, v, s in PROV]
    (GEN / "text_v3_provenance.txt").write_text("\n".join(lines) + "\n")
    print("wrote", rel(GEN / "text_v3_provenance.txt"))
    table_online_v4(boot)
    lines = ["# Provenance of the v4 outputs of fig_src/make_text_v3.py: gen/tab_online_v4.tex (main_v4.tex), "
             "gen/supp_tab_component_v4.tex (supp_v4.tex), and the pre-registration citations checked in supp_v4.tex",
             "# output | label | value | source"]
    lines += [f"{t:26} | {l:40} | {v} | {s}" for t, l, v, s in PROV4]
    (GEN / "text_v4_provenance.txt").write_text("\n".join(lines) + "\n")
    print("wrote", rel(GEN / "text_v4_provenance.txt"))


if __name__ == "__main__":
    main()
