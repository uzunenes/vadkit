"""Tables and number macros of the reviewer-driven measurements (docs/tez_notlari/hakem_duzeltmeleri_on_kayit_2026-11-06.md).

Inputs (kanit/hakem_20261106/): s1/<ds>.json (vad_bench/vast/runs_hakem/ablation.py run), cost/<ds>.json (ablation.py
cost). Outputs: gen/tab_scoring_v3.tex (S1 scoring ablation), gen/tab_hyper_v3.tex (selected NIW hyper-parameters),
gen/numbers_hakem_v3.tex (macros \\Vh...), gen/hakem_v3_provenance.txt (sha256 of every input).
v4 outputs (main_v4.tex, supp_v4.tex), from the same values; the v3 files stay unchanged: gen/tab_scoring_v4.tex
(compact, online micro AUC and scoring cost), gen/supp_tab_scoring_v4.tex, gen/supp_tab_latency_v4.tex and
gen/supp_tab_hyper_v4.tex (the v3 tables relabelled stab:...), listed in gen/hakem_v4_provenance.txt.
Rounding: AUCs in percent with one decimal, from the unrounded values; a row is marked when the 95% paired video
bootstrap interval of its difference to the NIW row excludes zero (up: higher than NIW, down: lower).
Run: python3 fig_src/make_hakem_v3.py [--partial]   (--partial: allow missing datasets, for checking only)
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
K = ROOT / "kanit" / "hakem_20261106"
GEN = ROOT / "gen"
DS = [("ped1", "Ped1", "PedOne"), ("ped2", "Ped2", "PedTwo"), ("avenue", "Avenue", "Avenue"),
      ("shanghaitech", "ShT", "SHT")]
ROWS = [  # code, label, macro stem
    ("D1", "NIW mixture (ours)", "Niw"),
    ("D2b", "Same components, no prior", "Prior"),
    ("D2", "ML Gaussian mixture, same $K$", "Gmm"),
    ("D3", "Single Gaussian", "Gauss"),
    ("D4", "1-NN distance (memory bank)", "Knn"),
    ("C1", "z-score calibration", "Zcal"),
    ("C2", "Maximum fusion", "Maxf"),
    ("T1", "No causal median", "Nomed"),
    ("L-ovel", "Without velocity", "NoVel"),
    ("L-opose", "Without pose", "NoPose"),
    ("L-oapp", "Without appearance", "NoApp"),
    ("K1", "With velocity magnitude$^{\\dagger}$", "Mag"),
    ("A-ovel", "Velocity alone", "OnlyVel"),       # the single-attribute rows of D1 (S1: given reproduced, no test)
    ("A-opose", "Pose alone", "OnlyPose"),
    ("A-oapp", "Appearance alone", "OnlyApp"),
]
GROUPS = [("Density or distance (same attributes, calibration and fusion)", ["D1", "D2b", "D2", "D3", "D4"]),
          ("Calibration, fusion and smoothing (NIW densities)", ["C1", "C2", "T1"]),
          ("Attributes (NIW densities)", ["A-ovel", "A-opose", "A-oapp", "L-ovel", "L-opose", "L-oapp", "K1"])]
ATTR = [("ovel", "Velocity"), ("opose", "Pose"), ("oapp", "Appearance")]


def pct(x):
    return f"{100 * x:.1f}"


def sg(x):
    """Signed points (x is an AUC difference) with a typographic minus, as make_tables_v3.signed."""
    return ("$-$" if x < 0 else "$+$") + f"{abs(100 * x):.1f}"


def spearman(x, y):
    """Spearman rank correlation with average ranks for ties (no SciPy on the writing machine)."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return cov / (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5


def quantile(v, q):
    """Linear interpolation between order statistics (NumPy's default)."""
    s = sorted(v)
    h = (len(s) - 1) * q
    lo = int(h)
    return s[lo] + (h - lo) * (s[min(lo + 1, len(s) - 1)] - s[lo])


def latency(mac, prov):
    """G3 (latency spread of the object branch, four datasets, two back-ends) and G1 (grid branch alone) of the
    pre-registered S2 measurements -> gen/tab_latency_v3.tex."""
    V = K / "vast_out" / "out"
    import statistics as st
    rows, loads, rng = [], [], {}
    for ds, lab, dm in [DS[1], DS[2], DS[3], DS[0]]:
        for be, blab in (("trt", "TensorRT"), ("pt", "PyTorch")):
            p = V / "g3" / f"{ds}_{be}.json"
            d = json.loads(p.read_text())
            prov.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT)}")
            if not d["parallel_flow"] or d["warmup_frames"] != 20 or d["frames_per_rep"] != 100 or len(d["reps"]) != 3:
                raise SystemExit(f"{p}: not the pre-registered setting (parallel flow, 20 warm-up, 3 x 100 frames)")
            t, o = d["timed_frames"]["total_ms"], d["timed_frames"]["objects"]
            if len(t) != 300:
                raise SystemExit(f"{p}: {len(t)} timed frames, expected 300")
            idle = (V / "g3" / f"idle_{ds}_{be}.txt").read_text().splitlines()
            loads.append(float(idle[-1].split("load average:")[1].split(",")[0]))
            med = st.median(r["total"]["median_ms"] for r in d["reps"])
            p50, p95, p99 = (quantile(t, q) for q in (0.5, 0.95, 0.99))
            rho = spearman(t, o)
            key = f"{'Trt' if be == 'trt' else 'Pt'}{dm}"
            mac(f"LatFifty{key}", f"{p50:.1f}")
            mac(f"LatNinetyNine{key}", f"{p99:.1f}")
            mac(f"LatRho{key}", f"{rho:.2f}")
            mac(f"LatFps{key}", f"{1000 / med:.1f}")
            obj = d["objects_per_frame"]
            first = f"{lab}$^{{c}}$" if ds == "ped1" else lab
            name = f"{first} ({obj['mean']:.1f}, {obj['max']})" if be == "trt" else ""
            rows.append(f"{name} & {blab} & {p50:.1f} & {p95:.1f} & {p99:.1f} & {max(t):.1f} & {rho:.2f} \\\\")
            for stat, v in (("Fifty", p50), ("NinetyNine", p99)):
                rng.setdefault(f"Lat{stat}{'Trt' if be == 'trt' else 'Pt'}", []).append(v)
            rng.setdefault("LatRho", []).append(rho)
        rows.append("\\addlinespace[2pt]")
    rows.pop()
    for k, v in rng.items():
        d_ = 2 if k == "LatRho" else 1
        mac(f"{k}Min", f"{min(v):.{d_}f}")
        mac(f"{k}Max", f"{max(v):.{d_}f}")
    grid = []
    for ds, lab, dm in (DS[1], DS[2]):
        cells = []
        for dev in ("cuda", "cpu"):
            p = V / "g1" / f"{ds}_{dev}.json"
            d = json.loads(p.read_text())
            prov.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT)}")
            if d["arguments"]["frames"] != 100 or len(d["repetitions"]) != 3:
                raise SystemExit(f"{p}: not 3 x 100 frames")
            med = st.median(r["total"] for r in d["repetitions"])
            p95 = st.median(r["p95_total_ms"] for r in d["repetitions"])
            ae = st.median(r["ae_frx"] for r in d["repetitions"])
            tag = f"{'Gpu' if dev == 'cuda' else 'Cpu'}{dm}"
            mac(f"GridMs{tag}", f"{med:.1f}")
            mac(f"GridAeMs{tag}", f"{ae:.1f}")
            mac(f"GridFps{tag}", f"{1000 / med:.1f}")
            cells += [f"{med:.1f}", f"{p95:.1f}"]
            idle = (V / "g1" / f"idle_{ds}_{dev}.txt").read_text().splitlines()
            loads.append(float(idle[-1].split("load average:")[1].split(",")[0]))
        h_, w_ = d["frame"].split("x")
        frame = f"{w_}$\\times${h_}"
        grid.append(f"{lab} ({frame}) & " + " & ".join(cells) + " \\\\")
    mac("LatLoadMin", f"{min(loads):.0f}")
    mac("LatLoadMax", f"{max(loads):.0f}")
    # (c) E1 addendum (e1_pozsuz_hiz_olcumu_on_kayit_2026-11-06.md): with and without the pose model, same session
    G4 = K / "g4"
    env4 = json.loads((G4 / "env.json").read_text())
    if "3090" not in env4["gpu"]:
        raise SystemExit("G4: not an RTX 3090")
    pose_rows, deltas, loads4, fits, util4 = [], {}, [], {}, []
    for ds, lab, dm in (DS[1], DS[3], DS[0]):
        for be, blab in (("trt", "TensorRT"), ("pt", "PyTorch")):
            v = {}
            for p_ in ("pose", "nopose"):
                f = G4 / f"{ds}_{p_}_{be}.json"
                d = json.loads(f.read_text())
                prov.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(ROOT)}")
                if not d["parallel_flow"] or d["warmup_frames"] != 20 or len(d["timed_frames"]["total_ms"]) != 300:
                    raise SystemExit(f"{f}: not the G3 setting")
                if (p_ == "nopose") != (st.median(r["pose"]["median_ms"] for r in d["reps"]) == 0.0):
                    raise SystemExit(f"{f}: pose stage does not match the configuration")
                t = d["timed_frames"]["total_ms"]
                v[p_] = {"p50": quantile(t, 0.5), "p95": quantile(t, 0.95), "p99": quantile(t, 0.99),
                         "fps": 1000 / st.median(r["total"]["median_ms"] for r in d["reps"]),
                         "flow": st.median(r["flow"]["median_ms"] for r in d["reps"])}
                idle = (G4 / f"idle_{ds}_{p_}_{be}.txt").read_text().splitlines()
                loads4.append(float(idle[-1].split("load average:")[1].split(",")[0]))
                util4.append(int(idle[1].split("%")[0]))
            dlt = round(v["pose"]["p50"], 1) - round(v["nopose"]["p50"], 1)     # of the values as printed
            deltas[(ds, be)] = dlt
            first = (f"{lab}$^{{c}}$" if ds == "ped1" else lab) if be == "trt" else ""
            pose_rows.append(f"{first} & {blab} & {v['pose']['p50']:.1f} & {v['pose']['p95']:.1f} & "
                             f"{v['pose']['p99']:.1f} & {v['nopose']['p50']:.1f} & {v['nopose']['p95']:.1f} & "
                             f"{v['nopose']['p99']:.1f} & {dlt:.1f} \\\\")
            tag = f"{'Trt' if be == 'trt' else 'Pt'}{dm}"
            mac(f"NoPoseFps{tag}", f"{v['nopose']['fps']:.1f}")
            mac(f"PoseFpsGfour{tag}", f"{v['pose']['fps']:.1f}")
            mac(f"NoPoseFlowWait{tag}", f"{v['nopose']['flow']:.1f}")
            mac(f"PoseFlowWait{tag}", f"{v['pose']['flow']:.1f}")
        for p_ in ("pose", "nopose"):
            fits[(ds, p_)] = float((G4 / f"fit_{ds}_{p_}.log").read_text().split("ELAPSED_S")[-1])
        pose_rows.append("\\addlinespace[2pt]")
    pose_rows.pop()
    ucsd = [deltas[(d, b)] for d in ("ped1", "ped2") for b in ("trt", "pt")]
    sht = [deltas[("shanghaitech", b)] for b in ("trt", "pt")]
    mac("NoPoseSaveUcsdMin", f"{min(ucsd):.1f}")
    mac("NoPoseSaveUcsdMax", f"{max(ucsd):.1f}")
    mac("NoPoseSaveShtMax", f"{max(sht):.1f}")
    mac("GfourLoadMax", f"{max(loads4):.1f}")
    mac("GfourPoseTrtPedTwo", f"{[r for r in pose_rows if r.startswith('Ped2 & TensorRT')][0].split(' & ')[2]}")
    fp = [fits[(d, "pose")] for d in ("ped1", "ped2")]
    fn = [fits[(d, "nopose")] for d in ("ped1", "ped2")]
    mac("GfourFitPoseMin", f"{min(fp):.1f}")
    mac("GfourFitPoseMax", f"{max(fp):.1f}")
    mac("GfourFitNoPoseMin", f"{min(fn):.1f}")
    mac("GfourFitNoPoseMax", f"{max(fn):.1f}")
    mac("GfourFitPoseSHT", f"{fits[('shanghaitech', 'pose')]:.1f}")
    mac("GfourFitNoPoseSHT", f"{fits[('shanghaitech', 'nopose')]:.1f}")
    cpu4 = env4["cpu"].replace(" with Radeon Graphics", "")
    util_all = [int((f.read_text().splitlines()[1]).split("%")[0]) for f in sorted(G4.glob("idle_*.txt"))]
    mac("GfourBusy", str(sum(u > 5 for u in util_all)))
    mac("GfourChecks", str(len(util_all)))
    t = ["% Generated by fig_src/make_hakem_v3.py -- do not edit by hand.", "\\begin{table}[t]", "\\centering",
         "\\caption{Latency per frame (ms): (a) object branch on the four datasets, (b) grid branch alone, (c) object "
         "branch with and without the pose model.}",
         "\\label{tab:latency}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
         "\\begin{tabular}{llccccc}", "\\toprule",
         "\\multicolumn{7}{l}{\\textit{(a) Object branch, decoding to fused score, parallel flow}} \\\\",
         "Dataset (objects) & YOLO & p50 & p95 & p99 & max & $\\rho$ \\\\", "\\midrule", *rows, "\\bottomrule",
         "\\end{tabular}", "\\par\\vspace{6pt}", "\\begin{tabular}{lcccc}", "\\toprule",
         "\\multicolumn{5}{l}{\\textit{(b) Grid branch alone, flow to calibrated cell scores}} \\\\",
         " & \\multicolumn{2}{c}{GPU} & \\multicolumn{2}{c}{CPU, 4 threads} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}", "Dataset (frame) & median & p95 & median & p95 \\\\", "\\midrule",
         *grid, "\\bottomrule", "\\end{tabular}", "\\par\\vspace{6pt}", "\\setlength{\\tabcolsep}{2.2pt}",
         "\\begin{tabular}{llccccccc}", "\\toprule",
         "\\multicolumn{9}{l}{\\textit{(c) Object branch with and without the pose model, third session}} \\\\",
         " & & \\multicolumn{3}{c}{with pose} & \\multicolumn{3}{c}{without} & \\\\",
         "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}", "Dataset & YOLO & p50 & p95 & p99 & p50 & p95 & p99 & saved \\\\",
         "\\midrule",
         *pose_rows, "\\bottomrule", "\\end{tabular}", "\\par\\vspace{3pt}",
         "\\parbox{\\columnwidth}{\\scriptsize Batch size 1, test frames in time order, RTX 3090 and AMD EPYC 7452, "
         "PyTorch 2.4.1, one session on a host shared with other tenants (1-min load average "
         f"{min(loads):.0f}--{max(loads):.0f} before the runs; GPU idle). (a) 20 warm-up frames, then 3$\\times$100 "
         "timed frames; percentiles over the 300 frames. Objects: mean and maximum number of objects per timed "
         "frame. $\\rho$: Spearman correlation between the latency and the number of objects of a frame. TensorRT: "
         "FP16 engines for the two YOLO11n models, the image encoder in PyTorch, as in Table~\\ref{tab:speed}; "
         "PyTorch: all networks in PyTorch. Avenue in this session versus Table~\\ref{tab:speed} (another session "
         "and host): see the text. $^{c}$Development set. (b) \\texttt{tools/bench\\_pipeline.py}: DIS flow, "
         "autoencoder, cell statistics, tail calibration, fusion and causal median, without the object branch; "
         "3$\\times$100 timed frames after 5 warm-up frames each, median over the three repetitions of the per-"
         "repetition median and 95th percentile; frame size (width$\\times$height) as processed by the grid branch "
         "(Section~\\ref{sec:grid}). This is not the latency of "
         "the object + grid configuration: the two branches are not combined in one streaming implementation. "
         f"(c) As (a), in a third session on an RTX 3090 with an {cpu4} CPU (1-min load average at most "
         f"{max(loads4):.1f}), with the configurations of Table~\\ref{{tab:main}} with and without the pose model, "
         f"alternating; objects per frame as in (a); saved: difference of the p50 values. The GPU was not idle: "
         f"before {sum(u > 5 for u in util_all)} of the {len(util_all)} measurements (fits included) nvidia-smi "
         f"reported {min(u for u in util_all if u > 5)}--{max(util_all)}\\% utilisation with no process listed, "
         "and a display was attached. The absolute values are therefore not comparable with (a).}",
         "\\end{table}"]
    (GEN / "tab_latency_v3.tex").write_text("\n".join(t) + "\n")


def supp_copy(src, dst, subs, prov4):
    """A v3 table copied for the supplement (supp_v4.tex) with its label and references re-pointed; each replaced
    string must occur exactly once."""
    t = (GEN / src).read_text()
    for old, new in subs:
        if t.count(old) != 1:
            raise SystemExit(f"{src}: {old!r} found {t.count(old)} times, expected once")
        t = t.replace(old, new)
    (GEN / dst).write_text(t)
    prov4.append(f"{dst}: {src} verbatim, " + "; ".join(f"{o} -> {n}" for o, n in subs))


def scoring_v4(lines_v3, v4cells, v4cost, label, prov4):
    """gen/tab_scoring_v4.tex (main_v4.tex): single column, online micro AUC only (the pre-registered metric, with the
    same arrows) and the scoring cost; the cells are those of the v3 table, checked against its rows."""
    order = [["D1", "D2b", "D2", "D3", "D4"], ["C1", "C2", "T1"], ["L-ovel", "L-opose", "L-oapp"]]
    head = " & ".join(lab for ds, lab, _ in DS)
    t = ["% Generated by fig_src/make_hakem_v3.py -- do not edit by hand.", "\\begin{table}[t]", "\\centering",
         "\\caption{Scoring ablation of the object branch: online micro AUC (\\%), the pre-registered metric, and "
         "scoring cost.}",
         "\\label{tab:scoring}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{lcccc}",
         "\\toprule", f"Variant & {head} \\\\", "\\midrule"]
    for grp in order:
        for c in grp:
            if len(v4cells.get(c, [])) != 4:
                raise SystemExit(f"{c}: online micro cells missing for the v4 scoring table")
            name = f"\\textbf{{{label[c]}}}" if c == "D1" else label[c]
            row = f"{name} & " + " & ".join(v4cells[c]) + " \\\\"
            # the same cells, in the same order, must start the row of the v3 table
            v3 = [x for x in lines_v3 if x.startswith(f"{name} & ")]
            if len(v3) != 1 or not v3[0].startswith(row[:-3]):
                raise SystemExit(f"{c}: v4 cells differ from the v3 table")
            t.append(row)
        t.append("\\midrule")
    if len(v4cost) != 4:
        raise SystemExit("v4 scoring table: expected 4 cost rows")
    for lab, cells in v4cost:
        t.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    t += ["\\bottomrule", "\\end{tabular}", "\\par\\vspace{3pt}",
          "\\parbox{\\columnwidth}{\\scriptsize Each row changes one step of the NIW row and keeps the rest "
          "(attribute rows, training subsample, cross-fitted normal reference, frame floor); ShT models are fitted per "
          "scene. Same components, no prior: the $k$-means components of the NIW mixture with maximum-likelihood "
          "Gaussians. ML Gaussian mixture: $K$ of the NIW selection, full covariance. 1-NN: Euclidean distance to the "
          "nearest training object. z-score: each channel standardised by its normal reference scores. Maximum "
          "fusion: the largest calibrated channel score. $^{\\uparrow}$/$^{\\downarrow}$ (the pre-registered test): "
          "the 95\\% paired video bootstrap interval of the difference to the NIW row (2,000 replicates, seed 0) lies "
          "above / below zero. Cost: 4 CPU threads, one scene's models on ShT, on the shared host of Supplementary "
          "Table~\\ref{stab:latency}a. Macro and standard AUCs, single-attribute, velocity-magnitude and "
          "quarter-data rows: Supplementary Table~\\ref{stab:scoring}. All rows were specified before the runs.}",
          "\\end{table}"]
    (GEN / "tab_scoring_v4.tex").write_text("\n".join(t) + "\n")
    prov4.append("tab_scoring_v4.tex: the online micro cells (with arrows) and the cost cells of tab_scoring_v3.tex, "
                 "same values, checked row by row")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args()
    s1, cost, prov = {}, {}, []
    prov4 = []              # v4 outputs: gen/hakem_v4_provenance.txt (gen/hakem_v3_provenance.txt stays unchanged)
    for ds, _, _ in DS:
        for kind, store in (("s1", s1), ("cost", cost)):
            p = K / kind / f"{ds}.json"
            if not p.exists():
                if a.partial:
                    continue
                raise SystemExit(f"missing {p}")
            store[ds] = json.loads(p.read_text())
            prov.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT)}")
    for ds, d in s1.items():
        if d["reproduction"]["abs_diff"] > 1e-6:
            raise SystemExit(f"{ds}: D1 does not reproduce the archived result")
        for r in d.get("F1", []):
            if r["D1_abs_diff"] > 1e-6:
                raise SystemExit(f"{ds}: F1 seed {r['seed']} does not reproduce the data-efficiency draw")
        if len(d.get("F1", [])) != 5 and not a.partial:
            raise SystemExit(f"{ds}: F1 has {len(d.get('F1', []))} draws, expected 5")
    macros = []
    mac = lambda name, val: macros.append(f"\\newcommand{{\\Vh{name}}}{{{val}}}")

    # KISS-1 (kiss1_hiz_buyuklugu_on_kayit_2026-11-06.md): a fourth attribute, the speed of an object; its base fit must
    # be the D1 row of S1. Shown as row K1 of the attribute block; object + grid only as macros.
    for ds, _, dm in DS:
        p = K / "kiss1" / f"{ds}.json"
        k = json.loads(p.read_text())
        prov.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT)}")
        fk = lambda r: r[max((x for x in r if "+" in x), key=lambda x: x.count("+"))]
        ck = k["checks"]
        if not ck["first_three_channels_identical"] or ck["obj_reproduces_archived"]["abs_diff"] > 1e-6 \
                or ck["grid_reproduces_archived"]["abs_diff"] > 1e-6:
            raise SystemExit(f"{ds}: KISS-1 failed its own reproduction checks")
        if ds in s1:
            if abs(fk(k["results"]["base"]["obj"])["causal"] - s1[ds]["rows"]["D1"]["fused"]["causal"]) > 1e-6:
                raise SystemExit(f"{ds}: KISS-1 base is not the S1 D1 row")
            s1[ds]["rows"]["K1"] = {"fused": fk(k["results"]["kiss1"]["obj"])}
            s1[ds]["boot"]["K1"] = k["boot"]["obj"]
        b = k["boot"]["obj"]["difference"]
        mac(f"Mag{dm}", pct(fk(k["results"]["kiss1"]["obj"])["causal"]))
        mac(f"MagDiff{dm}", sg(b["point"]))
        mac(f"MagCI{dm}", f"[{sg(b['ci95'][0])}, {sg(b['ci95'][1])}]")
        b = k["boot"]["grid"]["difference"]
        mac(f"MagGridDiff{dm}", sg(b['point']))
        mac(f"MagGridLo{dm}", sg(b['ci95'][0]))
        mac(f"MagGridHi{dm}", sg(b['ci95'][1]))
        mac(f"MagAlone{dm}", pct(k["results"]["kiss1"]["obj"]["omag"]["causal"]))

    # ---------------------------------------------------------------- scoring table
    # three metric groups (pre-registered: online micro AUC; also online macro AUC and the standard protocol); the
    # bootstrap marks (significance rule) belong to the online micro AUC only
    import statistics as st
    head = " & ".join(lab for ds, lab, _ in DS)
    METRICS = [("causal", "Online, micro"), ("causal_macro", "Online, macro"), ("offline", "Standard, micro")]
    NC = 1 + 4 * len(METRICS)
    lines = ["% Generated by fig_src/make_hakem_v3.py -- do not edit by hand.",
             "\\begin{table*}[t]", "\\centering",
             "\\caption{Scoring ablation of the object branch (three attributes unless stated): frame AUC (\\%) under the online "
             "protocol (micro and macro) and the standard protocol.}",
             "\\label{tab:scoring}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{l" + "cccc" * len(METRICS) + "}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{m}}}" for _, m in METRICS) + " \\\\",
             "".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(METRICS))),
             "Variant & " + " & ".join([head] * len(METRICS)) + " \\\\", "\\midrule"]
    code_stem = {c: s_ for c, _, s_ in ROWS}
    label = {c: l for c, l, _ in ROWS}
    worse_other = {}        # (code, metric) -> datasets where the variant's point value is above NIW's
    v4cells = {}            # code -> online micro cells (with the arrows) for gen/tab_scoring_v4.tex
    for title, codes in GROUPS:
        lines.append(f"\\multicolumn{{{NC}}}{{l}}{{\\textit{{{title}}}}} \\\\")
        for c in codes:
            cells = []
            for key, _ in METRICS:
                for ds, _, dm in DS:
                    if ds not in s1:
                        cells.append("--")
                        continue
                    if c.startswith("A-"):      # reproduced single-attribute rows, no bootstrap
                        v = s1[ds]["rows"]["D1"]["channels"][c[2:]][key]
                        cells.append(pct(v))
                        if key == "causal":
                            mac(f"{code_stem[c]}{dm}", pct(v))
                        continue
                    v = s1[ds]["rows"][c]["fused"][key]
                    cell = pct(v)
                    if key == "causal":
                        if c != "K1":       # KISS-1 macros come from its own files, for every dataset
                            mac(f"{code_stem[c]}{dm}", cell)
                        if c != "D1":
                            b = s1[ds]["boot"][c]["difference"]
                            lo, hi = b["ci95"]
                            cell += "$^{\\uparrow}$" if lo > 0 else ("$^{\\downarrow}$" if hi < 0 else "")
                            if c != "K1":
                                mac(f"{code_stem[c]}Diff{dm}", sg(b['point']))
                                mac(f"{code_stem[c]}Lo{dm}", sg(lo))
                                mac(f"{code_stem[c]}Hi{dm}", sg(hi))
                                mac(f"{code_stem[c]}CI{dm}", f"[{sg(lo)}, {sg(hi)}]")
                    elif c != "D1" and v > s1[ds]["rows"]["D1"]["fused"][key]:
                        worse_other.setdefault((c, key), []).append(ds)
                    cells.append(cell)
                    if key == "causal":
                        v4cells.setdefault(c, []).append(cell)
            name = f"\\textbf{{{label[c]}}}" if c == "D1" else label[c]
            lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    # secondary metrics in the text (point values, no intervals): checked here so that the text cannot drift
    dens = ["D1", "D2b", "D2", "D3", "D4"]
    dsn = dict((d, l) for d, l, _ in DS)
    for ds, _, dm in DS:
        if ds in s1:
            for c in dens:
                for key, tag in (("causal_macro", "MacroVal"), ("offline", "StdVal")):
                    mac(f"{code_stem[c]}{tag}{dm}", pct(s1[ds]["rows"][c]["fused"][key]))
    if len(s1) == 4:
        std_best = [d for d in s1 if max(dens, key=lambda c: s1[d]["rows"][c]["fused"]["offline"]) == "D1"]
        full = {"ped1": "Ped1", "ped2": "Ped2", "avenue": "Avenue", "shanghaitech": "ShanghaiTech"}
        nb = [full[d] for d, _, _ in DS if d in std_best]
        mac("NiwStdBest", ", ".join(nb[:-1]) + " and " + nb[-1])
        if sorted(set(s1) - set(std_best)) != ["ped2"] or \
                max(dens, key=lambda c: s1["ped2"]["rows"][c]["fused"]["offline"]) != "D4":
            raise SystemExit("standard protocol: NIW best except 1-NN on Ped2 no longer holds; revise the text")
        if any(max(dens, key=lambda c: s1[d]["rows"][c]["fused"]["causal_macro"]) == "D1" for d in s1):
            raise SystemExit("macro AUC: NIW highest on some dataset; revise the text")
    # the density rows whose macro or standard point value exceeds NIW somewhere (for the text)
    for c in ("D2b", "D2", "D3", "D4"):
        for key, tag in (("causal_macro", "Macro"), ("offline", "Std")):
            ds_ = worse_other.get((c, key), [])
            full = {"ped1": "Ped1", "ped2": "Ped2", "avenue": "Avenue", "shanghaitech": "ShanghaiTech"}
            names = [full[d] for d, _, _ in DS if d in ds_]
            mac(f"{code_stem[c]}Above{tag}", (", ".join(names[:-1]) + " and " + names[-1]) if len(names) > 1
                else (names[0] if names else "none"))
    # f = 0.25 block
    lines.append(f"\\multicolumn{{{NC}}}{{l}}{{\\textit{{A quarter of the normal training videos (mean over 5 draws; "
                 f"online micro: mean $\\pm$ s.d.)}}}} \\\\")
    for code, lab, stem in (("D1", "NIW mixture (ours)", "Niw"), ("D2", "ML Gaussian mixture", "Gmm"),
                            ("D4", "1-NN distance", "Knn")):
        cells = []
        for key, _ in METRICS:
            for ds, _, dm in DS:
                if ds not in s1 or len(s1[ds].get("F1", [])) != 5:      # the latter with --partial only
                    cells.append("--")
                    continue
                xs = [100 * r[code][key] for r in s1[ds]["F1"]]
                m, sd = st.mean(xs), st.stdev(xs)
                cells.append(f"{m:.1f}$\\pm${sd:.1f}" if key == "causal" else f"{m:.1f}")
                if key == "causal":
                    mac(f"Quarter{stem}{dm}", f"{m:.1f}")
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    # directions stated in the text for the quarter-data block (means of the five draws, online micro, no intervals)
    if len(s1) == 4 and all(len(s1[d].get("F1", [])) == 5 for d in s1):
        qm = {(d, c): st.mean(r[c]["causal"] for r in s1[d]["F1"]) for d in s1 for c in ("D1", "D2", "D4")}
        for d, _, dm in DS:
            for c, stem in (("D1", "Niw"), ("D2", "Gmm"), ("D4", "Knn")):
                full_v = s1[d]["rows"][c]["fused"]["causal"]
                mac(f"QuarterDiff{stem}{dm}", sg(qm[(d, c)] - full_v))
        small = max(abs(qm[(d, c)] - s1[d]["rows"][c]["fused"]["causal"]) for d in ("ped1", "ped2", "avenue")
                    for c in ("D1", "D2", "D4"))
        mac("QuarterMaxAbsSmall", f"{100 * small:.1f}")
        if not all(qm[("shanghaitech", c)] < s1["shanghaitech"]["rows"][c]["fused"]["causal"] for c in ("D1", "D2", "D4")):
            raise SystemExit("quarter data: not all scorers lost on ShT; revise the text")
        knn_ahead = sorted(d for d in s1 if qm[(d, "D4")] > qm[(d, "D1")])
        gmm_behind = sorted(d for d in s1 if qm[(d, "D2")] < qm[(d, "D1")])
        if knn_ahead != ["ped1", "ped2", "shanghaitech"] or gmm_behind != ["avenue", "ped1", "shanghaitech"]:
            raise SystemExit(f"quarter data: 1-NN ahead on {knn_ahead}, ML mixture behind on {gmm_behind}; revise the text")
    # cost rows (under the first group)
    v4cost = []             # (label, cells) of the cost rows, for gen/tab_scoring_v4.tex
    if cost:
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{{NC}}}{{l}}{{\\textit{{Scoring cost of the three attributes, 4 CPU threads}}}} \\\\")
        pad = " & " * (4 * (len(METRICS) - 1))
        for code, lab, key_b in (("D1", "NIW: model (MB)", "D1_niw_sufficient_stats"),
                                 ("D4", "1-NN: memory bank (MB)", "D4_knn_bank_float32")):
            cells = []
            for ds, _, dm in DS:
                if ds not in cost:
                    cells.append("--")
                    continue
                mb = cost[ds]["bytes"][key_b] / 1e6
                cells.append(f"{mb:.2f}" if mb < 10 else f"{mb:.1f}")
                mac(f"{'Niw' if code == 'D1' else 'Knn'}MB{dm}", cells[-1])
            lines.append(f"{lab} & " + " & ".join(cells) + pad + " \\\\")
            v4cost.append((lab, cells))
        for code, lab in (("D1", "NIW: ms per frame (median)"), ("D4", "1-NN: ms per frame (median)")):
            cells = []
            for ds, _, dm in DS:
                if ds not in cost:
                    cells.append("--")
                    continue
                ms = cost[ds]["latency"][code]["ms_per_frame_median"]
                cells.append(f"{ms:.2f}")
                mac(f"{'Niw' if code == 'D1' else 'Knn'}Ms{dm}", cells[-1])
            lines.append(f"{lab} & " + " & ".join(cells) + pad + " \\\\")
            v4cost.append((lab, cells))
    lines += ["\\bottomrule", "\\end{tabular}", "\\par\\vspace{3pt}",
              "\\parbox{\\textwidth}{\\scriptsize Protocols as in Table~\\ref{tab:online} (object branch, fused). "
              "Every variant uses the same attribute rows, training subsample, two-fold cross-fitted normal reference "
              "and frame floor as the NIW row; ShT models are fitted per scene. Same components, no prior: the "
              "$k$-means components of the NIW mixture with maximum-likelihood Gaussians. ML Gaussian mixture: "
              "scikit-learn, $K$ of the NIW selection, full covariance. 1-NN: Euclidean distance to the nearest "
              "training object (FAISS, exact), on our attributes, calibration and fusion. z-score calibration: each "
              "channel standardised by the mean and standard deviation of its normal reference scores. Maximum fusion: "
              "the largest calibrated channel score. No causal median changes only the online smoothing. "
              "$^{\\uparrow}$/$^{\\downarrow}$ (online micro only, the pre-registered test): the 95\\% paired video "
              "bootstrap interval of the difference to the NIW row (2,000 replicates, seed 0) lies above / below zero. "
              "Quarter: the five draws of Table~\\ref{tab:setup}b. Scoring cost: one scene's models on ShT; the test "
              "frames with objects of the first test videos, 300 frames after 20 warm-up frames, on the shared host of "
              "Table~\\ref{tab:latency}a; the attribute rows are computed beforehand for both scorers. With velocity "
              "magnitude: a fourth attribute, the sum of the eight velocity bins of an object, with its own NIW "
              "density; $^{\\dagger}$suggested by test results of an earlier experiment (Section~\\ref{sec:res_abl}). "
              "All rows were specified before the runs.}",
              "\\end{table*}"]
    (GEN / "tab_scoring_v3.tex").write_text("\n".join(lines) + "\n")
    scoring_v4(lines, v4cells, v4cost, label, prov4)

    # ---------------------------------------------------------------- hyper-parameters
    hl = ["% Generated by fig_src/make_hakem_v3.py -- do not edit by hand.", "\\begin{table}[t]", "\\centering",
          "\\caption{NIW hyper-parameters selected on normal data: $K$ / $\\kappa_0$ / $\\nu_0-C-1$.}",
          "\\label{tab:hyper}", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{lcccc}",
          "\\toprule", f"Attribute & {head} \\\\", "\\midrule"]
    for at, alab in ATTR:
        cells = []
        for ds, _, _ in DS:
            if ds not in s1:
                cells.append("--")
                continue
            hs = [g[at] for g in s1[ds]["hyper"].values() if at in g]
            if not hs:
                cells.append("--")
                continue
            def rng(key, f=lambda h, k: h[k]):
                vals = sorted({int(round(f(h, key))) for h in hs})
                return str(vals[0]) if len(vals) == 1 else f"{vals[0]}--{vals[-1]}"
            nu_extra = rng("nu0", lambda h, k: h["nu0"] - h["dim"] - 1)
            cells.append(f"{rng('K')} / {rng('kappa0')} / {nu_extra}")
        hl.append(f"{alab} & " + " & ".join(cells) + " \\\\")
    hl += ["\\bottomrule", "\\end{tabular}", "\\par\\vspace{3pt}",
           "\\parbox{\\columnwidth}{\\scriptsize Chosen by two-fold cross-validated predictive log-likelihood of "
           "normal training videos (Section~\\ref{sec:method}); ShT: range over the scene models. The selection "
           "subsample and the k-means initialisation use seed 0.}", "\\end{table}"]
    (GEN / "tab_hyper_v3.tex").write_text("\n".join(hl) + "\n")
    supp_copy("tab_hyper_v3.tex", "supp_tab_hyper_v4.tex", [("\\label{tab:hyper}", "\\label{stab:hyper}")], prov4)

    # E1, pose coverage rule (e1_poz_kapsama_kapisi_on_kayit_2026-11-06.md): coverage, and the default configuration
    # (rule applied) against all three attributes, object branch and object + grid, paired video bootstrap
    for ds, _, dm in DS:
        p = K / "e1" / f"{ds}.json"
        e = json.loads(p.read_text())
        prov.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT)}")
        if any(v > 1e-6 for v in e["checks"].values()):
            raise SystemExit(f"{p}: E1 check (a) failed")
        mac(f"Coverage{dm}", f"{100 * e['coverage']['c']:.1f}")
        mac(f"PersonBoxes{dm}", f"{e['coverage']['person_boxes']:,}")
        for br in ("obj", "grid"):
            b = e["boot"][f"{br}_gated_vs_{br}"]["difference"]
            tag = "Obj" if br == "obj" else "Grid"
            mac(f"Rule{tag}Diff{dm}", sg(b["point"]))
            mac(f"Rule{tag}CI{dm}", f"[{sg(b['ci95'][0])}, {sg(b['ci95'][1])}]")
        if not e["coverage"]["gate_on"]:
            if e["guard_c"]["holds"] is not True:
                raise SystemExit(f"{ds}: E1 guard (c) did not hold; revise the text")
    s1av = s1.get("avenue")
    if s1av:    # E1 check (b): on Avenue, the row without pose is below the NIW row (point estimate)
        if not s1av["rows"]["L-opose"]["fused"]["causal"] < s1av["rows"]["D1"]["fused"]["causal"]:
            raise SystemExit("E1 (b) did not hold on Avenue; revise the text")
    if cost:    # memory and time of the 1-NN bank relative to the NIW models, over the datasets with a cost file
        rm = [cost[d]["bytes"]["D4_knn_bank_float32"] / cost[d]["bytes"]["D1_niw_sufficient_stats"] for d in cost]
        rt = [cost[d]["latency"]["D4"]["ms_per_frame_median"] / cost[d]["latency"]["D1"]["ms_per_frame_median"]
              for d in cost]
        mac("KnnMemRatioMin", f"{min(rm):.0f}")
        mac("KnnMemRatioMax", f"{max(rm):.0f}")
        mac("KnnTimeRatioMin", f"{min(rt):.0f}")
        mac("KnnTimeRatioMax", f"{max(rt):.0f}")
    latency(mac, prov)
    supp_copy("tab_latency_v3.tex", "supp_tab_latency_v4.tex", [("\\label{tab:latency}", "\\label{stab:latency}")],
              prov4)
    supp_copy("tab_scoring_v3.tex", "supp_tab_scoring_v4.tex",
              [("\\label{tab:scoring}", "\\label{stab:scoring}"),
               ("Table~\\ref{tab:latency}a", "Table~\\ref{stab:latency}a"),
               ("Section~\\ref{sec:res_abl}", "Section~\\ref{ssec:changes}")], prov4)
    (GEN / "hakem_v4_provenance.txt").write_text(
        "# v4 outputs of fig_src/make_hakem_v3.py (main_v4.tex, supp_v4.tex), from the inputs listed with their sha256 "
        "in gen/hakem_v3_provenance.txt\n" + "\n".join(prov4) + "\n")
    (GEN / "numbers_hakem_v3.tex").write_text("% Generated by fig_src/make_hakem_v3.py -- do not edit by hand.\n"
                                              + "\n".join(macros) + "\n")
    (GEN / "hakem_v3_provenance.txt").write_text("\n".join(prov) + "\n")
    print(f"{len(macros)} macros; datasets s1={sorted(s1)} cost={sorted(cost)}")


if __name__ == "__main__":
    main()
