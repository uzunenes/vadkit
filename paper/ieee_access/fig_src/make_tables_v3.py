#!/usr/bin/env python3
"""Write the v3 tables (gen/tab_main_v3.tex, tab_online_v3.tex, tab_size_v3.tex, tab_speed_v3.tex, tab_setup_v3.tex),
the number macros gen/numbers_v3.tex and the provenance list gen/tables_v3_provenance.txt from the evidence files only.

    python3 paper/ieee_access/fig_src/make_tables_v3.py [--rep-agg median|mean]

It also writes the v4 outputs of main_v4.tex and supp_v4.tex from the same variables (the v3 files stay unchanged):
gen/tab_main_v4.tex (shorter note), gen/tab_size_v4.tex (compact) and gen/supp_tab_size_v4.tex (the full table,
relabelled stab:size), gen/online_cells_v4.json (cells of gen/tab_online_v4.tex, written by make_text_v3.py),
gen/tab_speed_v4.tex and gen/supp_speed_notes_v4.tex, gen/tab_setup_v4.tex and gen/supp_setup_notes_v4.tex, with
their provenance in gen/tables_v4_provenance.txt.

Table 1 (tab:main) has the published values and our standard-protocol results only, one row per configuration of ours, in
its supervision block (advisor decision of 5 Nov 2026: object branch -> training-free, object + grid -> one-class, the
same configuration on every dataset); our online results are in tab:online. The former switches --ours-protocol and
--ours-rows were removed with that decision.

Standard library only. Number sources:
  ours, accuracy  kanit/r2_20261105/res/base/<ds>_{obj,grid}.json ('offline' = standard protocol: per-video min-max
                  + Gaussian, vadkit/eval/protocol.py; 'causal' = online micro, 'causal_macro' = online macro)
  ours, size      kanit/boyut_20261105/size_ours.json (image tower counted in fp32, 4 bytes/parameter, as the pipeline
                  runs it), size_ours.py (where peak memory was measured), onnx_check.txt; grid autoencoder: parameter
                  count derived from aean/models.py and checked against results/runs_ped{1,2}_2d_noadv.log (not in
                  kanit/, see the provenance file)
  competitors     AUC: kanit/literature/dogrulanmis_literatur_2026-11-04.json; block: denetim_turu_2026-11-05.json;
                  published speed: 'speed' entries of fig_src/tables_v3_inputs.json (value, hardware and setting as
                  short verbatim quotes of the competitor's own paper, with location and URL);
                  size: kanit/boyut_20261105/{rakip_boyutlari_dogrulanmis,size_vpc,size_aed}.json through the quotes in
                  fig_src/tables_v3_inputs.json; Avenue label set: kanit/avenue_etiket_20261105/diff.json
  speed           kanit/hiz_20261105/{bench_*_allcores,bench_*_4cores,c1/bench_parflow_r*,c2/trtyolo/par_*}.json,
                  env.json, c2_step2/{export.log,ped1_*.json}, c2/trt/encoder_check.log, load files; TensorRT test
                  sets: kanit/r2_20261105/res/base_trt/*.json against res/base/*.json
  setup           kanit/hiz_20261105/fit_*.log, kanit/kurulum_20261105/dataeff_*.json (sample std over draws; feature
                  extraction estimated as training frames x the PyTorch end-to-end latency of the speed table)
  GFLOPs          'gflops' specs of fig_src/tables_v3_inputs.json
A missing or inconsistent input stops the script (no silent fallback).
"""
import argparse
import json
import re
import statistics
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # paper/ieee_access
REPO = ROOT.parents[1]                  # cctv-aean
K = ROOT / "kanit"
GEN = ROOT / "gen"
INPUTS = HERE / "tables_v3_inputs.json"

# --------------------------------------------------------------------------------------------------- switches
REP_AGG = "median"          # combining the per-repetition medians of the latency benchmarks: "median" or "mean"
# Pre-registered equivalence margin of the TensorRT detectors, online micro AUC points
# (docs/tez_notlari/hiz_on_kayit_2026-11-05.md, C2 step 2).
TRT_MARGIN = 0.5

DS = ["ped2", "avenue", "shanghaitech"]
DSN = {"ped1": "Ped1", "ped2": "Ped2", "avenue": "Avenue", "shanghaitech": "ShT"}
LIT_KEY = {"ped2": "ped2_auc", "avenue": "avenue_auc", "shanghaitech": "shtech_auc"}
OBJ = {"ovel", "opose", "oapp"}
GRID = OBJ | {"I", "Fs", "Fd"}

PROV = []                               # (table, label, value, source)
# v4 outputs (main_v4.tex, supp_v4.tex): their provenance goes to gen/tables_v4_provenance.txt, so that every v3 output,
# gen/tables_v3_provenance.txt included, stays byte-identical
PROV4 = []                              # (output, label, value, source)


def rel(p: Path) -> str:
    return str(p.relative_to(REPO))


def prov(table, label, value, source):
    PROV.append((table, label, value, source))


def prov4(out, label, value, source):
    PROV4.append((out, label, value, source))


def sub_once(text, old, new, name):
    """Replace one occurrence of old in text (the supplement copies of v3 tables); stop if it is not exactly one."""
    if text.count(old) != 1:
        raise SystemExit(f"{name}: {old!r} found {text.count(old)} times, expected once")
    return text.replace(old, new)


def jload(p: Path):
    if not p.exists():
        raise SystemExit(f"missing input: {p}")
    return json.loads(p.read_text())


def tload(p: Path) -> str:
    if not p.exists():
        raise SystemExit(f"missing input: {p}")
    return p.read_text(errors="replace").replace("\r", "\n")


def first_num(s: str) -> float:
    m = re.search(r"\d[\d,]*(?:\.\d+)?", s)
    if not m:
        raise SystemExit(f"no number in quote {s!r}")
    return float(m.group().replace(",", ""))


def f1(x, d=1):
    return f"{x:.{d}f}"


def big(x):
    """Parameter count in millions: thousands separator, 0 decimals at >= 1000, else 1 decimal."""
    return f"{x:,.0f}" if x >= 1000 else f"{x:.1f}"


def mb(x):
    """Size in MB: 1 decimal below 1000 MB, then GB with 2 decimals below 10 GB, else 1 decimal."""
    if x < 1000:
        return f"{x:.1f}~MB"
    g = x / 1000
    return f"{g:.2f}~GB" if g < 10 else f"{g:.1f}~GB"


def half_up(x, d=1):
    """Round half up on the shortest decimal form of x (minutes are frames / frame rate: 4.25 -> 4.3, 1.15 -> 1.2;
    Python's format would give 4.2 and 1.1)."""
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def mins(x):
    return half_up(x, 1) if x < 20 else half_up(x, 0)


def signed(x, d=1):
    """Signed number with a typographic minus."""
    return ("$-$" if x < 0 else "$+$") + f"{abs(x):.{d}f}"


def pct(x):
    return f"{100 * x:.1f}"


def tex_escape(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def write(name, text):
    GEN.mkdir(exist_ok=True)
    (GEN / name).write_text(text)
    print("wrote", rel(GEN / name))


# --------------------------------------------------------------------------------------------------- evidence access
class Evidence:
    def __init__(self):
        self.inp = jload(INPUTS)
        self.size = jload(ROOT / self.inp["size_json"])
        self.size_path = ROOT / self.inp["size_json"]
        lit = K / "literature/dogrulanmis_literatur_2026-11-04.json"
        aud = K / "literature/denetim_turu_2026-11-05.json"
        self.lit, self.lit_path = jload(lit), lit
        self.aud, self.aud_path = jload(aud), aud
        self.label_path = K / "avenue_etiket_20261105/diff.json"
        self.labels = jload(self.label_path)

    def size_entry(self, prefix):
        m = [e for e in self.size if e["name"].startswith(prefix)]
        if len(m) != 1:
            raise SystemExit(f"size entry {prefix!r}: {len(m)} matches")
        return m[0]

    def size_text(self, t):
        e = self.size_entry(t["entry"])
        if t.get("component"):
            c = [c for c in e["components"] if c["component"].startswith(t["component"])]
            if len(c) != 1:
                raise SystemExit(f"component {t['component']!r} of {t['entry']!r}: {len(c)} matches")
            if t.get("field") == "params_millions" and c[0]["used_at_test_time"] is not True:
                raise SystemExit(f"{t['component']!r} is not used at test time")
            return c[0][t["field"]]
        return e[t["field"]]

    def check(self, t):
        txt = self.size_text(t)
        if t["quote"] not in txt:
            raise SystemExit(f"quote not found in {t['entry']} / {t.get('component')} / {t['field']}: {t['quote']!r}")

    def term(self, t, table, label):
        """Value of one term: a quote in the verified size JSON, or a key path in a measured JSON file."""
        if "file" in t:
            p = ROOT / t["file"]
            v = jload(p)
            for k in t["path"]:
                v = v[k]
            prov(table, label, repr(v), f"{rel(p)} : {'/'.join(t['path'])}")
            return float(v)
        self.check(t)
        v = first_num(t["quote"]) * t.get("scale", 1)
        prov(table, label, f"{v:g}", f"{rel(self.size_path)} : {t['entry']} / {t.get('component', '-')} / "
                                     f"{t['field']} : quote {t['quote']!r}")
        return v

    def lit_paper(self, lst, prefix):
        m = [p for p in self.lit[lst]["papers"] if p["title"].startswith(prefix)]
        if len(m) != 1:
            raise SystemExit(f"literature {lst}/{prefix!r}: {len(m)} matches")
        return m[0]

    def audit(self, prefix):
        m = [p for p in self.aud if p["title"].startswith(prefix)]
        if len(m) != 1:
            raise SystemExit(f"audit {prefix!r}: {len(m)} matches")
        return m[0]

    def label_set(self, key):
        """'A' (official CUHK mask, = ours) or 'B' (interval file) from the identical groups of diff.json."""
        if key is None:
            return None
        groups = self.labels["identical_groups"]
        ga = [g for g in groups if "cuhk_mask_anypixel" in g]
        if len(ga) != 1 or "ours_gt_json" not in ga[0]:
            raise SystemExit("diff.json: official mask and our gt.json are not in one group")
        gk = [g for g in groups if key in g]
        if len(gk) != 1:
            raise SystemExit(f"diff.json: {key} not found")
        return "A" if gk[0] is ga[0] else "B"


def combine(ev, spec, table, label):
    """(display string, numeric value or (lo, hi)) for a params / weights spec."""
    b = spec["bound"]
    if b == "none":
        ev.check(spec["check"])
        return spec["text"], None
    if b == "range":
        vals = [ev.term(t, table, f"{label} term {i}") for i, t in enumerate(spec["terms"])]
        return None, (min(vals), max(vals))
    if b == "range_of_sums":
        sums = [sum(ev.term(t, table, f"{label} group {g} term {i}") for i, t in enumerate(grp))
                for g, grp in enumerate(spec["groups"])]
        return None, (min(sums), max(sums))
    tot = sum(ev.term(t, table, f"{label} term {i}") for i, t in enumerate(spec["terms"]))
    return b, tot


def fmt_combined(kind, val, unit):
    """unit 'M' (parameters) or 'MB'."""
    f = big if unit == "M" else mb
    if isinstance(val, tuple):
        lo, hi = val
        if unit == "M":
            return f"{f1(lo)}--{f1(hi)}"
        return f"{f1(lo)}--{f1(hi)}~MB"
    if val is None:
        return kind
    pre = {"ge": r"$\geq$", "approx": r"$\approx$", "eq": ""}[kind]
    return pre + f(val)


# --------------------------------------------------------------------------------------------------- our numbers
def fused(table: dict, chans: set, metric: str) -> float:
    m = [v for k, v in table.items() if "+" in k and set(k.split("+")) == chans]
    if len(m) != 1:
        raise SystemExit(f"fused set {sorted(chans)} not found")
    return m[0][metric]


def ours_auc():
    """{(ds, branch, protocol): AUC} from round 2 'base'; branch obj / grid, protocol standard / online (micro) /
    online_macro (mean of the per-video AUCs)."""
    out = {}
    for ds in ["ped1"] + DS:
        for br, chans in (("obj", OBJ), ("grid", GRID)):
            p = K / f"r2_20261105/res/base/{ds}_{br}.json"
            t = jload(p)
            for proto, key in (("standard", "offline"), ("online", "causal"), ("online_macro", "causal_macro")):
                out[(ds, br, proto)] = fused(t, chans, key)
                prov("main", f"ours {ds} {br} {proto}", pct(out[(ds, br, proto)]), f"{rel(p)} : fused {key}")
    return out


def protocol_constants():
    """Gaussian sigma of the standard protocol; causal median length and warm-up of the online protocol."""
    p = REPO / "vadkit/eval/protocol.py"
    src = tload(p)
    m = re.search(r"def offline\(.*?gaussian_filter1d\(s, (\d+)\)", src, re.S)
    if not m or "per-video min-max" not in src:
        raise SystemExit("standard protocol definition changed in vadkit/eval/protocol.py")
    sigma = int(m.group(1))
    prov("main", "standard protocol sigma", str(sigma), f"{rel(p)} : offline()")
    if "causal_macro: mean of the per-video causal AUCs of videos with both classes" not in src:
        raise SystemExit("macro AUC definition changed in vadkit/eval/protocol.py: revise the online table caption")
    prov("online", "macro AUC definition", "per-video causal AUCs of videos with both classes, averaged",
         f"{rel(p)} : evaluate() docstring and code")
    c = REPO / "vadkit/configs/r2/base.yaml"
    y = tload(c)
    k = re.search(r"temporal:\s*\n\s*name: causal_median\s*\n\s*args:\s*\n\s*k: (\d+)", y)
    w = re.search(r"^warmup: (\d+)", y, re.M)
    if not k or not w:
        raise SystemExit("causal median / warmup not found in base.yaml")
    prov("main", "online median length, warm-up", f"{k.group(1)}, {w.group(1)}", rel(c))
    return sigma, int(k.group(1)), int(w.group(1))


def grid_ae():
    """Parameter count of the grid branch autoencoder (AEAN '2d' variant, 1 channel, no discriminator at test time),
    derived from the layer definition in aean/models.py and checked against the training logs; training time of
    the last logged step from the same logs. These files are outside kanit/."""
    p = REPO / "aean/models.py"
    src = tload(p)
    ks = tuple(int(x) for x in re.search(r"^KERNELS = \(([\d, ]+)\)", src, re.M).group(1).split(","))
    ws = tuple(int(x) for x in re.search(r"^WIDTHS = \(([\d, ]+)\)", src, re.M).group(1).split(","))
    n, c = 0, 1                                             # encoder: conv(c -> w, k) + bias + BN(w)
    for k, w in zip(ks, ws):
        n += c * w * k * k + w + 2 * w
        c = w
    for k, w in zip(ks[::-1][:-1], ws[::-1][1:]):           # decoder: deconv + bias + BN
        n += c * w * k * k + w + 2 * w
        c = w
    n += c * 1 * ks[0] * ks[0] + 1                          # last deconv -> 1 channel (tanh)
    secs = []
    for log in ("results/runs_ped1_2d_noadv.log", "results/runs_ped2_2d_noadv.log"):
        txt = tload(REPO / log)
        logged = float(re.search(r'"ae_params_M": ([\d.]+)', txt).group(1))
        if abs(round(n / 1e6, 3) - logged) > 1e-9:
            raise SystemExit(f"derived AE size {n} does not match {log} ({logged} M)")
        if "adv=False" not in txt.splitlines()[0]:
            raise SystemExit(f"{log} is not the no-adversarial run")
        steps = [(int(a), float(b)) for a, b in re.findall(r'"step": (\d+),[^}]*?"sec": ([\d.]+)', txt)]
        last = max(steps)
        secs.append(last)
        prov("size", f"grid AE training ({log})", f"{last[1]} s for {last[0]} steps", f"{log} : last 'step' / 'sec'")
        gpu = re.search(r'"device": "([^"]+)"', txt).group(1)
        prov("size", f"grid AE training device ({log})", gpu, log)
    prov("size", "grid AE parameters", str(n), f"derived from {rel(p)} (KERNELS, WIDTHS; 2d variant, 1 channel); "
                                              f"logged 'ae_params_M' in results/runs_ped{{1,2}}_2d_noadv.log agrees")
    if len({s for s, _ in secs}) != 1:
        raise SystemExit("AE logs differ in the number of training steps")
    return n, {"steps": secs[0][0], "secs": [t for _, t in secs], "gpu": gpu.replace("NVIDIA GeForce ", "")}


# --------------------------------------------------------------------------------------------------- table 1
def table_main(ev, auc, params_ours, sigma, variants=("v3", "v4")):
    """Table 1: published / standard-protocol columns only; our two fixed configurations, each in its block.
    Variant v3 writes gen/tab_main_v3.tex; v4 the same tabular with a shorter note (gen/tab_main_v4.tex, main_v4.tex).
    Both are written in one call, so that the provenance of the v3 file is recorded once and stays unchanged."""
    comps = ev.inp["competitors"]
    rows = {b: [] for b in ev.inp["block_order"]}
    ncol = 6
    lbl_n = ev.labels["cuhk_vs_vpc_summary"]["n_differing"]
    lbl_tot = ev.labels["total_test_frames"]
    if ev.labels["pairwise_differing_frames"]["cuhk_mask_anypixel__vs__vpc_avenue_mat"] != lbl_n:
        raise SystemExit("diff.json: inconsistent number of differing frames")
    prov("main", "Avenue label sets differ on", f"{lbl_n} of {lbl_tot} frames", f"{rel(ev.label_path)}")
    num = {}
    for c in comps:
        p = ev.lit_paper(*c["lit"])
        a = ev.audit(c["audit"])
        if a["supervision"] != c["block"]:
            raise SystemExit(f"{c['name']}: audit says {a['supervision']}, inputs say {c['block']}")
        if c["venue_match"] not in p["venue"]:
            raise SystemExit(f"{c['name']}: venue {p['venue']!r}")
        prov("main", f"{c['name']} block", a["supervision"], f"{rel(ev.aud_path)} : {c['audit']!r}")
        lab = ev.label_set(c["avenue_label_key"])
        cells = []
        for ds in DS:
            v = p[LIT_KEY[ds]]
            if v is None:
                cells.append("--")
                continue
            s = f1(float(v))
            prov("main", f"{c['name']} {DSN[ds]}", s, f"{rel(ev.lit_path)} : {c['lit'][0]} / {c['lit'][1]!r} / "
                                                     f"{LIT_KEY[ds]}")
            if ds == "avenue" and lab == "B":
                s += r"$^{b}$"
            cells.append(s)
        kind, val = combine(ev, c["params"], "main", f"{c['name']} params")
        num[c["key"]] = val
        name = f"{c['name']}~\\cite{{{c['cite']}}}"
        venue = f"{c['venue']}'{p['year'] % 100:02d}"
        rows[c["block"]].append(" & ".join([name, venue, fmt_combined(kind, val, "M")] + cells) + r" \\")

    # our two configurations, fixed on every dataset; placement by the block definitions (object branch ->
    # training-free, object + grid -> one-class), written down in kurulum_olcumleri_on_kayit (e4031e4, 11:41) after
    # the round-2 test results (257e78b, 10:39), so it is not called pre-registered
    p_obj, p_grid = params_ours
    for br, blk, lab, prm in (("obj", "training_free", "object branch", p_obj),
                              ("grid", "one_class_trained", "object + grid", p_grid)):
        cells = [pct(auc[(ds, br, "standard")]) for ds in DS]
        for ds, s in zip(DS, cells):
            prov("main", f"Table 1 ours {lab} {DSN[ds]} (standard)", s, f"see 'ours {ds} {br} standard'")
        prov("main", f"Table 1 ours {lab} params (M)", f1(prm), "size table: 'ours params object / with grid'")
        rows[blk].append(" & ".join([rf"\textbf{{Ours}}, {lab}", "--", f1(prm)] + cells) + r" \\")

    body = []
    for b in ev.inp["block_order"]:
        body += [r"\midrule" if body else "", rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{{ev.inp['block_title'][b]}}}}} \\"]
        body += rows[b]
    body = [x for x in body if x]
    head = [r"Method & Venue & Params (M) & Ped2 & Avenue & ShT \\"]

    acc = [c for c in comps if c["key"] == "accivad"][0]
    _, acc_sht = combine(ev, acc["params_sht_config"], "main", "AccI-VAD params, ShT configuration")
    ev.check(acc["params_sht_config"]["check"])
    vlab = {ev.label_set(c["avenue_label_key"]): [] for c in comps if c["avenue_label_key"]}
    for c in comps:
        if c["avenue_label_key"]:
            vlab[ev.label_set(c["avenue_label_key"])].append(c["name"])
    notes = (rf"Frame-level micro AUC (\%). Competitor values are those published by the authors, computed with their "
             rf"own post-processing; competitors are not reported under the online protocol. Rows are grouped by "
             rf"supervision; compare only within a block. Our rows use the standard protocol: scores normalised to "
             rf"$[0,1]$ per test video and smoothed with a centred Gaussian ($\sigma={sigma}$ frames), for "
             rf"comparability with the published values; our results under the online protocol are in "
             rf"Table~\ref{{tab:online}}. Ours: light configuration (YOLO11n, YOLO11n-pose, DIS flow, MobileCLIP2-S0; "
             rf"PyTorch detectors), the same on every dataset apart from the frame gap and, on ShT, per-scene fitting "
             rf"from every fifth training frame (Section~\ref{{sec:impl}}). The object branch trains no network and is listed in the training-free "
             rf"block; with the grid branch, which trains a {f1(p_grid - p_obj, 2)}~M-parameter autoencoder on normal "
             rf"video, the detector is listed in the one-class block. $^{{b}}$Computed with the "
             rf"interval label file (set B), which differs from the official CUHK pixel-mask labels (set A, used for "
             rf"our rows) on {lbl_n:,} of {lbl_tot:,} Avenue test frames. Label set checked against the released files: "
             rf"A for {' and '.join(vlab.get('A', []))}, B for {' and '.join(vlab.get('B', []))}; not checked for "
             rf"the other methods. Params: networks that run at test time; $\geq$: a component's count was not found "
             rf"(AlphaPose; for MULDE also the density networks). AccI-VAD's ShT configuration runs no CLIP model "
             rf"($\geq${f1(acc_sht)}~M).")
    # v4 note (main_v4.tex): our protocol and configuration in short (Section IV-B and the Accuracy section give them
    # in full); the label-set and parameter sentences are those of v3, from the same variables
    notes_v4 = (rf"Frame-level micro AUC (\%). Competitors: values published by the authors, computed with their own "
                rf"post-processing. Rows are grouped by supervision; compare only within a block. Ours: standard "
                rf"protocol (Section~\ref{{sec:protocols}}; online results in Table~\ref{{tab:online}}), one "
                rf"configuration per row, the same on every dataset apart from the frame gap and one ShT setting "
                rf"(Section~\ref{{sec:impl}}). $^{{b}}$Computed with the "
                rf"interval label file (set B), which differs from the official CUHK pixel-mask labels (set A, used for "
                rf"our rows) on {lbl_n:,} of {lbl_tot:,} Avenue test frames. Label set checked against the released files: "
                rf"A for {' and '.join(vlab.get('A', []))}, B for {' and '.join(vlab.get('B', []))}; not checked for "
                rf"the other methods. Params: networks that run at test time; $\geq$: a component's count was not found "
                rf"(AlphaPose; for MULDE also the density networks). AccI-VAD's ShT configuration runs no CLIP model "
                rf"($\geq${f1(acc_sht)}~M).")
    for variant in variants:
        tex = "\n".join([
            r"% Generated by fig_src/make_tables_v3.py -- do not edit by hand.",
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Accuracy and model size, grouped by supervision type: published values of the competitors and our "
            r"results under the standard (offline) protocol.}",
            r"\label{tab:main}",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{5pt}",
            r"\begin{tabular}{lllccc}",
            r"\toprule", *head, r"\midrule", *body, r"\bottomrule",
            r"\end{tabular}",
            r"\par\vspace{3pt}",
            rf"\parbox{{\textwidth}}{{\scriptsize {notes if variant == 'v3' else notes_v4}}}",
            r"\end{table*}", ""])
        write(f"tab_main_{variant}.tex", tex)
    prov4("tab_main_v4.tex", "tabular (all cells)", "identical to tab_main_v3.tex",
          "same rows as tab_main_v3.tex (see the 'main' entries of tables_v3_provenance.txt)")
    prov4("tab_main_v4.tex", "note: Avenue label sets differ on", f"{lbl_n:,} of {lbl_tot:,}", rel(ev.label_path))
    prov4("tab_main_v4.tex", "note: AccI-VAD ShT configuration params (M)", f1(acc_sht),
          "tables_v3_provenance.txt : 'AccI-VAD params, ShT configuration'")
    return num


# --------------------------------------------------------------------------------------------------- online table
def table_online(auc, sigma, kmed, warm):
    """Our two configurations under the standard and the online protocol (micro AUC), plus the online macro AUC."""
    ds4 = ["ped1"] + DS
    lines = []
    cells_v4 = []           # the 24 cells for gen/tab_online_v4.tex (written by make_text_v3.py, with the grid effect)
    for br, lab in (("obj", "Object branch"), ("grid", "Object + grid")):
        if lines:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{lab}}}}} \\")
        for proto, name in (("standard", "Standard, micro"), ("online", "Online, micro"),
                            ("online_macro", "Online, macro")):
            cells = [pct(auc[(ds, br, proto)]) for ds in ds4]
            for ds, s in zip(ds4, cells):
                prov("online", f"ours {lab} {name} {DSN[ds]}", s, f"see 'ours {ds} {br} {proto}'")
                key = {"standard": "offline", "online": "causal", "online_macro": "causal_macro"}[proto]
                cells_v4.append({"config": br, "protocol": proto, "dataset": ds, "value": s,
                                 "source": f"kanit/r2_20261105/res/base/{ds}_{br}.json : fused {key}"})
            lines.append(" & ".join([r"\quad " + name] + cells) + r" \\")
    if len(cells_v4) != 24:
        raise SystemExit("online table: expected 24 cells")
    write("online_cells_v4.json", json.dumps(cells_v4, indent=1) + "\n")
    prov4("online_cells_v4.json", "24 cells (2 configurations x 3 metrics x 4 datasets)", "as tab_online_v3.tex",
          "kanit/r2_20261105/res/base/<ds>_{obj,grid}.json : fused offline / causal / causal_macro")
    notes = (rf"Micro: AUC over all test frames pooled; macro: mean of the per-video AUCs over the test videos that "
             rf"contain normal and anomalous frames (online protocol only). Standard protocol as in "
             rf"Table~\ref{{tab:main}} (per-video normalisation to $[0,1]$, centred Gaussian, $\sigma={sigma}$ frames). "
             rf"Light configuration of Table~\ref{{tab:main}}, PyTorch detectors. Only our detector is listed: "
             rf"competitors are compared only through their published values and are not reported under the online "
             rf"protocol. $^{{a}}$Development set.")
    tex = "\n".join([
        r"% Generated by fig_src/make_tables_v3.py -- do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Our two configurations under the standard and the online protocol, frame AUC (\%). Online: "
        rf"strictly causal scoring (no future frames, no per-video normalisation, median of the last {kmed} frame "
        rf"scores; the first {warm} frames of each video are not scored).}}",
        r"\label{tab:online}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Protocol & Ped1$^{a}$ & Ped2 & Avenue & ShT \\",
        r"\midrule", *lines, r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{3pt}",
        rf"\parbox{{\columnwidth}}{{\scriptsize {notes}}}",
        r"\end{table}", ""])
    write("tab_online_v3.tex", tex)


# --------------------------------------------------------------------------------------------------- table 2
# short training cells of the compact size table (main_v4.tex); the full cells stay in the supplement copy
TRAIN_V4 = {"accivad": "None (feature banks)", "vpc": "None (feature banks)",
            "anomalyruler": "None (rules induced with GPT-4)", "anyanomaly": "None (zero-shot)",
            "mulde": "MLPs per dataset", "stgnf": "Normalizing flow per dataset",
            "aedmae": "GPU training with synthetic anomalies, per dataset",
            "fastaed": "Two teachers distilled into a student, per dataset"}


def gflops_cell(ev, spec, label):
    """GFLOPs cell of the size table from a 'gflops' spec of tables_v3_inputs.json."""
    k = spec["kind"]
    if k == "none":
        return "--"
    if k == "frame_crop":
        fr = sum(ev.term(t, "size", f"{label} GFLOPs per frame, term {i}") for i, t in enumerate(spec["frame"]))
        cr = sum(ev.term(t, "size", f"{label} GFLOPs per crop, term {i}") for i, t in enumerate(spec["crop"]))
        return f"{f1(fr)} + {f1(cr)} per crop"
    vals = [ev.term(t, "size", f"{label} GFLOPs term {i}") for i, t in enumerate(spec["terms"])]
    if k == "range":
        return f"{f1(min(vals))}--{f1(max(vals))}"
    if k == "eq":
        return f1(sum(vals))
    raise SystemExit(f"{label}: unknown gflops kind {k!r}")


def published_speed(c):
    """(cell, note, fps values) of a competitor's 'speed' entry: every number comes from a verbatim quote of the
    competitor's paper (tables_v3_inputs.json); the templates themselves contain no digit."""
    sp = c["speed"]
    for k in ("cell", "note"):
        if re.search(r"\d", sp[k]):
            raise SystemExit(f"{c['name']}: digit in the speed {k} template")
    fill, fps = {}, {}
    for name, v in sp.get("values", {}).items():
        if not re.search(r"(?<![\d.])" + re.escape(v["value"]) + r"(?![\d.])", v["quote"]):
            raise SystemExit(f"{c['name']}: speed value {v['value']!r} not in its quote")
        fill[name] = tex_escape(v["value"])
        prov("size", f"{c['name']} published speed {name}", v["value"],
             f"{sp['source']} : quote {v['quote']!r} : {sp['url']}")
        if name in ("fps", "cpu"):
            fps[name] = float(v["value"])
    for name, h in sp.get("hardware", {}).items():
        if h["name"] not in h["quote"]:
            raise SystemExit(f"{c['name']}: hardware {h['name']!r} not in its quote")
        fill[name] = h["name"]
        prov("size", f"{c['name']} published speed hardware {name}", h["name"], f"quote {h['quote']!r}")
    if not sp["reported"]:
        prov("size", f"{c['name']} published speed", "not reported for the configuration in the tables",
             f"{sp['source']}{' : ' + sp['searched'] if 'searched' in sp else ''} : {sp['url']}")
        fps = {}
    return sp["cell"].format(**fill), sp["note"].format(**fill), fps


def table_size(ev, ours, g2, ae_secs, cols):
    # setup of one scene: the single-scene datasets of the G2 session (ShT fits 13 scene models in one run)
    fit1 = [g2["ds"][d]["fit"] for d in ("ped1", "ped2", "avenue")]
    ext1 = [g2["ds"][d]["extract"] / 60 for d in ("ped1", "ped2", "avenue")]
    comps = ev.inp["competitors"]
    cell = lambda s: r"\raggedright " + s
    ncol = 8
    so = ours["so"]
    B = K / "boyut_20261105"
    # where the peak GPU memory was measured (size_ours.py) and the one-frame ONNX check (onnx_check.txt)
    src = tload(B / "size_ours.py")
    m = re.search(r'frame_paths\("/workspace/frames/avenue", "testing"\)\["(\d+)"\]\[:(\d+)\]', src)
    if not m or "Perception(cfg" not in src:
        raise SystemExit("size_ours.py: peak-memory measurement not found")
    mem_video, mem_frames = m.group(1), int(m.group(2))
    prov("size", "peak GPU memory: Avenue test video, frames", f"{mem_video}, {mem_frames}",
         f"{rel(B / 'size_ours.py')} (object-branch Perception only)")
    oc = re.search(r"yolo11n test frame (\d+) of Avenue (\d+): torch (\d+) boxes, onnx (\d+) boxes \(conf ([\d.]+)\)",
                   tload(B / "onnx_check.txt"))
    if not oc:
        raise SystemExit("onnx_check.txt: box comparison not found")
    prov("size", "ONNX check YOLO11n boxes torch / onnx", f"{oc.group(3)} / {oc.group(4)}", rel(B / "onnx_check.txt"))
    for k in ("yolo11n.pt", "yolo11n-pose.pt", "MobileCLIP2-S0"):
        if so[k]["onnx_export"] != "ok":
            raise SystemExit(f"ONNX export of {k} not ok")
    prov("size", "ONNX export ok (3 networks; grid AE not exported)", "ok",
         "kanit/boyut_20261105/size_ours.json : */onnx_export")
    for k in ("state_ped2", "state_avenue"):
        prov("size", f"ours {k} total_MB", f1(so[k]["total_MB"], 3), "kanit/boyut_20261105/size_ours.json")
    prov("size", "ours peak GPU memory allocated (MiB)", f"{so['runtime']['peak_allocated_MB']:.1f}",
         "kanit/boyut_20261105/size_ours.json : runtime/peak_allocated_MB")
    g_frame = so["yolo11n.pt"]["gflops_640_ultralytics"] + so["yolo11n-pose.pt"]["gflops_640_ultralytics"]
    g_crop = so["MobileCLIP2-S0"]["visual_gflops_per_crop"]
    prov("size", "ours GFLOPs per frame (YOLO11n + YOLO11n-pose, 640) / per crop (MobileCLIP2-S0)",
         f"{g_frame:.3f} / {g_crop:.3f}", "kanit/boyut_20261105/size_ours.json : gflops_640_ultralytics, "
                                          "visual_gflops_per_crop")
    sp_cols = {c["meta"]["config"] + ("_par" if c["meta"]["parallel_flow"] else ""): c for _, c in cols}
    fps_trt, fps_par = sp_cols["r2_base_trt_par"]["fps"], sp_cols["r2_base_par"]["fps"]
    prov("size", "ours speed cell: object branch, TensorRT YOLO / PyTorch (parallel flow), FPS",
         f"{f1(fps_trt)} / {f1(fps_par)}", "speed table (tab_speed_v3.tex), 'Frames per second'")
    ours_speed = (f"Object branch, measured: {f1(fps_trt)} (TensorRT YOLO), {f1(fps_par)} (PyTorch)$^{{i}}$; "
                  f"object + grid: --")
    ours_row = [r"\textbf{Ours}",
                cell(f"{f1(ours['p_grid'])} ({f1(ours['p_obj'])} without grid branch)"),
                cell(f"{mb(ours['w_total'])}$^{{a}}$"),
                cell(f"{f1(g_frame)} + {f1(g_crop)} per crop"),
                cell(ours_speed),
                cell(f"Object branch: fitted densities and calibration {f1(so['state_ped2']['total_MB'], 2)}--"
                     f"{f1(so['state_avenue']['total_MB'])}~MB$^{{c}}$. Grid branch: autoencoder {f1(ours['ae_MB'])}~MB; "
                     f"cell statistics and references (size not measured)"),
                cell(f"Object branch: none (fit {f1(min(fit1))}--{f1(max(fit1))}~s per single-camera dataset on GPU, after "
                     f"{mins(min(ext1))}--{mins(max(ext1))}~min of feature extraction; Table~\\ref{{tab:setup}}); grid AE: {f1(min(ae_secs['secs']) / 60)}--{f1(max(ae_secs['secs']) / 60)}~min on "
                     f"GPU$^{{b}}$"),
                cell(f"ONNX export of the three object-branch networks$^{{h}}$; DIS flow on CPU (OpenCV); no custom "
                     f"CUDA extensions; peak GPU memory {so['runtime']['peak_allocated_MB']:.0f}~MiB$^{{d}}$")]
    rows = {b: [] for b in ev.inp["block_order"]}
    rows_v4 = {b: [] for b in ev.inp["block_order"]}    # compact table of main_v4 (no per-scene column, no notes a-i)
    sp_notes, sp_fps = [], {}
    for c in comps:
        sp_cell, sp_note, sp_fps[c["key"]] = published_speed(c)
        sp_notes.append(sp_note)
        kind, val = combine(ev, c["params"], "size", f"{c['name']} params")
        wk, wv = combine(ev, c["weights_MB"], "size", f"{c['name']} weights")
        gstr = gflops_cell(ev, c["gflops"], c["name"])
        named = {}
        for n, t in c.get("named", {}).items():
            v = ev.term(t, "size", f"{c['name']} {n}")
            named[n] = {"mb": mb, "int": lambda x: f"{x:,.0f}", ".1f": f1, ".2f": lambda x: f1(x, 2)}[t["fmt"]](v)
        for t in c.get("check", []):
            ev.check(t)
        txt = {k: tex_escape(v.format(**named)) for k, v in c["text"].items()}
        pstr = fmt_combined(kind, val, "M")
        if c["key"] == "aedmae":
            pstr += r"$^{e}$"
            gstr += r"$^{e}$"
        if c["key"] == "mulde":
            pstr += r"$^{f}$"
        if c["key"] == "stgnf":
            pstr += r"$^{g}$"
        if c["key"] == "anyanomaly":
            ev.check(c["params_note"])
        rows[c["block"]].append([f"{c['name']}~\\cite{{{c['cite']}}}", cell(pstr), cell(fmt_combined(wk, wv, "MB")),
                                 cell(gstr), cell(sp_cell), cell(txt["artefacts"]), cell(txt["training"]),
                                 cell(txt["portability"])])
        # compact v4 row: the same numeric cells without the note marks e-g, a short training cell and the first
        # clause of the portability cell (plus the clause naming custom CUDA extensions, where there is one)
        if TRAIN_V4.get(c["key"]) is None:
            raise SystemExit(f"{c['name']}: no short training cell for the v4 size table")
        port = txt["portability"].split("; ")
        port_v4 = [port[0]] + [p for p in port[1:] if "custom CUDA" in p and "custom CUDA" not in port[0]]
        rows_v4[c["block"]].append([f"{c['name']}~\\cite{{{c['cite']}}}", cell(fmt_combined(kind, val, "M")),
                                    cell(fmt_combined(wk, wv, "MB")), cell(gstr.replace(r"$^{e}$", "")),
                                    cell(sp_cell), cell(TRAIN_V4[c["key"]]), cell("; ".join(port_v4))])
    # the Size section compares the published speeds with ours at face value; keep that sentence true
    faster = sorted(k for k, v in sp_fps.items() if v and max(v.values()) > fps_trt)
    slower = sorted(k for k, v in sp_fps.items() if v and max(v.values()) < fps_trt)
    none = sorted(k for k, v in sp_fps.items() if not v)
    if (faster, slower, none) != (["aedmae", "fastaed", "stgnf", "vpc"], ["accivad", "anyanomaly"],
                                  ["anomalyruler", "mulde"]):
        raise SystemExit(f"published speeds vs ours changed: {faster} / {slower} / {none}; revise the Size section")
    prov("size", "published speed above / below our TensorRT FPS", f"{faster} / {slower}", "comparison of the above")
    mulde = [c for c in comps if c["key"] == "mulde"][0]
    mlp = ev.term(mulde["named"]["mlp_derived"], "size", "MULDE MLPs (derived)")
    aed = ev.lit_paper("sota", "Self-Distilled Masked Auto-Encoders")
    aed_paper = re.search(r"([\d.]+) GFLOPs", aed["params_or_flops"])
    if not aed_paper:
        raise SystemExit("AED-MAE: reported GFLOPs not found in the literature file")
    prov("size", "AED-MAE GFLOPs reported in the paper", aed_paper.group(1),
         f"{rel(ev.lit_path)} : sota / 'Self-Distilled Masked Auto-Encoders' / params_or_flops")
    mc = so["MobileCLIP2-S0"]
    yolo, pose = so["yolo11n.pt"]["file_MB"], so["yolo11n-pose.pt"]["file_MB"]
    notes = (rf"Params: networks that run at test time (M). $\geq$: lower bound, a component's size was not found "
             rf"(AlphaPose; MULDE's density networks). GFLOPs: per frame, YOLO models at 640$\times$640, plus per object "
             rf"crop for the image encoders (ours at {mc['image_size']}$\times${mc['image_size']}, VideoPatchCore at "
             rf"224$\times$224); counted by us with the Ultralytics and PyTorch FLOP counters for ours, VideoPatchCore "
             rf"and AED-MAE; Fast-AED as reported (input size not stated); --: not found for the configuration used. "
             rf"$^{{a}}$YOLO11n {f1(yolo)}~MB and YOLO11n-pose {f1(pose)}~MB (released files), MobileCLIP2-S0 image "
             rf"tower {f1(ours['enc_MB'])}~MB (fp32 as run, from its {ours['enc_n']:,} parameters; its ONNX export is "
             rf"{f1(mc['onnx_MB'])}~MB), grid autoencoder {f1(ours['ae_MB'])}~MB (fp32, from its {ours['ae_n']:,} "
             rf"parameters). The pipeline loads the full open\_clip model; its text tower "
             rf"({f1(mc['total_params_incl_text_M'] - mc['visual_params_M'])}~M parameters) is not used and not "
             rf"counted, whereas the CLIP files of AccI-VAD, MULDE and VideoPatchCore are counted as released, text "
             rf"tower included. $^{{b}}${ae_secs['steps']:,} training steps on an {ae_secs['gpu']} (Ped1 and Ped2 "
             rf"training logs; not timed on Avenue or ShT). $^{{c}}$Ped2 and Avenue. $^{{d}}$Allocated by PyTorch on an "
             rf"RTX 3090 for the object-branch perception on the first {mem_frames} frames of Avenue test video "
             rf"{mem_video} (process total {so['runtime']['nvidia_smi_process'].split(',')[1].strip()}); grid branch "
             rf"not included. $^{{e}}$ShT and Avenue models, measured with the authors' weights; GFLOPs of one forward "
             rf"pass of the released code at its input sizes (the paper reports {aed_paper.group(1)}). "
             rf"$^{{f}}$Without the three density networks, whose size is not reported ({f1(mlp)}~M derived from the "
             rf"stated layers). $^{{g}}$Pose estimator and tracker as stated in the paper, YOLOX-X detector named in "
             rf"the paper and the flow model; the released code invokes a different detector whose size was not "
             rf"found. $^{{h}}$YOLO11n, YOLO11n-pose and the MobileCLIP2-S0 image tower; the grid autoencoder was not "
             rf"exported, and the ONNX models were not used in our experiments. On one Avenue test frame the ONNX "
             rf"YOLO11n returned {oc.group(4)} boxes and the PyTorch model {oc.group(3)} (confidence {oc.group(5)}). "
             rf"VideoPatchCore and AED-MAE were measured with the authors' weights; other competitor sizes come from "
             rf"the papers, model cards and checkpoint files. $^{{i}}$Published speed: frames per second as reported by "
             rf"the authors, on their own hardware and with their own timing protocols, which mostly do not state the "
             rf"batch size or which stages are timed; the values are therefore not comparable with each other or with "
             rf"ours. " + ". ".join(sp_notes) + ". "
             rf"Ours: measured by us end to end at batch size 1 on an RTX 3090 (Table~\ref{{tab:speed}}), object branch "
             rf"with TensorRT FP16 detectors or with PyTorch, both with parallel flow. The grid branch was timed only "
             rf"alone (Table~\ref{{tab:latency}}b) and the two branches are not combined in one streaming "
             rf"implementation, so no speed is given for object + grid.")
    body = [rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{Ours: training-free object branch; one-class grid branch}}}} "
            r"\tabularnewline", " & ".join(ours_row) + r" \tabularnewline"]
    for b in ev.inp["block_order"]:
        body += [r"\midrule", rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{{ev.inp['block_title'][b]}}}}} \tabularnewline"]
        body += [" & ".join(r) + r" \tabularnewline" for r in rows[b]]
    tex = "\n".join([
        r"% Generated by fig_src/make_tables_v3.py -- do not edit by hand.",
        # [!t]: the table is taller than the class's top fraction for double-column floats; without '!' it and every
        # later table drift to the end of the paper
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Model size and portability, grouped by supervision type.}",
        r"\label{tab:size}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{p{2.45cm}p{1.35cm}p{1.4cm}p{1.45cm}p{1.85cm}p{2.5cm}p{2.5cm}p{2.5cm}}",
        r"\toprule",
        r"Method & \raggedright Params (M) & \raggedright Weights on disk & \raggedright GFLOPs & "
        r"\raggedright Published speed (FPS)$^{i}$ & "
        r"\raggedright Per-scene artefacts & \raggedright Training on the target data & \raggedright Portability "
        r"\tabularnewline",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{3pt}",
        rf"\parbox{{\textwidth}}{{\scriptsize {notes}}}",
        r"\end{table*}", ""])
    write("tab_size_v3.tex", tex)

    # supplement (supp_v4.tex): the full table verbatim, relabelled, with the latency table of the supplement
    supp = sub_once(tex, r"\label{tab:size}", r"\label{stab:size}", "supp_tab_size_v4")
    supp = sub_once(supp, r"Table~\ref{tab:latency}b", r"Table~\ref{stab:latency}b", "supp_tab_size_v4")
    supp = sub_once(supp, r"object + grid: --",
                    r"object + grid: \VbFpsGridTrtAvenue{} (TensorRT YOLO), \VbFpsGridPtAvenue{} (PyTorch)$^{i}$",
                    "supp_tab_size_v4")
    supp = sub_once(supp,
                    r"The grid branch was timed only alone (Table~\ref{stab:latency}b) and the two branches are not "
                    r"combined in one streaming implementation, so no speed is given for object + grid.",
                    r"Object + grid: measured in a separate session with both branches in one streaming class "
                    r"(Table~\ref{stab:combined}), on Avenue (object branch in the same session: \VbFpsObjTrtAvenue{} "
                    r"and \VbFpsObjPtAvenue{} FPS); the grid branch alone in the second session: "
                    r"Table~\ref{stab:latency}b.",
                    "supp_tab_size_v4")
    write("supp_tab_size_v4.tex", supp)
    prov4("supp_tab_size_v4.tex", "all cells and notes", "as tab_size_v3.tex, with combined object + grid speed macros",
          "tab_size_v3.tex with label stab:size, latency reference stab:latency, and \\VbFps* macros from stab:combined")

    # compact table of main_v4.tex: same numeric cells, no per-scene column, short training and portability cells
    ae_min = f"{f1(min(ae_secs['secs']) / 60)}--{f1(max(ae_secs['secs']) / 60)}"
    ours_v4 = [r"\textbf{Ours}",
               cell(f"{f1(ours['p_grid'])} ({f1(ours['p_obj'])} without grid)"),
               cell(mb(ours["w_total"])),
               cell(f"{f1(g_frame)} + {f1(g_crop)} per crop"),
               # both values with parallel flow; TensorRT for the two detectors only (the encoder stays in PyTorch)
               cell(f"Object branch, measured with parallel flow: {f1(fps_trt)} (TensorRT YOLO), {f1(fps_par)} "
                    r"(PyTorch); object + grid: \VbFpsGridTrtAvenue{} (TensorRT YOLO), \VbFpsGridPtAvenue{} (PyTorch)"),
               # timed on Ped1 and Ped2 only (note b of the full table), so the datasets are named in the cell
               cell(f"Object branch: none; grid AE per scene, {ae_min}~min on GPU (Ped1, Ped2)"),
               # the caveat of note h of the full table: the ONNX models were exported but not used
               cell("ONNX export of the three object-branch networks (not used in our experiments); no custom CUDA "
                    "extensions")]
    for lab, v, src in (("ours params with / without grid (M)", f"{f1(ours['p_grid'])} / {f1(ours['p_obj'])}",
                         "tables_v3_provenance.txt : 'ours params object / with grid (M)'"),
                        ("ours weights", mb(ours["w_total"]), "tables_v3_provenance.txt : 'ours weights total (MB)'"),
                        ("ours GFLOPs per frame / per crop", f"{f1(g_frame)} / {f1(g_crop)}",
                         "tables_v3_provenance.txt : 'ours GFLOPs per frame ... / per crop'"),
                        ("ours FPS TensorRT / PyTorch", f"{f1(fps_trt)} / {f1(fps_par)}",
                         "tables_v3_provenance.txt : 'ours speed cell'"),
                        ("grid AE training (min)", ae_min, "tables_v3_provenance.txt : 'grid AE training (...)' / 60")):
        prov4("tab_size_v4.tex", lab, v, src)
    prov4("tab_size_v4.tex", "competitor cells", "params, weights, GFLOPs, published speed as tab_size_v3.tex",
          "same variables (tables_v3_provenance.txt, 'size' entries); training cells: TRAIN_V4 in make_tables_v3.py; "
          "portability: first clause of the v3 cell, plus its custom-CUDA clause")
    ncol4 = 7
    body4 = [rf"\multicolumn{{{ncol4}}}{{l}}{{\textit{{Ours: training-free object branch; one-class grid branch}}}} "
             r"\tabularnewline", " & ".join(ours_v4) + r" \tabularnewline"]
    for b in ev.inp["block_order"]:
        body4 += [r"\midrule", rf"\multicolumn{{{ncol4}}}{{l}}{{\textit{{{ev.inp['block_title'][b]}}}}} \tabularnewline"]
        body4 += [" & ".join(r) + r" \tabularnewline" for r in rows_v4[b]]
    notes4 = (r"Params: networks that run at test time; for CLIP-type encoders, the towers that run (image tower; "
              r"AnyAnomaly also runs the text tower). $\geq$: lower bound, a component's size was not found. Weights: ours "
              r"without the unused text tower; competitors' CLIP files counted as released. GFLOPs: per frame, plus per "
              r"object crop for the image encoders; --: not found. Published FPS: as reported by the authors on their "
              r"own hardware and timing protocols, not comparable with each other or with ours; ours measured end to end "
              r"at batch size 1 on an RTX 3090 (object branch: Table~\ref{tab:speed}; object + grid, separate session: "
              r"Section~\ref{sec:res_speed} and Supplementary Table~\ref{stab:combined}). Full table with per-scene "
              r"artefacts and all notes: Supplementary Table~\ref{stab:size}.")
    tex4 = "\n".join([
        r"% Generated by fig_src/make_tables_v3.py -- do not edit by hand.",
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Model size and portability, grouped by supervision type.}",
        r"\label{tab:size}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{p{2.4cm}p{1.45cm}p{1.3cm}p{1.4cm}p{2.8cm}p{3.4cm}p{3.4cm}}",
        r"\toprule",
        r"Method & \raggedright Params (M) & \raggedright Weights & \raggedright GFLOPs & "
        r"\raggedright Published FPS & \raggedright Training on the target data & \raggedright Portability "
        r"\tabularnewline",
        r"\midrule",
        *body4,
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{3pt}",
        rf"\parbox{{\textwidth}}{{\scriptsize {notes4}}}",
        r"\end{table*}", ""])
    write("tab_size_v4.tex", tex4)


# --------------------------------------------------------------------------------------------------- table 3
STAGES = [("decode", "Decoding"), ("detect", "Detector (YOLO11n)"), ("pose", "Pose (YOLO11n-pose)"),
          ("flow", "Flow (DIS, CPU)"), ("embed", "Embedding (MobileCLIP2-S0)"), ("score", "Scoring, fusion")]
META = ("config", "parallel_flow", "device", "gap", "frame_hw", "warmup_frames", "frames_per_rep", "torch_threads")


def agg(vals, how):
    return statistics.median(vals) if how == "median" else statistics.fmean(vals)


def bench_column(paths, how, label):
    runs = [jload(p) for p in paths]
    for r in runs[1:]:
        for m in META:
            if r.get(m) != runs[0].get(m):
                raise SystemExit(f"{label}: runs differ in {m}")
    out = {}
    for st, _ in STAGES + [("total", "")]:
        vals = [rep[st]["median_ms"] for r in runs for rep in r["reps"]]
        out[st] = agg(vals, how)
    per_run = [agg([rep["total"]["median_ms"] for rep in r["reps"]], how) for r in runs]
    nrep = sum(len(r["reps"]) for r in runs)
    src = ", ".join(rel(p) for p in paths)
    for st in out:
        prov("speed", f"{label} {st} ms", f1(out[st], 2), f"{how} of {nrep} per-repetition medians : {src}")
    return {"ms": out, "fps": 1000 / out["total"], "fps_runs": [1000 / x for x in per_run], "meta": runs[0],
            "nrep": nrep, "nruns": len(runs)}


def load_avgs(paths):
    v = []
    for p in paths:
        for m in re.finditer(r"load average: ([\d.]+),", tload(p)):
            v.append(float(m.group(1)))
    return min(v), max(v)


def table_speed(how, auc):
    H = K / "hiz_20261105"
    cols = [("Sequential", bench_column([H / "bench_seq_allcores.json"], how, "sequential")),
            ("Parallel flow", bench_column([H / "bench_par_allcores.json", H / "c1/bench_parflow_r1.json",
                                            H / "c1/bench_parflow_r2.json"], how, "parallel flow")),
            ("+ TensorRT YOLO", bench_column([H / f"c2/trtyolo/{n}.json" for n in
                                              ("par_r1", "par_r2", "par_cv4_r1", "par_cv4_r2")], how,
                                             "parallel flow + TensorRT YOLO"))]
    if cols[0][1]["meta"]["parallel_flow"] or not cols[1][1]["meta"]["parallel_flow"] \
            or not cols[2][1]["meta"]["parallel_flow"] or cols[2][1]["meta"]["config"] != "r2_base_trt":
        raise SystemExit("speed: unexpected run configuration")
    four = {k: bench_column([H / f"bench_{k}_4cores.json"], how, f"{k} 4 threads") for k in ("seq", "par")}
    env = jload(H / "env.json")
    exp = tload(H / "c2_step2/export.log")
    trt_ver = re.search(r"starting export with TensorRT (\d+(?:\.\d+)+)", exp).group(1)
    if "quantize=16" not in exp:
        raise SystemExit("export.log: FP16 engines not confirmed")
    prov("speed", "TensorRT version / FP16", trt_ver, f"{rel(H / 'c2_step2/export.log')}")
    enc = tload(H / "c2/trt/encoder_check.log")
    m = re.search(r"crops (\d+) cosine min ([\d.]+) mean ([\d.]+)", enc)
    n_crops, cos_mean = int(m.group(1)), float(m.group(3))
    prov("speed", "TensorRT encoder cosine (mean, crops)", f"{cos_mean:.4f}, {n_crops}", rel(H / "c2/trt/encoder_check.log"))
    d_trt = {}
    for br, chans in (("obj", OBJ), ("grid", GRID)):
        p = H / f"c2_step2/ped1_{br}.json"
        d_trt[br] = 100 * (fused(jload(p), chans, "causal") - auc[("ped1", br, "online")])
        prov("speed", f"TensorRT YOLO - PyTorch, Ped1 {br} causal (points)", f"{d_trt[br]:+.2f}",
             f"{rel(p)} minus kanit/r2_20261105/res/base/ped1_{br}.json")
    # pre-registered follow-up (hiz_on_kayit C2 step 2): the test sets re-run with the TensorRT detectors
    d_test = {}
    for ds in DS:
        for br, chans in (("obj", OBJ), ("grid", GRID)):
            p = K / f"r2_20261105/res/base_trt/{ds}_{br}.json"
            q = K / f"r2_20261105/res/base/{ds}_{br}.json"
            d_test[(ds, br)] = 100 * (fused(jload(p), chans, "causal") - fused(jload(q), chans, "causal"))
            prov("speed", f"TensorRT YOLO - PyTorch, {DSN[ds]} {br} causal (points)", f"{d_test[(ds, br)]:+.3f}",
                 f"{rel(p)} minus {rel(q)}")
    over = [k for k, v in d_test.items() if abs(round(v, 2)) > TRT_MARGIN]
    prov("speed", "test-set TensorRT changes beyond the margin", str(over), f"|change| > {TRT_MARGIN} (hiz_on_kayit)")
    trt_txt = ", ".join(f"{DSN[ds]} {signed(d_test[(ds, 'obj')], 2)}\\,/\\,{signed(d_test[(ds, 'grid')], 2)}"
                        for ds in DS)
    over_txt = ("; " + " and ".join(f"{DSN[ds]} {'object + grid' if br == 'grid' else 'object branch'}"
                                    for ds, br in over)
                + f" exceed{'s' if len(over) == 1 else ''} the pre-registered margin of {TRT_MARGIN:g}") if over else ""
    lo, hi = load_avgs([H / "gpu.txt", H / "load_before_bench.txt", H / "load_after_bench.txt",
                        H / "c1/load_pose.txt", H / "c2/trtyolo/load.txt"])
    prov("speed", "host 1-min load average range", f"{lo}-{hi}", "kanit/hiz_20261105/{gpu.txt,load_*.txt,"
                                                                  "c1/load_pose.txt,c2/trtyolo/load.txt}")
    for k in ("gpu", "cpu", "torch"):
        prov("speed", f"env {k}", env[k], rel(H / "env.json"))
    lines = []
    for st, name in STAGES:
        cells = []
        for i, (_, c) in enumerate(cols):
            v = f1(c["ms"][st])
            if st == "flow" and c["meta"]["parallel_flow"]:
                if c["ms"][st] >= 0.05:
                    raise SystemExit("parallel flow wait is not negligible; revise the table note")
                v = r"$<$0.1$^{a}$"
            cells.append(v)
        lines.append(" & ".join([name] + cells) + r" \\")
    lines.append(r"\midrule")
    lines.append(" & ".join([r"End to end$^{b}$"] + [r"\textbf{" + f1(c["ms"]["total"]) + "}" for _, c in cols])
                 + r" \\")
    lines.append(" & ".join(["Frames per second"] + [r"\textbf{" + f1(c["fps"]) + "}" for _, c in cols]) + r" \\")
    lines.append(" & ".join([r"\quad range over runs"] + [f"{f1(min(c['fps_runs']))}--{f1(max(c['fps_runs']))}"
                                                           if c["nruns"] > 1 else "--" for _, c in cols]) + r" \\")
    h, w = cols[0][1]["meta"]["frame_hw"]
    gpu = env["gpu"].replace("NVIDIA GeForce ", "")
    cpu = env["cpu"].replace(" 32-Core Processor", "")
    notes = (rf"Batch size 1, Avenue test frames at {w}$\times${h}, {gpu} GPU and {cpu} CPU, PyTorch "
             rf"{env['torch'].split('+')[0]}. Each run: {cols[0][1]['meta']['warmup_frames']} warm-up frames, then "
             rf"{len(jload(K / 'hiz_20261105/bench_seq_allcores.json')['reps'])}$\times$"
             rf"{cols[0][1]['meta']['frames_per_rep']} timed frames; each value is the {how} of the per-repetition "
             rf"medians over {cols[0][1]['nruns']}, {cols[1][1]['nruns']} and {cols[2][1]['nruns']} runs (columns left "
             rf"to right). TensorRT {trt_ver} FP16 engines for the two YOLO11n models; the image "
             rf"encoder stays in PyTorch because its TensorRT engine changed the embeddings (mean cosine "
             rf"{cos_mean:.3f} to the reference embeddings over {n_crops} crops). On the Ped1 development set the TensorRT detectors "
             rf"change the online micro AUC by {signed(d_trt['obj'], 2)} (object branch) and {signed(d_trt['grid'], 2)} (object "
             rf"+ grid) points; on the test sets by {trt_txt} (object branch / object + grid){over_txt}. All accuracy "
             rf"numbers in this paper use the PyTorch detectors. With PyTorch limited to {four['seq']['meta']['torch_threads']} threads: "
             rf"{f1(four['seq']['fps'])} FPS sequential, {f1(four['par']['fps'])} FPS with parallel flow. The host "
             rf"was shared (1-min load average {lo:g}--{hi:g}, recorded before and after the runs). Par.\ flow: the optical flow runs in "
             rf"a CPU thread, in parallel with the GPU stages. $^{{a}}$Time the main loop waits for the flow thread. $^{{b}}$Measured "
             rf"per frame; it includes steps not listed (cropping, attribute computation), so the stage values do "
             rf"not add up to it.")
    # v4 note (main_v4.tex): without the encoder check, the 4-thread rates and the load average, which go to the
    # supplement (gen/supp_speed_notes_v4.tex) verbatim
    notes_v4 = (rf"Batch size 1, Avenue test frames at {w}$\times${h}, {gpu} GPU and {cpu} CPU on a shared host, PyTorch "
                rf"{env['torch'].split('+')[0]}. Each run: {cols[0][1]['meta']['warmup_frames']} warm-up frames, then "
                rf"{len(jload(K / 'hiz_20261105/bench_seq_allcores.json')['reps'])}$\times$"
                rf"{cols[0][1]['meta']['frames_per_rep']} timed frames; each value is the {how} of the per-repetition "
                rf"medians over {cols[0][1]['nruns']}, {cols[1][1]['nruns']} and {cols[2][1]['nruns']} runs (columns left "
                rf"to right). TensorRT {trt_ver} FP16 engines for the two YOLO11n models; the image encoder stays in "
                rf"PyTorch (its TensorRT check, the rates with PyTorch limited to "
                rf"{four['seq']['meta']['torch_threads']} threads and the load average of the host: Supplementary "
                rf"Section~\ref{{ssec:latency}}). On the Ped1 development set the TensorRT detectors "
                rf"change the online micro AUC by {signed(d_trt['obj'], 2)} (object branch) and {signed(d_trt['grid'], 2)} (object "
                rf"+ grid) points; on the test sets by {trt_txt} (object branch / object + grid){over_txt}. All accuracy "
                rf"numbers in this paper use the PyTorch detectors. Par.\ flow: the optical flow runs in "
                rf"a CPU thread, in parallel with the GPU stages. $^{{a}}$Time the main loop waits for the flow thread. $^{{b}}$Measured "
                rf"per frame; it includes steps not listed (cropping, attribute computation), so the stage values do "
                rf"not add up to it.")
    for variant, nt in (("v3", notes), ("v4", notes_v4)):
        tex = "\n".join([
            r"% Generated by fig_src/make_tables_v3.py -- do not edit by hand.",
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Latency per frame (ms) of the object-branch pipeline, from frame decoding to the fused score (grid branch not included).}",
            r"\label{tab:speed}",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{2.5pt}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r" & \multicolumn{2}{c}{PyTorch} & TensorRT YOLO \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}",
            r"Stage & Sequential & Par.\ flow & Par.\ flow \\",
            r"\midrule", *lines, r"\bottomrule",
            r"\end{tabular}",
            r"\par\vspace{3pt}",
            rf"\parbox{{\columnwidth}}{{\scriptsize {nt}}}",
            r"\end{table}", ""])
        write(f"tab_speed_{variant}.tex", tex)
    prov4("tab_speed_v4.tex", "tabular and note values", "identical to tab_speed_v3.tex",
          "same variables (tables_v3_provenance.txt, 'speed' entries)")
    supp = (r"% Generated by fig_src/make_tables_v3.py -- do not edit by hand." "\n"
            rf"Notes of Table~\ref{{tab:speed}} of the paper that are given here only. The image "
            rf"encoder stays in PyTorch because its TensorRT engine changed the embeddings (mean cosine "
            rf"{cos_mean:.3f} to the reference embeddings over {n_crops} crops). With PyTorch limited to "
            rf"{four['seq']['meta']['torch_threads']} threads: "
            rf"{f1(four['seq']['fps'])} FPS sequential, {f1(four['par']['fps'])} FPS with parallel flow. The host "
            rf"was shared (1-min load average {lo:g}--{hi:g}, recorded before and after the runs)." "\n")
    write("supp_speed_notes_v4.tex", supp)
    for lab, v in (("TensorRT encoder mean cosine, crops", f"{cos_mean:.3f}, {n_crops}"),
                   ("4-thread FPS sequential / parallel", f"{f1(four['seq']['fps'])} / {f1(four['par']['fps'])}"),
                   ("host 1-min load average", f"{lo:g}--{hi:g}")):
        prov4("supp_speed_notes_v4.tex", lab, v, "tables_v3_provenance.txt : same 'speed' entries as tab_speed_v3.tex")
    return cols, four, d_test, over


# --------------------------------------------------------------------------------------------------- table 4
FRACS = ["0.1", "0.25", "0.5", "1.0"]


def fit_times():
    H = K / "hiz_20261105"
    out = {"gpu": {}, "cpu": {}}
    for ds, gpu_tag in (("ped2", "cuda"), ("avenue", "gpu")):
        for dev, tag in (("gpu", gpu_tag), ("cpu", "cpu")):
            p = H / f"fit_{ds}_{tag}.log"
            out[dev][ds] = float(re.search(r"ELAPSED_S ([\d.]+)", tload(p)).group(1))
            prov("setup", f"fit {ds} {dev} (s)", f"{out[dev][ds]:.2f}", rel(p))
    return out


HK = K / "hakem_20261106" / "vast_out"     # pre-registered G2 session (hakem_duzeltmeleri_on_kayit_2026-11-06.md)
SETUP_DS = ["ped2", "avenue", "shanghaitech", "ped1"]


def setup_g2():
    """Measured setup times of the G2 session: training-split feature extraction and fitting on the GPU, and the
    TensorRT export; host load averages from the idle records written before every measurement."""
    G = HK / "out" / "g2"
    elapsed = lambda p: float(re.search(r"ELAPSED_S ([\d.]+)", tload(p)).group(1))
    g2 = {}
    for ds in SETUP_DS:
        g2[ds] = {"extract": elapsed(G / f"extract_{ds}.log"), "fit": elapsed(G / f"fit_{ds}.log")}
        prov("setup", f"G2 feature extraction {ds} (s)", f"{g2[ds]['extract']:.1f}", rel(G / f"extract_{ds}.log"))
        prov("setup", f"G2 fit on GPU {ds} (s)", f"{g2[ds]['fit']:.1f}", rel(G / f"fit_{ds}.log"))
    run = tload(REPO / "vad_bench" / "vast" / "runs_hakem" / "run.sh")
    m = re.search(r"te=1; \[ \$ds = shanghaitech \] && te=(\d+)", run)
    if not m:
        raise SystemExit("runs_hakem/run.sh: train_every of the G2 extraction not found")
    te_sht = int(m.group(1))
    exp = tload(G / "export.log")
    tr = [float(x) for x in re.findall(r"Export complete \(([\d.]+)s\)", exp)]
    if len(tr) != 2:
        raise SystemExit("G2 export.log: expected two TensorRT exports")
    loads = [float(re.search(r"load average: ([\d.]+)", tload(p)).group(1)) for p in sorted(G.glob("idle_*.txt"))]
    # the GPU must be idle before every measurement (pre-registration S2): no compute process (the line after the
    # utilisation line is uptime's) and under 100 MiB in use
    busy = [p.name for p in sorted(G.glob("idle_*.txt"))
            if not (int(re.match(r"\d+ %, (\d+) MiB", tload(p).splitlines()[1]).group(1)) < 100
                    and "load average" in tload(p).splitlines()[2])]
    if busy:
        raise SystemExit(f"G2: GPU not idle before {busy}")
    env = jload(HK / "logs" / "env.json")
    prov("setup", "G2 TensorRT export (s)", " + ".join(map(str, tr)), rel(G / "export.log"))
    prov("setup", "G2 host 1-min load average (min-max)", f"{min(loads):.1f}-{max(loads):.1f}", rel(G) + "/idle_*.txt")
    return {"ds": g2, "te_sht": te_sht, "trt": tr, "loads": loads, "gpu": env["gpu"].replace("NVIDIA GeForce ", ""),
            "cpu": env["cpu"]}


def table_setup(fit, cols, g2):
    S = K / "kurulum_20261105"
    de = {}
    for ds in ["ped1", "ped2", "avenue", "shanghaitech"]:
        for fr in FRACS:
            p = S / f"dataeff_{ds}_{fr}.json"
            d = jload(p)
            if d["dataset"] != ds or str(d["frac"]) != fr:
                raise SystemExit(f"{p}: wrong dataset or fraction")
            m = [r["minutes"] for r in d["rows"]]
            v = [r["videos"] for r in d["rows"]]
            c = [r["causal"] for r in d["rows"]]
            if abs(statistics.fmean(c) - d["causal_mean"]) > 1e-12:
                raise SystemExit(f"{p}: causal_mean differs from the rows")
            # sample standard deviation over the draws (n - 1); the file's causal_std is the population value
            sd = statistics.stdev(c) if len(c) > 1 else 0.0
            nopose = sum("opose" not in r["fused"].split("+") for r in d["rows"])
            de[(ds, fr)] = {"mean": d["causal_mean"], "std": sd, "n": len(d["rows"]), "min": min(m),
                            "max": max(m), "videos": (min(v), max(v)), "macro": d["macro_mean"], "config": d["config"],
                            "frames": [r["frames"] for r in d["rows"]], "nopose": nopose,
                            "seeds": [r["seed"] for r in d["rows"]]}
            prov("setup", f"data efficiency {ds} f={fr}", f"{pct(d['causal_mean'])} +- {pct(sd)} (sample std), "
                                                          f"{min(m):.2f}-{max(m):.2f} min, n={len(d['rows'])}, "
                                                          f"draws without pose {nopose}", rel(p))
    tr = g2["trt"]

    # feature extraction, estimated by the pre-registered method (kurulum_olcumleri_on_kayit M2: per-frame latency x training
    # frames) from the PyTorch columns of the speed table (parallel flow and sequential); reported next to the
    # measurement of the G2 session
    lat = sorted(c["ms"]["total"] for name, c in cols if c["meta"]["config"] == "r2_base")
    if len(lat) != 2:
        raise SystemExit("speed: expected two PyTorch columns")
    extr = {}
    for ds in ("ped2", "avenue"):
        nfr = de[(ds, "1.0")]["frames"][0]
        extr[ds] = (nfr * lat[0] / 6e4, nfr * lat[-1] / 6e4)
        prov("setup", f"feature extraction estimate {ds} (min)", f"{extr[ds][0]:.2f}-{extr[ds][1]:.2f}",
             f"{nfr} training frames (dataeff_{ds}_1.0.json) x {lat[0]:.2f}-{lat[-1]:.2f} ms (speed table, PyTorch)")

    # (a) setup time
    a = []
    for ds in SETUP_DS:
        full = de[(ds, "1.0")]
        name = DSN[ds] + ("$^{c}$" if ds == "ped1" else "")
        extract = mins(g2["ds"][ds]["extract"] / 60) + ("$^{f}$" if ds == "shanghaitech" else "")
        fit_g = f1(g2["ds"][ds]["fit"]) + ("$^{g}$" if ds == "shanghaitech" else "")
        fit_c = f1(fit["cpu"][ds] / 60) if ds in fit["cpu"] else "--"
        total = mins((g2["ds"][ds]["extract"] + g2["ds"][ds]["fit"]) / 60)
        a.append(" & ".join([name, f"{mins(full['min'])} ({full['videos'][0]})", extract, fit_g, total, fit_c]) + r" \\")
    # (b) data efficiency
    b = []
    for ds in ["ped2", "avenue", "shanghaitech", "ped1"]:
        auc_c, min_c = [], []
        for fr in FRACS:
            d = de[(ds, fr)]
            s = pct(d["mean"]) + (f"$\\pm${pct(d['std'])}" if d["n"] > 1 else "") + ("$^{e}$" if d["nopose"] else "")
            auc_c.append(s)
            rng = mins(d["min"]) if mins(d["min"]) == mins(d["max"]) else f"{mins(d['min'])}--{mins(d['max'])}"
            min_c.append(rng)
        name = DSN[ds] + ("$^{c}$" if ds == "ped1" else "")
        b.append(" & ".join([name] + auc_c) + r" \\")
        b.append(" & ".join([r"\quad min"] + min_c) + r" \\")
    held = [ds for ds in ["ped1", "ped2", "avenue", "shanghaitech"]
            if de[(ds, "0.25")]["mean"] >= de[(ds, "1.0")]["mean"] - 0.02]
    failed = [ds for ds in ["ped1", "ped2", "avenue", "shanghaitech"] if ds not in held]
    fail_txt = "; ".join(rf"{DSN[ds]}: {signed(100 * (de[(ds, '0.25')]['mean'] - de[(ds, '1.0')]['mean']))} micro, "
                         rf"{signed(100 * (de[(ds, '0.25')]['macro'] - de[(ds, '1.0')]['macro']))} macro"
                         for ds in failed)
    nseed = de[("ped2", "0.1")]["n"]
    # draws are numbered 0..n-1 in every file with f < 1; the videos of draw s are chosen with
    # numpy default_rng(1000 + s) (vad_bench/vast/runs_r2/deploy_measures.py, subset())
    seeds = de[("ped2", "0.1")]["seeds"]
    if any(v["seeds"] != seeds for (ds_, fr_), v in de.items() if fr_ != "1.0") or seeds != list(range(nseed)):
        raise SystemExit("data efficiency: draw numbers differ between files; revise the setup table note")
    dm = REPO / "vad_bench" / "vast" / "runs_r2" / "deploy_measures.py"
    om = re.search(r"def subset\(.*?rng = np\.random\.default_rng\((\d+) \+ seed\)", tload(dm), re.S)
    if not om:
        raise SystemExit(f"{dm}: subset() generator not found")
    rng_off = int(om.group(1))
    if rng_off != 1000:     # the dataeff_*.json files were produced with default_rng(1000 + s)
        raise SystemExit(f"{dm}: subset() seed offset is {rng_off}, not the 1000 of the data-efficiency runs")
    prov("setup", "data efficiency: video draw generator offset", str(rng_off),
         f"{rel(dm)} : subset(), np.random.default_rng({rng_off} + seed)")
    nopose = [(ds, fr) for ds in ["ped2", "avenue", "shanghaitech", "ped1"] for fr in FRACS if de[(ds, fr)]["nopose"]]
    nopose_txt = ", ".join(rf"{DSN[ds]} $f={fr if fr != '1.0' else '1'}$: {de[(ds, fr)]['nopose']} of "
                           rf"{de[(ds, fr)]['n']}" for ds, fr in nopose)
    H = K / "hiz_20261105"
    gpu_old = jload(H / "env.json")["gpu"].replace("NVIDIA GeForce ", "")
    notes = (rf"(a) Normal video: all training videos of the dataset (number of videos in parentheses). Total: "
             rf"feature extraction plus GPU fitting, the setup of the object branch. Feature "
             rf"extraction: \texttt{{vadkit extract}} of the training videos (decoding, detector, pose, flow, encoder "
             rf"and attributes; one process, PyTorch). Fitting: \texttt{{vadkit fit}} from the extracted features "
             rf"(object densities and calibration; Python start-up and loading included). Extraction and GPU "
             rf"fitting: one session on an {g2['gpu']} with an {g2['cpu'].replace(' 32-Core Processor', '')} CPU, "
             rf"shared with other tenants (1-min load average {min(g2['loads']):.0f}--{max(g2['loads']):.0f} before "
             rf"the measurements; GPU idle). The pre-registered estimate of the extraction time, training frames "
             rf"({de[('ped2', '1.0')]['frames'][0]:,} and {de[('avenue', '1.0')]['frames'][0]:,}) times the "
             rf"PyTorch end-to-end latency of Table~\ref{{tab:speed}} ({f1(lat[0])}--{f1(lat[-1])}~ms), was "
             rf"{mins(extr['ped2'][0])}--{mins(extr['ped2'][1])}~min on Ped2 and "
             rf"{mins(extr['avenue'][0])}--{mins(extr['avenue'][1])}~min on Avenue. One-time, per GPU type: "
             rf"building the TensorRT engines of the two YOLO11n models took {f1(sum(tr) / 60)}~min. "
             rf"$^{{d}}$An earlier session on another host with an {gpu_old} (Ped2 and Avenue only; GPU fitting "
             rf"there took {f1(fit['gpu']['ped2'])} and {f1(fit['gpu']['avenue'])}~s); the number of CPU threads "
             rf"was not recorded. $^{{f}}$Every {g2['te_sht']}th training frame, as for the features of this paper. "
             rf"$^{{g}}$The 13 scene models. ")
    notes += (
             rf"(b) Online micro AUC (\%) of the object branch fitted on a random fraction $f$ of the normal training "
             rf"videos (at least two; on ShT per scene), mean$\pm$sample standard deviation over {nseed} draws ($f=1$: "
             rf"one run; draw $s={seeds[0]},\dots,{seeds[-1]}$ selects its videos with NumPy's default\_rng$({rng_off}+s)$), and the minutes of normal video used (range over draws). Pre-registered criterion, $f=0.25$ "
             rf"within 2 points of $f=1$: held on {len(held)} of 4 datasets ({fail_txt}). $^{{c}}$Development set. "
             rf"$^{{e}}$Some draws had no pose model and fuse velocity and appearance only ({nopose_txt}).")
    # v4 note (main_v4.tex); the sentences it drops go verbatim to gen/supp_setup_notes_v4.tex
    notes_v4 = (rf"(a) Normal video: all training videos of the dataset (number of videos in parentheses). Total: "
                rf"feature extraction plus GPU fitting, the setup of the object branch. Feature "
                rf"extraction: \texttt{{vadkit extract}} of the training videos (decoding, detector, pose, flow, encoder "
                rf"and attributes; one process, PyTorch). Fitting: \texttt{{vadkit fit}} from the extracted features "
                rf"(object densities and calibration; Python start-up and loading included). Extraction and GPU "
                rf"fitting: one session on an {g2['gpu']} with an {g2['cpu'].replace(' 32-Core Processor', '')} CPU, "
                rf"shared with other tenants (Supplementary Section~\ref{{ssec:timing}}). "
                rf"$^{{d}}$An earlier session on another host (Supplementary Section~\ref{{ssec:setup}}). "
                rf"$^{{f}}$Every {g2['te_sht']}th training frame, as for the features of this paper. "
                rf"$^{{g}}$The 13 scene models. "
                rf"(b) Online micro AUC (\%) of the object branch fitted on a random fraction $f$ of the normal training "
                rf"videos (at least two; on ShT per scene), mean$\pm$sample standard deviation over {nseed} draws ($f=1$: "
                rf"one run), and the minutes of normal video used (range over draws); plotted in Supplementary "
                rf"Fig.~\ref{{sfig:data_efficiency}}. Pre-registered criterion, $f=0.25$ "
                rf"within 2 points of $f=1$: held on {len(held)} of 4 datasets ({fail_txt}). $^{{c}}$Development set. "
                rf"$^{{e}}$Some draws had no pose model and fuse velocity and appearance only ({nopose_txt}).")
    for variant, nt in (("v3", notes), ("v4", notes_v4)):
        tex = "\n".join([
            r"% Generated by fig_src/make_tables_v3.py -- do not edit by hand.",
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Setting up a new scene: (a) setup time, (b) normal video needed.}",
            r"\label{tab:setup}",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"\multicolumn{6}{l}{\textit{(a) Setup time}} \\",
            r" & Normal & Feature & Fitting & Total & Fitting \\",
            r"Dataset & video, min & extr., min & GPU, s & GPU, min & CPU, min$^{d}$ \\",
            r"\midrule", *a, r"\bottomrule",
            r"\end{tabular}",
            r"\par\vspace{6pt}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"\multicolumn{5}{l}{\textit{(b) Normal video needed}} \\",
            r"Dataset & $f=0.1$ & $f=0.25$ & $f=0.5$ & $f=1$ \\",
            r"\midrule", *b, r"\bottomrule",
            r"\end{tabular}",
            r"\par\vspace{3pt}",
            rf"\parbox{{\columnwidth}}{{\scriptsize {nt}}}",
            r"\end{table}", ""])
        write(f"tab_setup_{variant}.tex", tex)
    prov4("tab_setup_v4.tex", "tabulars (a), (b) and note values", "identical to tab_setup_v3.tex",
          "same variables (tables_v3_provenance.txt, 'setup' entries)")
    # the note sentences of v3 that the v4 note drops, verbatim, for Supplementary Section S5
    supp = (r"% Generated by fig_src/make_tables_v3.py -- do not edit by hand." "\n"
            rf"Notes of Table~\ref{{tab:setup}} of the paper that are given here only. (a) Extraction and GPU "
            rf"fitting: one session on an {g2['gpu']} with an {g2['cpu'].replace(' 32-Core Processor', '')} CPU, "
            rf"shared with other tenants (1-min load average {min(g2['loads']):.0f}--{max(g2['loads']):.0f} before "
            rf"the measurements; GPU idle). The pre-registered estimate of the extraction time, training frames "
            rf"({de[('ped2', '1.0')]['frames'][0]:,} and {de[('avenue', '1.0')]['frames'][0]:,}) times the "
            rf"PyTorch end-to-end latency of Table~\ref{{tab:speed}} ({f1(lat[0])}--{f1(lat[-1])}~ms), was "
            rf"{mins(extr['ped2'][0])}--{mins(extr['ped2'][1])}~min on Ped2 and "
            rf"{mins(extr['avenue'][0])}--{mins(extr['avenue'][1])}~min on Avenue. One-time, per GPU type: "
            rf"building the TensorRT engines of the two YOLO11n models took {f1(sum(tr) / 60)}~min. "
            rf"$^{{d}}$An earlier session on another host with an {gpu_old} (Ped2 and Avenue only; GPU fitting "
            rf"there took {f1(fit['gpu']['ped2'])} and {f1(fit['gpu']['avenue'])}~s); the number of CPU threads "
            rf"was not recorded. "
            rf"(b) Draw $s={seeds[0]},\dots,{seeds[-1]}$ selects its videos with NumPy's default\_rng$({rng_off}+s)$." "\n")
    write("supp_setup_notes_v4.tex", supp)
    for lab, v in (("G2 host 1-min load average", f"{min(g2['loads']):.0f}--{max(g2['loads']):.0f}"),
                   ("extraction estimate Ped2 / Avenue (min)", f"{mins(extr['ped2'][0])}--{mins(extr['ped2'][1])} / "
                                                               f"{mins(extr['avenue'][0])}--{mins(extr['avenue'][1])}"),
                   ("TensorRT engine build (min)", f1(sum(tr) / 60)),
                   ("earlier session GPU fitting Ped2 / Avenue (s)", f"{f1(fit['gpu']['ped2'])} / {f1(fit['gpu']['avenue'])}"),
                   ("draw generator", f"default_rng({rng_off}+s), s={seeds[0]}..{seeds[-1]}")):
        prov4("supp_setup_notes_v4.tex", lab, v, "tables_v3_provenance.txt : same 'setup' entries as tab_setup_v3.tex")
    return de, tr, extr


# --------------------------------------------------------------------------------------------------- numbers
def numbers(auc, ours, comp_params, cols, fit, g2, de, d_test, over):
    n = {}      # accuracy macros per configuration and protocol: make_text_v3.py (VtObjStd..., VtGridOn..., ...)
    n["VthreeParamsObj"] = f1(ours["p_obj"])
    n["VthreeParamsGrid"] = f1(ours["p_grid"])
    # without the pose model: the pose coverage rule of Section V-E (reported, not adopted), on Ped1, Ped2 and ShT
    n["VthreeParamsObjNoPose"] = f1(ours["p_obj"] - ours["so"]["yolo11n-pose.pt"]["params_M"])
    n["VthreeWeightsMB"] = f1(ours["w_total"])
    n["VthreeAEParamsM"] = f1(ours["ae_n"] / 1e6, 2)
    vpc = comp_params["vpc"]
    n["VthreeVpcParams"] = f1(vpc)
    n["VthreeRatioVpc"] = f1(vpc / ours["p_grid"])
    n["VthreeRatioAccMin"] = f"{comp_params['accivad'] / ours['p_grid']:.0f}"
    vlm = sorted([comp_params["anyanomaly"], comp_params["anomalyruler"]])
    n["VthreeRatioVlmMin"] = f"{vlm[0] / ours['p_grid']:,.0f}"
    n["VthreeRatioVlmMax"] = f"{vlm[1] / ours['p_grid']:,.0f}"
    for key, (_, c) in zip(("Seq", "Par", "Trt"), cols):
        n[f"VthreeFps{key}"] = f1(c["fps"])
        n[f"VthreeFps{key}Min"] = f1(min(c["fps_runs"]))
        n[f"VthreeFps{key}Max"] = f1(max(c["fps_runs"]))
    for ds, tag in (("ped1", "PedOne"), ("ped2", "PedTwo"), ("avenue", "Avenue"), ("shanghaitech", "SHT")):
        n[f"VthreeFitGpu{tag}"] = f1(g2["ds"][ds]["fit"])
        n[f"VthreeExtrMeas{tag}"] = mins(g2["ds"][ds]["extract"] / 60)
        n[f"VthreeSetupMin{tag}"] = mins((g2["ds"][ds]["extract"] + g2["ds"][ds]["fit"]) / 60)
    n["VthreeFitCpuPedTwoMin"] = f1(fit["cpu"]["ped2"] / 60)
    n["VthreeFitCpuAvenueMin"] = f1(fit["cpu"]["avenue"] / 60)
    lab = jload(K / "avenue_etiket_20261105/diff.json")
    n["VthreeAvenueLabelDiff"] = f"{lab['cuhk_vs_vpc_summary']['n_differing']:,}"
    n["VthreeAvenueFrames"] = f"{lab['total_test_frames']:,}"
    for ds, tag in (("ped2", "PedTwo"), ("avenue", "Avenue")):
        d = de[(ds, "0.1")]
        n[f"VthreeMinTenth{tag}"] = f"{mins(d['min'])}--{mins(d['max'])}"
        n[f"VthreeAucTenth{tag}"] = pct(d["mean"])
        n[f"VthreeAucFull{tag}"] = pct(de[(ds, "1.0")]["mean"])
    for (ds, br), v in d_test.items():
        tag = {"ped2": "PedTwo", "avenue": "Avenue", "shanghaitech": "SHT"}[ds]
        n[f"VthreeTrt{'Grid' if br == 'grid' else 'Obj'}{tag}"] = signed(v, 2)
    n["VthreeTrtMargin"] = f"{TRT_MARGIN:g}"
    within = [abs(v) for k, v in d_test.items() if k not in over]
    n["VthreeTrtMaxWithin"] = f"{max(within):.2f}"
    lines = [r"% Generated by fig_src/make_tables_v3.py -- do not edit by hand. Sources: gen/tables_v3_provenance.txt"]
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in n.items()]
    write("numbers_v3.tex", "\n".join(lines) + "\n")
    for k, v in n.items():
        prov("numbers", k, v, "computed from the values above")


# --------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rep-agg", choices=["median", "mean"], default=REP_AGG)
    args = ap.parse_args()

    ev = Evidence()
    auc = ours_auc()
    sigma, kmed, warm = protocol_constants()
    ae_n, ae_secs = grid_ae()
    so_p = K / "boyut_20261105/size_ours.json"
    so = jload(so_p)
    p_obj = so["yolo11n.pt"]["params_M"] + so["yolo11n-pose.pt"]["params_M"] + so["MobileCLIP2-S0"]["visual_params_M"]
    p_grid = p_obj + ae_n / 1e6
    ae_MB = ae_n * 4 / 1e6
    # image tower as the pipeline runs it: fp32 (vadkit/perception/embed.py loads open_clip in fp32 and casts only the
    # output embeddings to fp16), 4 bytes per parameter; the YOLO weights are the released .pt files
    enc_n = round(so["MobileCLIP2-S0"]["visual_params_M"] * 1e6)
    enc_MB = enc_n * 4 / 1e6
    w_total = so["yolo11n.pt"]["file_MB"] + so["yolo11n-pose.pt"]["file_MB"] + enc_MB + ae_MB
    for k in ("yolo11n.pt/params_M", "yolo11n-pose.pt/params_M", "MobileCLIP2-S0/visual_params_M",
              "yolo11n.pt/file_MB", "yolo11n-pose.pt/file_MB", "MobileCLIP2-S0/onnx_MB",
              "MobileCLIP2-S0/total_params_incl_text_M"):
        a, b = k.split("/")
        prov("size", f"ours {k}", f"{so[a][b]:.6g}", rel(so_p))
    prov("size", "ours params object / with grid (M)", f"{p_obj:.3f} / {p_grid:.3f}", "sum of the above")
    prov("size", "ours MobileCLIP2-S0 image tower fp32 (MB)", f"{enc_MB:.3f}", "visual_params_M x 4 bytes")
    prov("size", "ours weights total (MB)", f"{w_total:.2f}", "YOLO files + image tower fp32 + grid AE fp32 "
                                                              "(4 bytes/parameter)")
    ours = {"so": so, "p_obj": p_obj, "p_grid": p_grid, "ae_n": ae_n, "ae_MB": ae_MB, "w_total": w_total,
            "enc_n": enc_n, "enc_MB": enc_MB}

    comp_params = table_main(ev, auc, (p_obj, p_grid), sigma)
    table_online(auc, sigma, kmed, warm)
    fit = fit_times()
    g2 = setup_g2()
    cols, _, d_test, over = table_speed(args.rep_agg, auc)
    table_size(ev, ours, g2, ae_secs, cols)
    de, _, _ = table_setup(fit, cols, g2)
    numbers(auc, ours, comp_params, cols, fit, g2, de, d_test, over)
    w = max(len(p[1]) for p in PROV)
    text = [f"# Provenance of every number in gen/*_v3.tex (written by fig_src/make_tables_v3.py; switch: "
            f"rep_agg={args.rep_agg})",
            "# table | label | value | source"]
    text += [f"{t:7s} | {lab:{w}s} | {v} | {s}" for t, lab, v, s in PROV]
    write("tables_v3_provenance.txt", "\n".join(text) + "\n")
    w4 = max(len(p[1]) for p in PROV4)
    text = ["# Provenance of the v4 outputs of fig_src/make_tables_v3.py (main_v4.tex, supp_v4.tex): gen/tab_main_v4.tex, "
            "gen/tab_size_v4.tex, gen/supp_tab_size_v4.tex, gen/online_cells_v4.json, gen/tab_speed_v4.tex, "
            "gen/supp_speed_notes_v4.tex, gen/tab_setup_v4.tex, gen/supp_setup_notes_v4.tex. Every value comes from "
            "the variables of the v3 tables (gen/tables_v3_provenance.txt).",
            "# output | label | value | source"]
    text += [f"{o:28s} | {lab:{w4}s} | {v} | {s}" for o, lab, v, s in PROV4]
    write("tables_v4_provenance.txt", "\n".join(text) + "\n")


if __name__ == "__main__":
    main()
