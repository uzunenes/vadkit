"""Accuracy vs test-time model size (parameters, log scale), Avenue and ShanghaiTech, with the values of Table 1
(competitors: published values; ours: standard (offline) protocol, for comparability with the published values).

Every number comes from the same inputs and code as gen/tab_main_v3.tex (fig_src/make_tables_v3.py is imported):
competitor AUCs, supervision blocks, Avenue label sets and parameter counts (incl. lower bounds and AccI-VAD's ShT
configuration without CLIP) through tables_v3_inputs.json; ours from kanit/boyut_20261105/size_ours.json plus the grid
autoencoder of make_tables_v3.grid_ae(), AUCs from kanit/r2_20261105/res/base ('offline' = standard protocol: per-video
min-max + Gaussian). Both of our configurations are drawn on both panels: object branch (circle, training-free block)
and object + grid (triangle, one-class block). Our online results are not plotted (they are in tab:online;
competitors are compared only through their published values and are not reported under the online protocol). Lower bounds are drawn with an arrow to the right; Avenue values computed with the
interval label file (set B) carry a superscript b.
Run: python paper/ieee_access/fig_src/fig_size_accuracy.py
"""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("make_tables_v3", HERE / "make_tables_v3.py")
mt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt)

ROOT = HERE.parent
OUT = ROOT / "fig" / "size_accuracy.png"
BLUE, GRAY, ORANGE = "#2a78d6", "#3a3936", "#eb6834"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True})

ev = mt.Evidence()
so = mt.jload(mt.K / "boyut_20261105/size_ours.json")
ours_obj = so["yolo11n.pt"]["params_M"] + so["yolo11n-pose.pt"]["params_M"] + so["MobileCLIP2-S0"]["visual_params_M"]
ae_n, _ = mt.grid_ae()
ours_grid = ours_obj + ae_n / 1e6
auc = mt.ours_auc()
DSK = {"avenue": "avenue", "shanghaitech": "shanghai"}      # dataset key in the per-dataset size files


def params_for(c, ds):
    """(parameters in M, lower bound?) of competitor c for dataset ds, as in the tables."""
    sp = c.get("params_sht_config") if ds == "shanghaitech" and "params_sht_config" in c else c["params"]
    if sp["bound"] == "range":                     # per-dataset models (AED-MAE): take this dataset's model
        t = [t for t in sp["terms"] if t["path"][0] == DSK[ds]]
        if len(t) != 1:
            raise SystemExit(f"{c['name']}: no {ds} model in the range terms")
        return ev.term(t[0], "fig", f"{c['name']} params {ds}"), False
    kind, val = mt.combine(ev, sp, "fig", f"{c['name']} params {ds}")
    return val, kind == "ge"


# label offsets (points) where the default (centred above the marker) would overlap a neighbour. On ShT, STG-NF and
# AccI-VAD have the same AUC and nearly the same size: STG-NF is labelled on the left, AccI-VAD after the tip of its
# lower-bound arrow, so that the arrow reads as AccI-VAD's (positions of the markers unchanged)
LABEL_POS = {("avenue", "AccI-VAD"): (-6, -3, "right"), ("shanghaitech", "STG-NF"): (-7, -3, "right"),
             ("shanghaitech", "VideoPatchCore"): (0, -12, "center"), ("shanghaitech", "AccI-VAD"): (20, -3, "left")}
STYLE = {"training_free": dict(marker="o", color=GRAY), "one_class_trained": dict(marker="^", color=GRAY),
         "synthetic_or_pseudo_anomaly": dict(marker="s", color=ORANGE)}
rows = []
for c in ev.inp["competitors"]:
    p = ev.lit_paper(*c["lit"])
    lab = ev.label_set(c["avenue_label_key"])
    for ds in ("avenue", "shanghaitech"):
        v = p[mt.LIT_KEY[ds]]
        if v is None:
            continue
        n, lb = params_for(c, ds)
        rows.append((ds, c["name"] + ("$^{b}$" if ds == "avenue" and lab == "B" else ""), c["block"], n, lb, float(v)))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
for ax, ds, name in ((axes[0], "avenue", "CUHK Avenue"), (axes[1], "shanghaitech", "ShanghaiTech")):
    for d, label, block, n, lb, a in rows:
        if d != ds:
            continue
        st = STYLE[block]
        ax.scatter([n], [a], s=46, facecolor=st["color"], edgecolor="white", lw=1.2, marker=st["marker"], zorder=3)
        if lb:
            ax.annotate("", (n * 1.9, a), (n * 1.08, a), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
        dx, dy, ha = LABEL_POS.get((ds, label.split("$")[0]), (0, 7, "center"))
        ax.annotate(label, (n, a), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=7.5, color=INK)
    std = {b: float(mt.pct(auc[(ds, b, "standard")])) for b in ("obj", "grid")}   # rounded as in Table 1
    hi = max(std, key=std.get)
    for b, p, mk, lab in (("obj", ours_obj, "o", "Ours, object branch"), ("grid", ours_grid, "^", "Ours, object + grid")):
        ax.scatter([p], [std[b]], s=64, color=BLUE, edgecolor="white", lw=1.2, marker=mk, zorder=4)
        ax.annotate(f"{lab} {std[b]:.1f}", (p, std[b]), xytext=(9, 3 if b == hi else -10), textcoords="offset points",
                    fontsize=8, color=BLUE, fontweight="bold")
    ys = [r[5] for r in rows if r[0] == ds] + list(std.values())
    ax.set_ylim(min(ys) - 0.8, max(ys) + 0.6)
    ax.set_xscale("log")
    ax.set_xlim(1, 60000)
    ax.grid(color=GRID, lw=0.8)
    ax.set_title(name, loc="left", fontsize=9.5, color=INK)
    ax.set_xlabel("parameters at test time (millions, log scale)")
axes[0].set_ylabel("frame micro AUC (%)")
h = [plt.Line2D([], [], ls="", marker="o", color=GRAY, label="training-free (published)"),
     plt.Line2D([], [], ls="", marker="^", color=GRAY, label="one-class (published)"),
     plt.Line2D([], [], ls="", marker="s", color=ORANGE, label="synthetic-anomaly training (published)"),
     plt.Line2D([], [], ls="", marker="o", color=BLUE, label="ours, object branch (training-free)"),
     plt.Line2D([], [], ls="", marker="^", color=BLUE, label="ours, object + grid (one-class)")]
fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.09), fontsize=8)
fig.savefig(OUT, dpi=200, facecolor=SURF, bbox_inches="tight")
print(OUT, round(ours_obj, 2), round(ours_grid, 2))
for r in rows:
    print(r)
