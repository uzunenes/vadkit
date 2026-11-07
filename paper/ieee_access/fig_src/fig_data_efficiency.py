"""Figure: online (causal) micro AUC vs minutes of normal training video (M2, kurulum_olcumleri_on_kayit_2026-11-05.md).

Small multiples, one panel per dataset (own y-range), log x. Mean +- sample std (n - 1) over 5 video draws, as in
gen/tab_setup_v3.tex; dashed line = all training videos; x ticks = mean minutes, rounded half up as in the table. Source: paper/ieee_access/kanit/kurulum_20261105/dataeff_<ds>_<f>.json.
Run: python paper/ieee_access/fig_src/fig_data_efficiency.py
"""
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kanit" / "kurulum_20261105"
OUT = ROOT / "fig" / "data_efficiency.png"
DS = [("ped1", "UCSD Ped1"), ("ped2", "UCSD Ped2"), ("avenue", "CUHK Avenue"), ("shanghaitech", "ShanghaiTech")]
FR = ["0.1", "0.25", "0.5", "1.0"]
SERIES, REF = "#2a78d6", "#52514e"          # reference palette slot 1; secondary text ink for the reference line
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def half_up(x, d):
    """Minutes rounded half up, as in gen/tab_setup_v3.tex (4.25 -> 4.3)."""
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 4, figsize=(10, 2.6), constrained_layout=True)
for ax, (ds, name) in zip(axes, DS):
    rows = [json.loads((SRC / f"dataeff_{ds}_{f}.json").read_text()) for f in FR]
    mins = np.array([np.mean([r["minutes"] for r in d["rows"]]) for d in rows])
    mu = np.array([d["causal_mean"] for d in rows]) * 100
    sd = np.array([np.std([r["causal"] for r in d["rows"]], ddof=1) if len(d["rows"]) > 1 else 0.0
                   for d in rows]) * 100
    full = mu[-1]
    ax.axhline(full, color=REF, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.fill_between(mins, mu - sd, mu + sd, color=SERIES, alpha=0.15, lw=0, zorder=2)
    ax.plot(mins, mu, color=SERIES, lw=2, marker="o", ms=5, mec="white", mew=1.5, zorder=3)
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks(mins, [half_up(m, 1) if m < 10 else half_up(m, 0) for m in mins])
    ax.set_xlim(mins[0] / 1.35, mins[-1] * 1.35)
    ax.set_title(name, fontsize=9.5, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_xlabel("normal video used (min)")
    ax.annotate(f"all videos {full:.1f}", (mins[-1], full), xytext=(0, 6), textcoords="offset points",
                ha="right", fontsize=8, color=MUTED)
    above = mu[0] >= full
    ax.annotate(f"{mu[0]:.1f}", (mins[0], mu[0]), xytext=(0, 8 if above else -13), textcoords="offset points",
                ha="center", fontsize=8, color=INK)
    lo, hi = min((mu - sd).min(), full), max((mu + sd).max(), full)
    pad = max(1.0, (hi - lo) * 0.25)
    ax.set_ylim(lo - pad, hi + pad)
axes[0].set_ylabel("online micro AUC (%)")
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, facecolor="#fcfcfb")
print(OUT)
