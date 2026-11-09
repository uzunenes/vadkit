"""Figure 3 (qualitative): one Avenue test frame chosen by the pre-registered rule
(docs/tez_notlari/birlesik_hiz_on_kayit_2026-11-08.md, section 4) and the per-channel scores of its video.

Inputs (written by tools/stream_eval.py on the vast session, copied to kanit/birlesik_20261108/vast_out/out/stream/avenue):
channels.npz, selected.npz, result.json. Frames are not stored in the repository's evidence: selected.npz holds the
one displayed frame (dataset frame; do not copy it to the public repository).

    python fig_src/fig_qualitative_v4.py      # -> fig/qualitative_v4.pdf, gen/numbers_qual_v4.tex
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7, "font.family": "sans-serif"})
ROOT = Path(__file__).resolve().parents[1]
K = ROOT / "kanit/birlesik_20261108/vast_out/out/stream/avenue"
WARMUP = 4
CH = [("ovel", "velocity"), ("opose", "pose"), ("oapp", "appearance"),
      ("I", "grid residual (I)"), ("Fs", "grid flow speed (Fs)"), ("Fd", "grid flow direction (Fd)")]
COL = {"ovel": "#1f77b4", "opose": "#2ca02c", "oapp": "#9467bd", "I": "#d62728", "Fs": "#ff7f0e", "Fd": "#8c564b"}
SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9),
            (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]


def keypoints(box: np.ndarray, rel: np.ndarray) -> np.ndarray:
    """Normalised keypoints (17, 2) from the box-relative pose row (vadkit/perception/pose.py); NaN where the pose
    model returned (0, 0) (keypoint not found)."""
    c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    kp = c + rel.reshape(17, 2) * max(box[3] - box[1], 1e-3)
    kp[np.all(np.abs(kp) < 1e-4, 1)] = np.nan
    return kp


def main(k: Path = K, out: Path = ROOT / "fig/qualitative_v4.pdf", macros: bool = True):
    sel = np.load(k / "selected.npz")
    z = np.load(k / "channels.npz")
    res = json.loads((k / "result.json").read_text())
    dsname = {"avenue": "Avenue", "ped2": "Ped2"}.get(res["dataset"], res["dataset"])
    v, t0 = str(sel["video"]), int(sel["frame"])
    assert (v, t0) == (res["selection"]["video"], res["selection"]["frame"])
    img = cv2.cvtColor(sel["image"], cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    names = [c for c, _ in CH]
    names = [c for c in names if f"{c}/{v}" in z.files]
    curves = {c: z[f"{c}/{v}"] for c in names}
    score, gt = z[f"score/{v}"], z[f"gt/{v}"]
    n = len(score)
    x = np.arange(n)
    # dominant grid channel at the frame -> its 48 x 48 local-mean map, upsampled to the colour frame
    gsel = max(("I", "Fs", "Fd"), key=lambda c: curves[c][t0])
    hm = cv2.resize(sel[f"map/{gsel}"], (W, H), interpolation=cv2.INTER_LINEAR)
    lo, hi = np.percentile(hm, [50, 99.9])
    hm = np.clip((hm - lo) / (hi - lo + 1e-12), 0, 1)

    fig = plt.figure(figsize=(7.16, 4.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 1], width_ratios=[1.0, 1.0], hspace=0.12, wspace=0.04)
    # top left: frame + detections + pose
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img)
    obj_nlogp = {c: sel[f"nlogp/{c}"] for c in ("ovel", "opose", "oapp") if f"nlogp/{c}" in sel.files}
    obj_ch = {c: curves[c][t0] for c in obj_nlogp}
    top_c = max(obj_ch, key=obj_ch.get) if obj_ch else None
    top_i = int(np.argmax(obj_nlogp[top_c])) if top_c and len(obj_nlogp[top_c]) else -1
    for i, (b, src) in enumerate(zip(sel["box"], sel["src"])):
        x0, y0, x1, y1 = b * [W, H, W, H]
        hl = i == top_i and top_c != "opose"
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.6 if hl else 0.8,
                               ec="#ffd400" if hl else ("#00e5ff" if src == 0 else "#ffffff"),
                               ls="-" if src == 0 else "--"))
        if sel["pose_ok"][i]:
            kp = keypoints(b, sel["pose"][i]) * [W, H]
            for a, c in SKELETON:
                ax.plot(kp[[a, c], 0], kp[[a, c], 1], lw=0.8, color="#ff4fd8" if not (i == top_i and top_c == "opose")
                        else "#ffd400")
    ax.set_title(f"(a) {dsname} video {v}, frame {t0}: objects and poses", fontsize=6.5)
    ax.axis("off")
    # top right: grid heat map over the grey frame
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(cv2.cvtColor(sel["image"], cv2.COLOR_BGR2GRAY), cmap="gray")
    ax2.imshow(hm, cmap="inferno", alpha=0.55)
    label = dict(CH)[gsel]
    ax2.set_title(f"(b) grid branch: {label}", fontsize=6.5)
    ax2.axis("off")
    # bottom: channel scores, fused online score, dominant channel strip
    ax3 = fig.add_subplot(gs[1, :])
    ev = np.flatnonzero(np.diff(np.r_[0, gt, 0]))
    for a, b in zip(ev[::2], ev[1::2]):
        ax3.axvspan(a, b - 1, color="0.85", lw=0)
    for c, lab in [(c, l) for c, l in CH if c in curves]:
        ax3.plot(x[WARMUP:], curves[c][WARMUP:], lw=0.7, color=COL[c], label=f"$-\\log p$ {lab}")
    ax3.set_xlim(0, n - 1)
    ax3.set_ylabel("$-\\log p_j$")
    ax3.set_xlabel("frame")
    ax4 = ax3.twinx()
    ax4.plot(x, score, color="k", lw=1.2, label="online fused score")
    ax4.set_ylabel("online fused score")
    ax3.axvline(t0, color="k", ls=":", lw=0.8)
    # strip: channel with the largest -log p per frame
    dom = np.array([np.nanargmax([curves[c][t] for c in names]) if t >= WARMUP else -1 for t in range(n)])
    y0 = ax3.get_ylim()[1]
    ax3.set_ylim(top=y0 * 1.12)
    cmap = ListedColormap([COL[c] for c in names])
    ax3.imshow(np.ma.masked_less(dom[None], 0), aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(names) - 0.5,
               extent=(-0.5, n - 0.5, y0 * 1.03, y0 * 1.11), interpolation="nearest")
    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax4.get_legend_handles_labels()
    ax3.legend(h1 + h2 + [Patch(color="0.85")], l1 + l2 + ["annotated anomaly"], ncol=4, fontsize=5.5,
               loc="upper left", bbox_to_anchor=(0, -0.28), frameon=False)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print("wrote", out)
    m = {"VqVideo": v, "VqFrame": str(t0), "VqVideoAuc": f"{100 * res['selection']['video_auc']:.1f}",
         "VqNVideos": str(res["selection"]["n_videos"]), "VqGridMap": label,
         "VqTopChannel": dict(CH)[names[int(dom[t0])]]}
    if macros:
        (ROOT / "gen/numbers_qual_v4.tex").write_text(
            "% generated by fig_src/fig_qualitative_v4.py from kanit/birlesik_20261108/vast_out/out/stream/avenue\n"
            + "".join(f"\\newcommand{{\\{kk}}}{{{val}}}\n" for kk, val in m.items()))
    print(m)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:                  # smoke test: input dir, output pdf, no macros
        main(Path(sys.argv[1]), Path(sys.argv[2]), False)
    else:
        main()
