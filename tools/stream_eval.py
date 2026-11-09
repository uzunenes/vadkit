"""Stream every test video of a dataset through the combined streaming detector (object branch + native grid branch),
record per frame the calibrated channel scores -log p_j, the online fused score and the objects, then select the
frame of the qualitative figure by the pre-registered rule and dump its grid maps
(docs/tez_notlari/birlesik_hiz_on_kayit_2026-11-08.md, section 3).

    python tools/stream_eval.py --config vadkit/configs/r2/base.yaml --frames /workspace/frames/avenue --dataset avenue \
        --state STATE --grid-state GSTATE --grid-run CK/runs/avenue_2d_noadv --gt /workspace/cache/avenue/gt.json \
        --out OUT/stream_avenue

Outputs: channels.npz (per video: '<channel>/<video>' calibrated scores, 'score/<video>' online fused score, NaN
before the warm-up, 'gt/<video>'), objects.npz (boxes, classes, sources, poses and per-object -log p of the selected
video), selected.npz (the selected frame: BGR image, grid maps I / Fs / Fd as 48 x 48 local means), result.json
(causal micro / macro AUC of object and object + grid from these streamed scores, the selection and its inputs).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vadkit.eval.protocol import evaluate
from vadkit.grid.stream import GridBranch, GridState
from vadkit.io.frames import frame_paths, read, resize_width
from vadkit.pipeline.offline import FittedState, Perception, gap_of, load_config, object_rows
from vadkit.pipeline.stream import StreamingVAD
from vadkit.temporal.median import causal_median

WARMUP = 4


def stream_video(sv: StreamingVAD, paths: list[Path], keep: int | None = None):
    """Per frame: FrameResult, objects and per-object -log p per attribute; with keep=t, also the grid maps at t."""
    sv.reset()
    rows, objs_all, maps = [], [], None
    for t, p in enumerate(paths):
        bgr0 = read(p, 10 ** 6)
        if sv.grid is not None:
            sv.grid.keep_maps = keep == t
            raw = sv.grid.step(bgr0)
        bgr = resize_width(bgr0, int(sv.config.get("width", 640)))
        objs = sv.perception(bgr)
        r = sv.step_objects(t, objs, bgr.shape[:2], raw if sv.grid is not None else None)
        per_obj = {}
        for a, st in sv.attrs.items():
            X = object_rows(st.extractor, objs, tuple(bgr.shape[:2]))
            per_obj[a] = -st.density.logpdf(X) if len(X) else np.zeros(0)
        rows.append(r)
        objs_all.append((objs, per_obj, bgr.shape[:2]))
        if keep == t:
            maps = {k: v.cpu().numpy() for k, v in sv.grid.maps.items()}
            maps["frame"] = bgr0
    return rows, objs_all, maps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--grid-state", required=True)
    ap.add_argument("--grid-run", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(a.config)
    state = FittedState.load(a.state, a.device)
    sv = StreamingVAD(state, cfg, gap_of(cfg, a.dataset), None, a.device,
                      grid=GridBranch(GridState.load(a.grid_state), a.grid_run, a.device))
    sv.perception = Perception(cfg, sv.gap, a.device)
    gt = {k: np.asarray(v, np.uint8) for k, v in json.loads(Path(a.gt).read_text()).items()}
    videos = frame_paths(a.frames, "testing")
    if set(videos) != set(gt):
        raise SystemExit(f"videos {sorted(set(videos) ^ set(gt))} are not in both the frames and the gt")
    ch, z = {}, {}
    for v, paths in videos.items():
        if len(paths) != len(gt[v]):
            raise SystemExit(f"{v}: {len(paths)} frames, {len(gt[v])} labels")
        rows, _, _ = stream_video(sv, paths)
        names = list(rows[-1].channels)
        for n in names:
            z.setdefault(n, {})[v] = np.array([r.channels.get(n, np.nan) for r in rows])
        ch[f"score/{v}"] = np.array([np.nan if r.score is None else r.score for r in rows])
        ch[f"gt/{v}"] = gt[v]
        print(v, len(paths), flush=True)
    for n, d in z.items():
        for v, s in d.items():
            ch[f"{n}/{v}"] = s
    np.savez_compressed(out / "channels.npz", **ch)

    # accuracy of the streamed scores (the evaluate() protocol: fused raw sums, causal median 9, warmup 4)
    obj_names = [n for n in z if n not in ("I", "Fs", "Fd")]
    fill = lambda s: np.nan_to_num(s, nan=0.0)                 # t < gap: grid channels absent (warm-up frames)
    fused = lambda names: {v: sum(fill(z[n][v]) for n in names) for v in videos}
    smooth = lambda x: causal_median(x, 9)
    res = {"dataset": a.dataset, "channels": list(z)}
    for name, names in (("object", obj_names), ("object+grid", list(z))):
        e = evaluate(fused(names), gt, WARMUP, smooth)
        res[name] = {"causal": e["causal"], "causal_macro": e["causal_macro"], "offline": e["offline"]}
    online = {v: ch[f"score/{v}"] for v in videos}
    check = {v: np.r_[np.full(WARMUP, np.nan), smooth(sum(fill(z[n][v]) for n in z)[WARMUP:])] for v in videos}
    res["max_abs_stream_vs_offline_fused"] = float(max(np.nanmax(np.abs(online[v] - check[v])) for v in videos))

    # selection (pre-registered): video with the median per-video causal AUC of the online object + grid score among
    # videos with both classes; frame: highest online score inside the video's first annotated event
    from sklearn.metrics import roc_auc_score
    per = {v: roc_auc_score(gt[v][WARMUP:], online[v][WARMUP:]) for v in sorted(videos) if 0 < gt[v][WARMUP:].mean() < 1}
    order = sorted(per, key=lambda v: (per[v], v))
    vsel = order[len(order) // 2]
    y = gt[vsel]
    start = int(np.argmax(y[WARMUP:] > 0)) + WARMUP
    end = start + int(np.argmax(y[start:] == 0)) if (y[start:] == 0).any() else len(y)
    tsel = start + int(np.nanargmax(online[vsel][start:end]))
    res["selection"] = {"video": vsel, "frame": tsel, "video_auc": per[vsel], "n_videos": len(per),
                        "first_event": [start, end - 1], "per_video_auc": per}
    rows, objs, maps = stream_video(sv, videos[vsel], keep=tsel)
    o, per_obj, hw = objs[tsel]
    np.savez_compressed(out / "selected.npz", video=vsel, frame=tsel, frame_hw=np.array(hw),
                        box=o.box, cls=o.cls, src=o.src, conf=o.conf,
                        pose=o.pose if o.pose is not None else np.zeros((len(o), 34), np.float32),
                        pose_ok=o.pose_ok if o.pose_ok is not None else np.zeros(len(o), np.uint8),
                        **{f"nlogp/{k}": v for k, v in per_obj.items()},
                        **{f"map/{k}": v for k, v in maps.items() if k != "frame"}, image=maps["frame"])
    same = max(abs((r.score or 0) - (0 if np.isnan(online[vsel][t]) else online[vsel][t])) for t, r in enumerate(rows))
    res["selection"]["replay_max_abs_diff"] = float(same)
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("object", "object+grid", "max_abs_stream_vs_offline_fused")}),
          res["selection"]["video"], res["selection"]["frame"])


if __name__ == "__main__":
    main()
