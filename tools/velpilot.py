"""Light velocity pilot (Nov 2026): is the light velocity attribute limited by the flow, the feature construction
or the boxes? Every step is causal: a feature at frame t uses frames t-gap..t only, and frames t < gap get none.

    # boxes of another detector, same rules as tools/objfeat2.py (detector boxes + motion-contour boxes)
    python tools/velpilot.py boxes   --frames F/avenue --out P/boxes/y11x/avenue --det yolo11x.pt --gap 2
    # flow between t-gap and t, 8-bin box velocity (tools/objfeat2.velocity) for every box source
    python tools/velpilot.py flowvel --frames F/avenue --flow raft --src y11n=FEAT2/avenue y11x=P/boxes/y11x/avenue \
        --out P/vel/avenue --gap 2 [--train-every 5]
    # YOLO + ByteTrack, box-trajectory velocity (no optical flow)
    python tools/velpilot.py track   --frames F/avenue --out P/vel/avenue/track --gap 2
    # NIW score of one variant (normal training videos only), causal median 9 after warmup 4, micro/macro AUC
    python tools/velpilot.py score   --feat P/vel/avenue/raft/y11n --gt C/avenue/gt.json --cons c8 --out res.json

Feature constructions, all derived at scoring time from the stored per-box vector:
  c8   8-bin velocity of tools/objfeat2.py: mean L1 flow magnitude per orientation bin over the box pixels
  c1   c8.sum() = mean L1 magnitude over the box (AccI-VAD's Ped2 setting; here one rule for every dataset)
  c8h  c8 / box height in pixels (perspective normalisation; AccI-VAD applies it to ShT persons only)
  c1h  c1 / box height in pixels
  trk  stored track vector [vx/h, vy/h, speed/h, dlog(h)/dt] (track subcommand only)
--train-every N keeps features only on training frames t with t % N == 0, the same subsample for every variant.
No setting here is chosen on test labels; the pilot reports every variant.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ----------------------------------------------------------------------------------------------- frames
def frame_paths(frames: Path, split: str):
    """Frames in TIME order. vad_bench/vast/hazirla_frames.py names them 000.jpg ... 999.jpg, 1000.jpg, so a plain
    sorted() would put 1000.jpg after 100.jpg (the bug of tools/objfeat2.py before 4 Nov 2026); sort numerically."""
    root = frames / split / "frames"
    return {vd.name: sorted(vd.glob("*.jpg"), key=lambda p: int(p.stem))
            for vd in sorted(p for p in root.iterdir() if p.is_dir())}


def read(p: Path, width: int):
    im = cv2.imread(str(p))
    if im.shape[1] > width:   # identical to tools/objfeat2.py
        im = cv2.resize(im, (width, int(round(im.shape[0] * width / im.shape[1]))), interpolation=cv2.INTER_AREA)
    return im


def wanted(t: int, gap: int, split: str, every: int) -> bool:
    return t >= gap and (split != "training" or t % every == 0)


def save_npz(dst: Path, **arrays):
    """Atomic write: a killed process never leaves a truncated file that the exists()-based resume would skip."""
    tmp = dst.with_name(dst.stem + ".part.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, dst)


def write_timing(path: Path, tim: dict):
    if any(v for k, v in tim.items() if k in ("frames", "pairs")):     # a pure resume writes nothing
        path.write_text(json.dumps(tim, indent=1))


def shard(items, spec: str):
    i, n = map(int, spec.split("/"))
    return items[i::n]


# ----------------------------------------------------------------------------------------------- flow back-ends
NEUFLOW_WH = (768, 432)


class Flow:
    """flow(prev_bgr_list, cur_bgr_list) -> list of (H, W, 2) float32 flows in pixels of the processed frame."""

    def __init__(self, kind: str, device: str, neuflow_dir: Path | None = None, batch: int = 8):
        self.kind, self.dev, self.batch = kind, device, batch
        if kind == "dis":
            self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)   # as tools/objfeat2.py
        elif kind == "raft":
            from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
            w = Raft_Large_Weights.C_T_SKHT_V2      # Chairs+Things+Sintel+KITTI+HD1K, like neuflow_mixed
            self.net = raft_large(weights=w).to(device).eval()
            self.tf = w.transforms()
        elif kind == "neuflow":
            sys.path.insert(0, str(neuflow_dir))
            from NeuFlow.neuflow import NeuFlow
            from NeuFlow.backbone_v7 import ConvBlock
            net = NeuFlow().to(device)
            ck = torch.load(neuflow_dir / "neuflow_mixed.pth", map_location=device, weights_only=True)
            net.load_state_dict(ck["model"], strict=True)
            for m in net.modules():          # conv+bn fusion exactly as the official infer.py
                if type(m) is ConvBlock:
                    m.conv1 = _fuse(m.conv1, m.norm1); m.conv2 = _fuse(m.conv2, m.norm2)
                    delattr(m, "norm1"); delattr(m, "norm2"); m.forward = m.forward_fuse
            self.net = net.eval().half()
            self.shape = None
        else:
            raise ValueError(kind)

    @staticmethod
    def _pad(x: torch.Tensor, mult: int):
        H, W = x.shape[-2:]
        ph, pw = (-H) % mult, (-W) % mult
        return torch.nn.functional.pad(x, (0, pw, 0, ph), mode="replicate"), H, W

    @torch.no_grad()
    def __call__(self, prevs, curs):
        if self.kind == "dis":
            return [self.dis.calc(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY), cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), None)
                    for p, c in zip(prevs, curs)]
        out = []
        for i in range(0, len(prevs), self.batch):
            P, C = prevs[i:i + self.batch], curs[i:i + self.batch]
            n = len(P)
            if self.kind == "raft":
                a = torch.stack([torch.from_numpy(x[:, :, ::-1].copy()).permute(2, 0, 1) for x in P]).to(self.dev)
                b = torch.stack([torch.from_numpy(x[:, :, ::-1].copy()).permute(2, 0, 1) for x in C]).to(self.dev)
                a, b = self.tf(a, b)
                a, H, W = self._pad(a, 8); b, _, _ = self._pad(b, 8)
                f = self.net(a, b)[-1][:, :, :H, :W]
            else:   # neuflow: RGB in [0, 255] (its training loader uses PIL; the infer.py demo passes BGR), fp16,
                    # at its official inference size 768x432 (infer.py); fixed (B, H, W) after init_bhwd. Chosen before any
                    # AUC: at native Ped2 size (240x360) its flow agreed poorly with RAFT (corr 0.51 vs 0.66 at 768x432).
                H, W = P[0].shape[:2]
                rs = lambda x: cv2.resize(x, NEUFLOW_WH, interpolation=cv2.INTER_LINEAR)[:, :, ::-1].copy()
                a = torch.stack([torch.from_numpy(rs(x)).permute(2, 0, 1) for x in P]).to(self.dev).half()
                b = torch.stack([torch.from_numpy(rs(x)).permute(2, 0, 1) for x in C]).to(self.dev).half()
                if n < self.batch:       # fixed batch: repeat the last pair
                    a = torch.cat([a, a[-1:].repeat(self.batch - n, 1, 1, 1)]); b = torch.cat([b, b[-1:].repeat(self.batch - n, 1, 1, 1)])
                if self.shape != a.shape:
                    self.net.init_bhwd(a.shape[0], a.shape[2], a.shape[3], self.dev)
                    self.shape = a.shape
                f = self.net(a, b)[-1][:n].float().permute(0, 2, 3, 1).cpu().numpy()
                sx, sy = W / NEUFLOW_WH[0], H / NEUFLOW_WH[1]       # back to pixels of the processed frame
                out += [np.dstack([cv2.resize(g[..., 0], (W, H)) * sx, cv2.resize(g[..., 1], (W, H)) * sy]).astype(np.float32)
                        for g in f]
                continue
            out += list(f.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32))
        return out


def _fuse(conv, bn):
    """Conv2d + BatchNorm2d fusion, copied from NeuFlow_v2/infer.py."""
    fused = torch.nn.Conv2d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride,
                            padding=conv.padding, dilation=conv.dilation, groups=conv.groups,
                            bias=True).requires_grad_(False).to(conv.weight.device)
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fused.weight.copy_(torch.mm(w_bn, w_conv).view(fused.weight.shape))
    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fused


# ----------------------------------------------------------------------------------------------- boxes
def cmd_boxes(a):
    from ultralytics import YOLO
    from objfeat2 import motion_boxes, remove_covered
    det = YOLO(a.det)
    tim = {"detect": 0.0, "motion": 0.0, "frames": 0}
    for split in a.splits:
        for vid, paths in shard(list(frame_paths(a.frames, split).items()), a.shard):
            dst = a.out / split / f"{vid}.npz"
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            F, B, S, cache = [], [], [], {}
            for t in range(len(paths)):
                if not wanted(t, a.gap, split, a.train_every):
                    continue
                for k in (t - a.gap, t):
                    if k not in cache:
                        cache[k] = read(paths[k], a.width)
                prev, cur = (cv2.cvtColor(cache[k], cv2.COLOR_BGR2GRAY) for k in (t - a.gap, t))
                s = time.time()
                r = det.predict(cache[t], conf=a.conf, verbose=False, device=a.device)[0]
                b = r.boxes.xyxyn.cpu().numpy()
                b = b[remove_covered(b)]
                tim["detect"] += time.time() - s; s = time.time()
                mb = motion_boxes(prev, cur, b)
                tim["motion"] += time.time() - s; tim["frames"] += 1
                bx = np.concatenate([b, mb]) if len(mb) else b
                F += [t] * len(bx); B.append(bx.astype(np.float32)); S.append(np.r_[np.zeros(len(b)), np.ones(len(mb))])
                cache = {k: v for k, v in cache.items() if k >= t - a.gap}
            H, W = read(paths[0], a.width).shape[:2]
            save_npz(dst, f=np.asarray(F, np.int32), box=np.concatenate(B) if B else np.zeros((0, 4), np.float32),
                                src=np.concatenate(S).astype(np.uint8) if S else np.zeros(0, np.uint8),
                                n_frames=len(paths), startup=a.gap, frame_hw=np.array([H, W]),
                                train_every=a.train_every if split == "training" else 1)
            print(f"boxes {split}/{vid}: {len(F)} boxes", flush=True)
    write_timing(a.out / f"timing_{a.shard.replace('/', 'of')}.json", tim)


# ----------------------------------------------------------------------------------------------- flow velocity
def cmd_flowvel(a):
    from objfeat2 import velocity
    srcs = dict(s.split("=", 1) for s in a.src)
    flow = Flow(a.flow, a.device, a.neuflow_dir, a.batch)
    tim = {"decode": 0.0, "flow": 0.0, "vel": 0.0, "pairs": 0}
    for split in a.splits:
        for vid, paths in shard(list(frame_paths(a.frames, split).items()), a.shard):
            outs = {n: a.out / a.flow / n / split / f"{vid}.npz" for n in srcs}
            if all(p.exists() for p in outs.values()):
                continue
            boxes = {}
            for n, d in srcs.items():
                r = np.load(Path(d) / split / f"{vid}.npz")
                assert int(r["n_frames"]) == len(paths), (n, vid, int(r["n_frames"]), len(paths))
                assert int(r["startup"]) == a.gap, (n, vid, int(r["startup"]), a.gap)
                boxes[n] = (r["f"], r["box"])
            T = [t for t in range(len(paths)) if wanted(t, a.gap, split, a.train_every)
                 and any((f == t).any() for f, _ in boxes.values())]
            rec = {n: {"f": [], "box": [], "vel": []} for n in srcs}
            for i in range(0, len(T), a.chunk):
                ts = T[i:i + a.chunk]
                s = time.time()
                need = sorted({k for t in ts for k in (t - a.gap, t)})
                im = {k: read(paths[k], a.width) for k in need}
                tim["decode"] += time.time() - s; s = time.time()
                fl = flow([im[t - a.gap] for t in ts], [im[t] for t in ts])
                if a.device.startswith("cuda"):
                    torch.cuda.synchronize()
                tim["flow"] += time.time() - s; tim["pairs"] += len(ts); s = time.time()
                for t, F in zip(ts, fl):
                    for n, (f, b) in boxes.items():
                        k = np.flatnonzero(f == t)
                        if len(k):
                            rec[n]["f"] += [t] * len(k); rec[n]["box"].append(b[k])
                            rec[n]["vel"].append(np.stack([velocity(F, bb) for bb in b[k]]))
                tim["vel"] += time.time() - s
            H, W = im[need[-1]].shape[:2] if T else read(paths[0], a.width).shape[:2]
            for n, p in outs.items():
                p.parent.mkdir(parents=True, exist_ok=True)
                r = rec[n]
                save_npz(p, f=np.asarray(r["f"], np.int32),
                                    box=np.concatenate(r["box"]).astype(np.float32) if r["box"] else np.zeros((0, 4), np.float32),
                                    vel=np.concatenate(r["vel"]).astype(np.float32) if r["vel"] else np.zeros((0, 8), np.float32),
                                    n_frames=len(paths), startup=a.gap, frame_hw=np.array([H, W]),
                                    train_every=a.train_every if split == "training" else 1)
            print(f"flowvel {a.flow} {split}/{vid}: {len(T)} frames", flush=True)
    (a.out / a.flow).mkdir(parents=True, exist_ok=True)
    write_timing(a.out / a.flow / f"timing_{a.shard.replace('/', 'of')}.json", tim)


def cmd_bench(a):
    """Per-pair flow latency at batch 1 on real frames: 20 warm-up pairs, then 3 x 100 timed pairs."""
    flow = Flow(a.flow, a.device, a.neuflow_dir, batch=1)
    paths = next(iter(frame_paths(a.frames, "testing").values()))
    im = [read(p, a.width) for p in paths[:121 + a.gap]]
    sync = (lambda: torch.cuda.synchronize()) if a.device.startswith("cuda") else (lambda: None)
    for t in range(a.gap, 20 + a.gap):
        flow([im[t - a.gap]], [im[t]])
    reps = []
    for _ in range(3):
        ms = []
        for t in range(a.gap, 100 + a.gap):
            sync(); s = time.perf_counter(); flow([im[t - a.gap]], [im[t]]); sync()
            ms.append((time.perf_counter() - s) * 1e3)
        reps.append({"median_ms": float(np.median(ms)), "p95_ms": float(np.percentile(ms, 95))})
    res = {"flow": a.flow, "device": a.device, "frame_hw": list(im[0].shape[:2]), "reps": reps,
           "torch_threads": torch.get_num_threads(), "cv2_threads": cv2.getNumThreads()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


# ----------------------------------------------------------------------------------------------- tracks
def cmd_track(a):
    from ultralytics import YOLO
    tim = {"track": 0.0, "frames": 0}
    for split in a.splits:
        for vid, paths in shard(list(frame_paths(a.frames, split).items()), a.shard):
            dst = a.out / split / f"{vid}.npz"
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            det = YOLO(a.det)                       # fresh tracker state per video
            hist, F, B, V = {}, [], [], []
            for t, p in enumerate(paths):         # every frame is tracked; features are kept on wanted frames only
                im = read(p, a.width)
                H, W = im.shape[:2]
                s = time.time()
                kw = {} if a.conf is None else {"conf": a.conf}       # default: ultralytics' 0.1 (ByteTrack 2nd stage)
                r = det.track(im, persist=True, tracker=a.tracker, verbose=False, device=a.device, **kw)[0]
                tim["track"] += time.time() - s; tim["frames"] += 1
                if r.boxes.id is None:
                    continue
                xy = r.boxes.xyxy.cpu().numpy(); ids = r.boxes.id.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), i in zip(xy, ids):
                    hist.setdefault(i, []).append((t, (x1 + x2) / 2, (y1 + y2) / 2, max(y2 - y1, 1.0)))
                if not wanted(t, a.gap, split, a.train_every):
                    continue
                for (x1, y1, x2, y2), i in zip(xy, ids):
                    past = [e for e in hist[i] if t - a.window <= e[0] < t]
                    if not past:
                        continue                   # new track: no velocity yet
                    t0, cx0, cy0, h0 = past[0]
                    _, cx, cy, h = hist[i][-1]
                    dt = t - t0; hm = (h + h0) / 2
                    vx, vy = (cx - cx0) / dt / hm, (cy - cy0) / dt / hm
                    F.append(t); B.append([x1 / W, y1 / H, x2 / W, y2 / H])
                    V.append([vx, vy, np.hypot(vx, vy), np.log(h / h0) / dt])
            save_npz(dst, f=np.asarray(F, np.int32), box=np.asarray(B, np.float32).reshape(-1, 4),
                                vel=np.asarray(V, np.float32).reshape(-1, 4), n_frames=len(paths), startup=a.gap,
                                frame_hw=np.array([H, W]), train_every=a.train_every if split == "training" else 1)
            print(f"track {split}/{vid}: {len(F)} rows", flush=True)
    write_timing(a.out / f"timing_{a.shard.replace('/', 'of')}.json", tim)


# ----------------------------------------------------------------------------------------------- scoring
def construct(r: dict, cons: str) -> np.ndarray:
    v = r["vel"].astype(np.float64)
    if cons in ("c8", "trk"):
        return v
    H = float(r["frame_hw"][0])
    h = np.maximum((r["box"][:, 3] - r["box"][:, 1]).astype(np.float64) * H, 1.0)[:, None]
    return {"c1": v.sum(1, keepdims=True), "c8h": v / h, "c1h": v.sum(1, keepdims=True) / h}[cons]


def cmd_score(a):
    from aean.niw import NIWMixture, select_mixture
    from fuse_runs import normaliser
    from objfuse import evaluate
    dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    load = lambda d: {p.stem: dict(np.load(p)) for p in sorted(d.glob("*.npz"))}
    train, test = load(a.feat / "training"), load(a.feat / "testing")
    gt = json.loads(a.gt.read_text())
    assert sorted(test) == sorted(gt), "test videos differ from gt.json"
    for v in gt:
        assert int(test[v]["n_frames"]) == len(gt[v]), v

    def rows(r, sub):
        f, X = r["f"], construct(r, a.cons)
        k = (f >= int(r["startup"])) if a.drop_startup else np.ones(len(f), bool)
        if sub:
            k &= (f % a.train_every == 0)
        return f[k], X[k]

    def frame_scores(m, r, sub=False):
        f, X = rows(r, sub)
        s = np.full(int(r["n_frames"]), -np.inf)
        if len(X):
            np.maximum.at(s, f, (-m.logpdf(torch.from_numpy(X))).cpu().numpy())
        return s

    for v, r in train.items():   # features written with a training subsample must be scored with a multiple of it
        assert a.train_every % int(r.get("train_every", 1)) == 0, (v, int(r.get("train_every", 1)), a.train_every)

    def fit_group(names, vids):
        """tools/objscore2.py on one group of training videos: 2-fold split by video, held-out likelihood selection,
        cross-fitted reference, floor for frames without objects. -> test scores for vids, reference, hyper."""
        folds = (names[0::2], names[1::2])
        rng = np.random.default_rng(0)
        X = {v: torch.from_numpy(rows(train[v], True)[1]) for v in names}
        fit_sets, eval_sets = [], []
        for fit_n, ev_n in (folds, folds[::-1]):
            Xe = torch.cat([X[v] for v in ev_n])
            fit_sets.append(torch.cat([X[v] for v in fit_n]))
            eval_sets.append(Xe[rng.permutation(len(Xe))[:20000]])
        (K, k0, e), _ = select_mixture(fit_sets, eval_sets, device=dev)
        C = fit_sets[0].shape[1]
        # reference = training frames of the subsample (and t >= startup with --drop-startup); with the defaults
        # (train_every 1, no drop) this is every training frame, exactly as tools/objscore2.py
        ref, keep = [], []
        for (fit_n, ev_n), Xf in zip((folds, folds[::-1]), fit_sets):
            m = NIWMixture(K, k0, C + 1 + e, device=dev).fit(Xf)
            for v in ev_n:
                ref.append(frame_scores(m, train[v], True))
                t = np.arange(int(train[v]["n_frames"]))
                keep.append((t % a.train_every == 0) & (t >= int(train[v]["startup"]) if a.drop_startup else True))
        ref = np.concatenate(ref)[np.concatenate(keep)]
        floor = ref[np.isfinite(ref)].min()
        ref = np.where(np.isfinite(ref), ref, floor)
        full = NIWMixture(K, k0, C + 1 + e, device=dev).fit(torch.cat(list(X.values())))
        S = {}
        for v in vids:
            s = frame_scores(full, test[v])
            S[v] = np.where(np.isfinite(s), s, floor)
        return S, ref, {"K": K, "kappa0": k0, "nu0": C + 1 + e, "dim": C, "train_rows": int(sum(len(x) for x in X.values()))}

    if a.scenes:   # ShanghaiTech as archived (runs_20261104/sht_retry.sh + tools/objfuse_sht.py): one model, reference
        scene = lambda v: v.split("_")[0]          # and Fisher calibration per scene, AUC pooled over all test videos
        grp = {g: ([n for n in sorted(train) if scene(n) == g], [v for v in sorted(gt) if scene(v) == g])
               for g in sorted({scene(v) for v in gt})}
    else:
        grp = {"all": (sorted(train), sorted(gt))}
    z, S, refs, hyper = {}, {}, {}, {}
    for g, (names, vids) in grp.items():
        Sg, ref, hyper[g] = fit_group(names, vids)
        nz = normaliser(ref, "fisher")
        z.update({v: nz(Sg[v]) for v in vids}); S.update(Sg); refs[g] = ref
    res = evaluate({v: z[v] for v in sorted(gt)}, gt, a.warmup)
    out = {"feat": str(a.feat), "cons": a.cons, "train_every": a.train_every, "warmup": a.warmup,
           "drop_startup": a.drop_startup, "scenes": a.scenes, "train_rows": sum(h["train_rows"] for h in hyper.values()),
           "hyper": hyper, **res}
    print(json.dumps(out), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    if a.save_scores:
        np.savez_compressed(a.out.with_suffix(".npz"), **{f"s/{v}": S[v] for v in gt}, **{f"ref/{g}": r for g, r in refs.items()})


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def common(p, out=True):
        p.add_argument("--frames", type=Path, required=True)
        if out:
            p.add_argument("--out", type=Path, required=True)
        p.add_argument("--gap", type=int, default=1)
        p.add_argument("--width", type=int, default=640)
        p.add_argument("--train-every", type=int, default=1)
        p.add_argument("--splits", nargs="+", default=["training", "testing"])
        p.add_argument("--shard", default="0/1")
        p.add_argument("--device", default=dev)

    p = sp.add_parser("boxes"); common(p)
    p.add_argument("--det", default="yolo11x.pt"); p.add_argument("--conf", type=float, default=0.25)
    p = sp.add_parser("flowvel"); common(p)
    p.add_argument("--flow", choices=["dis", "raft", "neuflow"], required=True)
    p.add_argument("--src", nargs="+", required=True, help="name=dir with <split>/<video>.npz holding f and box")
    p.add_argument("--neuflow-dir", type=Path); p.add_argument("--batch", type=int, default=8)
    p.add_argument("--chunk", type=int, default=64)
    p = sp.add_parser("bench"); common(p)
    p.add_argument("--flow", choices=["dis", "raft", "neuflow"], required=True); p.add_argument("--neuflow-dir", type=Path)
    p = sp.add_parser("track"); common(p)
    p.add_argument("--det", default="yolo11n.pt"); p.add_argument("--conf", type=float, default=None)
    p.add_argument("--tracker", default="bytetrack.yaml"); p.add_argument("--window", type=int, default=4)
    p = sp.add_parser("score")
    p.add_argument("--feat", type=Path, required=True); p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cons", choices=["c8", "c1", "c8h", "c1h", "trk"], default="c8")
    p.add_argument("--train-every", type=int, default=1); p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--drop-startup", action="store_true", help="drop rows with f < startup (released feat2 rows)")
    p.add_argument("--save-scores", action="store_true"); p.add_argument("--cpu", action="store_true")
    p.add_argument("--scenes", action="store_true", help="ShanghaiTech: per-scene model and calibration (prefix XX_)")
    a = ap.parse_args()
    if os.environ.get("VP_CV2_THREADS"):
        cv2.setNumThreads(int(os.environ["VP_CV2_THREADS"])); torch.set_num_threads(int(os.environ["VP_CV2_THREADS"]))
        assert cv2.getNumThreads() == int(os.environ["VP_CV2_THREADS"]), cv2.getNumThreads()
    {"boxes": cmd_boxes, "flowvel": cmd_flowvel, "bench": cmd_bench, "track": cmd_track, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
