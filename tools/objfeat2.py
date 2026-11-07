"""Light object attributes for embedded use: detector + motion-contour boxes, velocity, pose and a small embedding.

    python tools/objfeat2.py --frames /workspace/frames/avenue --out /workspace/feat2/avenue --gap 2

One fixed configuration for every dataset (no per-dataset thresholds). Only frames t-gap..t are used (causal).
Boxes:
  * detector boxes (YOLO, COCO classes, conf >= --conf), boxes covered by a larger one (> 60% of their area) removed;
  * motion-contour boxes: |blur(I_t) - blur(I_{t-gap})| > 18 grey levels, external contours outside the detector
    boxes with area >= 0.1% of the frame and aspect ratio < 10 (after the foreground boxes of Reiss & Hoshen, with one
    fixed setting instead of per-dataset ones). They cover moving things that the detector does not know.
Attributes per box:
  vel  (8): mean L1 flow magnitude per orientation bin (DIS optical flow, pixels of the processed frame);
  pose (34): COCO-17 keypoints of the matched YOLO-pose person, relative to the box centre and divided by box height;
             zeros and pose_ok = 0 for boxes without a matched person;
  app  (D): L2-normalised image embedding of the crop (MobileCLIP2-S0 by default).
Output: <out>/<split>/<video>.npz with f, box (normalised xyxy), src (0 detector, 1 motion), cls, conf, vel, pose,
pose_ok, app, n_frames, plus timing.json with per-stage seconds.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def iou_cover(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fraction of each box in a covered by each box in b, (len(a), len(b))."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    return inter / np.maximum(area[:, None], 1e-9)


def remove_covered(b: np.ndarray, thr: float = 0.6) -> np.ndarray:
    if len(b) < 2:
        return np.arange(len(b))
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    order = area.argsort()
    keep = []
    for i, k in enumerate(order):
        larger = order[i + 1:]
        if not len(larger) or iou_cover(b[k:k + 1], b[larger]).max() <= thr:
            keep.append(k)
    return np.array(sorted(keep), int)


def motion_boxes(prev: np.ndarray, cur: np.ndarray, det: np.ndarray, min_frac: float = 1e-3) -> np.ndarray:
    H, W = cur.shape
    d = cv2.absdiff(cv2.GaussianBlur(prev, (5, 5), 0), cv2.GaussianBlur(cur, (5, 5), 0))
    m = (d > 18).astype(np.uint8) * 255
    for x1, y1, x2, y2 in (det * [W, H, W, H]).astype(int):
        m[max(y1 - 2, 0):y2 + 3, max(x1 - 2, 0):x2 + 3] = 0
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        x, y, w, h = cv2.boundingRect(c)
        if (w + 1) * (h + 1) >= min_frac * H * W and w / max(h, 1) < 10 and h / max(w, 1) < 10:
            out.append([max(x - 2, 0) / W, max(y - 2, 0) / H, min(x + w + 2, W) / W, min(y + h + 2, H) / H])
    return np.asarray(out, np.float32).reshape(-1, 4)


def velocity(flow: np.ndarray, b: np.ndarray) -> np.ndarray:
    H, W = flow.shape[:2]
    x1, y1, x2, y2 = (b * [W, H, W, H]).round().astype(int)
    f = flow[max(y1, 0):max(y2, y1 + 1), max(x1, 0):max(x2, x1 + 1)].reshape(-1, 2)
    if not len(f):
        return np.zeros(8, np.float32)
    mag = np.abs(f).sum(1)
    ang = (np.arctan2(f[:, 1], f[:, 0]) % (2 * np.pi)) / (2 * np.pi) * 8
    return (np.bincount(np.minimum(ang.astype(int), 7), weights=mag, minlength=8) / len(f)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gap", type=int, default=1)
    ap.add_argument("--det", default="yolo11n.pt")
    ap.add_argument("--pose", default="yolo11n-pose.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--emb", default="MobileCLIP2-S0")
    ap.add_argument("--emb-pretrained", default="dfndr2b")
    ap.add_argument("--width", type=int, default=640, help="frames are processed at this width (smaller kept)")
    ap.add_argument("--splits", nargs="+", default=["training", "testing"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    from ultralytics import YOLO
    import open_clip
    dev = a.device
    det, pose_m = YOLO(a.det), YOLO(a.pose)
    emb, _, prep = open_clip.create_model_and_transforms(a.emb, pretrained=a.emb_pretrained, device=dev)
    emb.eval()
    norm = prep.transforms[-1]
    mean = torch.tensor(norm.mean, device=dev)[:, None, None]
    std = torch.tensor(norm.std, device=dev)[:, None, None]
    S = emb.visual.image_size if hasattr(emb.visual, "image_size") else 224
    S = S[0] if isinstance(S, (tuple, list)) else int(S)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    si, sn = map(int, a.shard.split("/"))
    timing = {"decode": 0.0, "detect": 0.0, "pose": 0.0, "motion": 0.0, "flow": 0.0, "embed": 0.0, "frames": 0}

    for split in a.splits:
        vids = sorted(p for p in (a.frames / split / "frames").iterdir() if p.is_dir())[si::sn]
        (a.out / split).mkdir(parents=True, exist_ok=True)
        for vd in vids:
            dst = a.out / split / f"{vd.name}.npz"
            if dst.exists():
                continue
            t0 = time.time()
            imgs = []
            for p in sorted(vd.glob("*.jpg"), key=lambda p: int(p.stem)):   # time order (1000.jpg after 999.jpg)
                im = cv2.imread(str(p))
                if im.shape[1] > a.width:
                    im = cv2.resize(im, (a.width, int(round(im.shape[0] * a.width / im.shape[1]))), interpolation=cv2.INTER_AREA)
                imgs.append(im)
            grey = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in imgs]
            H, W = grey[0].shape
            timing["decode"] += time.time() - t0
            rec = {k: [] for k in ("f", "box", "src", "cls", "conf", "vel", "pose", "pose_ok", "app")}
            for t in range(len(imgs)):
                tp = t - a.gap if t >= a.gap else min(a.gap, len(imgs) - 1)   # startup: first available pair
                prev, cur = (grey[tp], grey[t]) if t >= a.gap else (grey[t], grey[tp])
                s = time.time()
                r = det.predict(imgs[t], conf=a.conf, verbose=False, device=dev)[0]
                b, c, cf = r.boxes.xyxyn.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()
                timing["detect"] += time.time() - s; s = time.time()
                pr = pose_m.predict(imgs[t], conf=a.conf, verbose=False, device=dev)[0]
                pb = pr.boxes.xyxyn.cpu().numpy()
                kp = pr.keypoints.xyn.cpu().numpy() if pr.keypoints is not None and len(pb) else np.zeros((0, 17, 2))
                timing["pose"] += time.time() - s
                P = np.zeros((len(b), 34), np.float32); P_ok = np.zeros(len(b), np.uint8)
                if len(pb) and len(b):
                    cov = np.minimum(iou_cover(b, pb), iou_cover(pb, b).T)          # mutual coverage
                    for i in np.flatnonzero(c == 0):
                        j = int(cov[i].argmax())
                        if cov[i, j] > 0.5:
                            cx, cy = (b[i, 0] + b[i, 2]) / 2, (b[i, 1] + b[i, 3]) / 2
                            P[i] = ((kp[j] - [cx, cy]) / max(b[i, 3] - b[i, 1], 1e-3)).reshape(-1); P_ok[i] = 1
                k = remove_covered(b); b, c, cf, P, P_ok = b[k], c[k], cf[k], P[k], P_ok[k]
                s = time.time()
                mb = motion_boxes(prev, cur, b)
                timing["motion"] += time.time() - s
                boxes = np.concatenate([b, mb]) if len(mb) else b
                if not len(boxes):
                    continue
                s = time.time()
                flow = dis.calc(prev, cur, None)
                timing["flow"] += time.time() - s
                P = np.concatenate([P, np.zeros((len(mb), 34), np.float32)])
                ok = np.r_[P_ok, np.zeros(len(mb), np.uint8)]
                s = time.time()
                rgb = torch.from_numpy(imgs[t][:, :, ::-1].copy()).to(dev).permute(2, 0, 1).float() / 255
                crops = []
                for bb in boxes:
                    x1, y1, x2, y2 = (bb * [W, H, W, H]).round().astype(int)
                    x1, y1 = max(x1, 0), max(y1, 0); x2, y2 = min(max(x2, x1 + 2), W), min(max(y2, y1 + 2), H)
                    crop = torch.nn.functional.interpolate(rgb[None, :, y1:y2, x1:x2], size=(S, S), mode="bilinear",
                                                           align_corners=False)[0]
                    crops.append((crop - mean) / std)
                with torch.no_grad():
                    e = emb.encode_image(torch.stack(crops)).float()
                timing["embed"] += time.time() - s
                rec["app"].append(torch.nn.functional.normalize(e, dim=1).cpu().numpy().astype(np.float16))
                rec["vel"].append(np.stack([velocity(flow, bb) for bb in boxes]))
                rec["f"] += [t] * len(boxes)
                rec["box"].append(boxes.astype(np.float32))
                rec["src"].append(np.r_[np.zeros(len(b)), np.ones(len(mb))].astype(np.uint8))
                rec["cls"].append(np.r_[c, -np.ones(len(mb))].astype(np.int16))
                rec["conf"].append(np.r_[cf, np.zeros(len(mb))].astype(np.float32))
                rec["pose"].append(P); rec["pose_ok"].append(ok)
            timing["frames"] += len(imgs)
            cat = lambda k, shape, dt: np.concatenate(rec[k]).astype(dt) if rec[k] else np.zeros(shape, dt)
            np.savez_compressed(dst, f=np.asarray(rec["f"], np.int32), box=cat("box", (0, 4), np.float32),
                                src=cat("src", 0, np.uint8), cls=cat("cls", 0, np.int16), conf=cat("conf", 0, np.float32),
                                vel=cat("vel", (0, 8), np.float32), pose=cat("pose", (0, 34), np.float32),
                                pose_ok=cat("pose_ok", 0, np.uint8), app=cat("app", (0, 512), np.float16),
                                n_frames=len(imgs), startup=a.gap, frame_hw=np.array([H, W]))
            print(f"{split}/{vd.name}: {len(imgs)} frames, {len(rec['f'])} objects, {time.time() - t0:.1f}s", flush=True)
    (a.out / f"timing_{si}.json").write_text(json.dumps(timing, indent=1))


if __name__ == "__main__":
    main()
