"""Write the original RGB frames in the layout most VAD repositories expect, with video names matching our gt.json:

    <out>/<ds>/training/frames/<video>/000.jpg ...
    <out>/<ds>/testing/frames/<video>/000.jpg ...

    python hazirla_frames.py --ds ped2 --src /workspace/data/UCSD_Anomaly_Dataset.v1p2 --out /workspace/frames
    python hazirla_frames.py --ds avenue --src /workspace/data/avenue/Avenue_Dataset.zip --out /workspace/frames
    python hazirla_frames.py --ds shanghaitech --src /workspace/data/shanghaitech --test-src /workspace/data/shtech_test --out /workspace/frames

Video names: UCSD TestXXX / TrainXXX (as in prepare_ucsd.py), Avenue 01..21 (as in prepare_avenue.py),
ShanghaiTech <scene>_<clip> (as in the official test_frame_mask). Frame counts are checked against gt.json when given.
"""
import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import cv2


def write_frames(frames, d: Path):
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    for n, f in enumerate(frames, 1):
        cv2.imwrite(str(d / f"{n - 1:03d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return n


def video_frames(path):
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield f


def ucsd(ds, src: Path, out: Path):
    ped = {"ped1": "UCSDped1", "ped2": "UCSDped2"}[ds]
    counts = {}
    for split, sub in (("training", "Train"), ("testing", "Test")):
        for vd in sorted(p for p in (src / ped / sub).iterdir() if p.is_dir() and not p.name.endswith("_gt")):
            frames, last = [], None
            for t in sorted(vd.glob("*.tif")):
                im = cv2.imread(str(t), cv2.IMREAD_COLOR)
                if im is None:          # one unreadable frame in Ped1 Test017: repeat the previous one (as prepare_ucsd.py)
                    im = last
                frames.append(im)
                last = im
            counts[(split, vd.name)] = write_frames(frames, out / ds / split / "frames" / vd.name)
    return counts


def avenue(src: Path, out: Path):
    counts = {}
    z = zipfile.ZipFile(src)
    with tempfile.TemporaryDirectory() as tmp:
        for name in z.namelist():
            if not name.endswith(".avi"):
                continue
            split = "training" if "training_videos" in name else "testing" if "testing_videos" in name else None
            if split is None:
                continue
            p = Path(z.extract(name, tmp))
            counts[(split, p.stem)] = write_frames(video_frames(p), out / "avenue" / split / "frames" / p.stem)
    return counts


def shanghaitech(src: Path, test_src: Path, out: Path):
    counts = {}
    for avi in sorted(src.rglob("*.avi")):
        counts[("training", avi.stem)] = write_frames(video_frames(avi), out / "shanghaitech" / "training" / "frames" / avi.stem)
    frames_root = next(p for p in test_src.rglob("frames") if p.is_dir())
    for vd in sorted(p for p in frames_root.iterdir() if p.is_dir()):
        imgs = sorted(vd.glob("*.jpg"), key=lambda p: int(p.stem))
        counts[("testing", vd.name)] = write_frames((cv2.imread(str(i)) for i in imgs),
                                                    out / "shanghaitech" / "testing" / "frames" / vd.name)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["ped1", "ped2", "avenue", "shanghaitech"])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--test-src", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gt", type=Path, help="gt.json to check test frame counts")
    a = ap.parse_args()
    if a.ds in ("ped1", "ped2"):
        c = ucsd(a.ds, a.src, a.out)
    elif a.ds == "avenue":
        c = avenue(a.src, a.out)
    else:
        c = shanghaitech(a.src, a.test_src, a.out)
    print(f"{a.ds}: {sum(1 for k in c if k[0] == 'training')} train / {sum(1 for k in c if k[0] == 'testing')} test videos")
    if a.gt:
        gt = json.loads(a.gt.read_text())
        bad = [(v, n, len(gt[v])) for (s, v), n in c.items() if s == "testing" and v in gt and n != len(gt[v])]
        missing = [v for v in gt if ("testing", v) not in c]
        if bad or missing:
            raise SystemExit(f"frame count mismatch {bad[:5]} / missing {missing[:5]}")
        print("test frame counts match gt.json")


if __name__ == "__main__":
    main()
