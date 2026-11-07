"""ShanghaiTech frame labels -> one gt.json {video: [0/1 per frame]} from the official testing/test_frame_mask/*.npy.

    python gt_sht.py --root /workspace/data --out /workspace/cache/shanghaitech/gt.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
dirs = [p for p in a.root.rglob("test_frame_mask") if p.is_dir()]
assert len(dirs) >= 1, "no test_frame_mask folder found"
gt = {}
for d in dirs:
    for f in sorted(d.glob("*.npy")):
        y = np.load(f, allow_pickle=False).astype(int).ravel().tolist()
        assert f.stem not in gt or gt[f.stem] == y, f
        gt[f.stem] = y
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(gt))
print(f"{len(gt)} test videos, {sum(map(len, gt.values()))} frames, {sum(map(sum, gt.values()))} anomalous, from {dirs}")
