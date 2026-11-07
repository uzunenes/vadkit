"""Frame input in TIME order: dataset frame folders (<root>/<split>/frames/<video>/<t>.jpg), video files, cameras."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def time_order(paths: list[Path]) -> list[Path]:
    """Sort frame files by int(stem) and require stems == 0..n-1 (text order puts 1000.jpg after 100.jpg)."""
    out = sorted(paths, key=lambda p: int(p.stem))
    if [int(p.stem) for p in out] != list(range(len(out))):
        raise ValueError(f"frame stems of {out[0].parent if out else '?'} are not 0..{len(out) - 1} (gap or duplicate)")
    return out


def frame_paths(frames_root: Path | str, split: str) -> dict[str, list[Path]]:
    """{video: [jpg paths in time order]} for <frames_root>/<split>/frames/<video>/, videos sorted by name."""
    root = Path(frames_root) / split / "frames"
    return {vd.name: time_order(list(vd.glob("*.jpg"))) for vd in sorted(p for p in root.iterdir() if p.is_dir())}


def resize_width(im: np.ndarray, width: int) -> np.ndarray:
    """Shrink to `width` (INTER_AREA, height rounded) when wider; smaller frames are kept (tools/objfeat2.py)."""
    if im.shape[1] > width:
        im = cv2.resize(im, (width, int(round(im.shape[0] * width / im.shape[1]))), interpolation=cv2.INTER_AREA)
    return im


def read(path: Path | str, width: int = 640) -> np.ndarray:
    """BGR frame at the processing width, identical to tools/velpilot.read."""
    im = cv2.imread(str(path))
    if im is None:
        raise FileNotFoundError(path)
    return resize_width(im, width)


class VideoSource:
    """Iterator of (t, bgr) from a frame folder (time order), a video file or a camera index, at the processing width."""

    def __init__(self, src: Path | str | int, width: int = 640):
        self.src, self.width = src, width

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        src = self.src
        if isinstance(src, (str, Path)) and Path(src).is_dir():
            for t, p in enumerate(time_order([p for p in Path(src).iterdir() if p.suffix.lower() in IMAGE_EXT])):
                yield t, read(p, self.width)
            return
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        cap = cv2.VideoCapture(src if isinstance(src, int) else str(src))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video source {self.src!r}")
        try:
            t = 0
            while True:
                ok, im = cap.read()
                if not ok:
                    return
                yield t, resize_width(im, self.width)
                t += 1
        finally:
            cap.release()
