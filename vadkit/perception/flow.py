"""Optical flow back-ends, ported from tools/velpilot.py Flow: DIS (CPU), RAFT-large and NeuFlow v2 (GPU)."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from vadkit.core.interfaces import FlowEstimator
from vadkit.core.registry import register
from vadkit.perception.detector import default_device

NEUFLOW_WH = (768, 432)


@register("flow", "dis")
class DISFlow(FlowEstimator):
    """OpenCV DIS, preset medium, on grey frames (tools/objfeat2.py)."""

    def __init__(self):
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    def flow(self, prev_bgr: list[np.ndarray], cur_bgr: list[np.ndarray]) -> list[np.ndarray]:
        """DIS flow prev -> cur for each pair."""
        return [self.dis.calc(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY), cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), None)
                for p, c in zip(prev_bgr, cur_bgr)]


def _pad(x: torch.Tensor, mult: int) -> tuple[torch.Tensor, int, int]:
    """Replicate-pad H, W up to a multiple of mult; returns the padded tensor and the original H, W."""
    H, W = x.shape[-2:]
    ph, pw = (-H) % mult, (-W) % mult
    return torch.nn.functional.pad(x, (0, pw, 0, ph), mode="replicate"), H, W


def _rgb_batch(frames: list[np.ndarray]) -> torch.Tensor:
    """(B, 3, H, W) uint8 RGB tensor of BGR frames."""
    return torch.stack([torch.from_numpy(x[:, :, ::-1].copy()).permute(2, 0, 1) for x in frames])


@register("flow", "raft")
class RAFTFlow(FlowEstimator):
    """torchvision raft_large, weights C_T_SKHT_V2, RGB, padded to a multiple of 8, batched."""

    def __init__(self, device: str | None = None, batch: int = 8):
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        w = Raft_Large_Weights.C_T_SKHT_V2      # Chairs+Things+Sintel+KITTI+HD1K, like neuflow_mixed
        self.dev, self.batch = default_device(device), batch
        self.net = raft_large(weights=w).to(self.dev).eval()
        self.tf = w.transforms()

    @torch.no_grad()
    def flow(self, prev_bgr: list[np.ndarray], cur_bgr: list[np.ndarray]) -> list[np.ndarray]:
        """RAFT final-iteration flow prev -> cur for each pair."""
        out = []
        for i in range(0, len(prev_bgr), self.batch):
            a = _rgb_batch(prev_bgr[i:i + self.batch]).to(self.dev)
            b = _rgb_batch(cur_bgr[i:i + self.batch]).to(self.dev)
            a, b = self.tf(a, b)
            a, H, W = _pad(a, 8); b, _, _ = _pad(b, 8)
            f = self.net(a, b)[-1][:, :, :H, :W]
            out += list(f.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32))
        return out


def _fuse(conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d) -> torch.nn.Conv2d:
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


@register("flow", "neuflow")
class NeuFlowFlow(FlowEstimator):
    """Official NeuFlow v2 (neuflow_mixed.pth in neuflow_dir), conv-bn fused, fp16, RGB at 768x432; needs CUDA."""

    def __init__(self, neuflow_dir: Path | str, device: str = "cuda", batch: int = 8):
        if not str(device).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError(f"NeuFlowFlow runs in fp16 and needs CUDA (device={device!r})")
        neuflow_dir = Path(neuflow_dir)
        sys.path.insert(0, str(neuflow_dir))
        from NeuFlow.backbone_v7 import ConvBlock
        from NeuFlow.neuflow import NeuFlow
        self.dev, self.batch = device, batch
        net = NeuFlow().to(device)
        ck = torch.load(neuflow_dir / "neuflow_mixed.pth", map_location=device, weights_only=True)
        net.load_state_dict(ck["model"], strict=True)
        for m in net.modules():          # conv+bn fusion exactly as the official infer.py
            if type(m) is ConvBlock:
                m.conv1 = _fuse(m.conv1, m.norm1); m.conv2 = _fuse(m.conv2, m.norm2)
                delattr(m, "norm1"); delattr(m, "norm2"); m.forward = m.forward_fuse
        self.net = net.eval().half()
        self.shape: torch.Size | None = None

    @torch.no_grad()
    def flow(self, prev_bgr: list[np.ndarray], cur_bgr: list[np.ndarray]) -> list[np.ndarray]:
        """NeuFlow flow prev -> cur, resized back to the frame and rescaled to its pixels."""
        out = []
        for i in range(0, len(prev_bgr), self.batch):
            P, C = prev_bgr[i:i + self.batch], cur_bgr[i:i + self.batch]
            n = len(P)
            # RGB in [0, 255] (its training loader uses PIL; the infer.py demo passes BGR), fp16, at its official
            # inference size 768x432; fixed (B, H, W) after init_bhwd
            H, W = P[0].shape[:2]
            rs = lambda x: cv2.resize(x, NEUFLOW_WH, interpolation=cv2.INTER_LINEAR)[:, :, ::-1].copy()
            a = torch.stack([torch.from_numpy(rs(x)).permute(2, 0, 1) for x in P]).to(self.dev).half()
            b = torch.stack([torch.from_numpy(rs(x)).permute(2, 0, 1) for x in C]).to(self.dev).half()
            if n < self.batch:       # fixed batch: repeat the last pair
                a = torch.cat([a, a[-1:].repeat(self.batch - n, 1, 1, 1)])
                b = torch.cat([b, b[-1:].repeat(self.batch - n, 1, 1, 1)])
            if self.shape != a.shape:
                self.net.init_bhwd(a.shape[0], a.shape[2], a.shape[3], self.dev)
                self.shape = a.shape
            f = self.net(a, b)[-1][:n].float().permute(0, 2, 3, 1).cpu().numpy()
            sx, sy = W / NEUFLOW_WH[0], H / NEUFLOW_WH[1]       # back to pixels of the processed frame
            out += [np.dstack([cv2.resize(g[..., 0], (W, H)) * sx, cv2.resize(g[..., 1], (W, H)) * sy]).astype(np.float32)
                    for g in f]
        return out
