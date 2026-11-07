"""Box-crop image embeddings with open_clip, the app attribute of tools/objfeat2.py."""
from __future__ import annotations

import numpy as np
import torch

from vadkit.core.interfaces import AppearanceEncoder
from vadkit.core.registry import register
from vadkit.perception.detector import default_device


@register("encoder", "open_clip")
class OpenClipEncoder(AppearanceEncoder):
    """Crop (clamped, >= 2 px) -> SxS bilinear -> model normalisation -> encode_image -> L2 -> fp16."""

    def __init__(self, model: str = "MobileCLIP2-S0", pretrained: str = "dfndr2b", device: str | None = None):
        import open_clip
        self.dev = default_device(device)
        self.model, _, prep = open_clip.create_model_and_transforms(model, pretrained=pretrained, device=self.dev)
        self.model.eval()
        norm = prep.transforms[-1]
        self.mean = torch.tensor(norm.mean, device=self.dev)[:, None, None]
        self.std = torch.tensor(norm.std, device=self.dev)[:, None, None]
        S = self.model.visual.image_size if hasattr(self.model.visual, "image_size") else 224
        self.size = S[0] if isinstance(S, (tuple, list)) else int(S)
        with torch.no_grad():
            self.dim = int(self.model.encode_image(torch.zeros(1, 3, self.size, self.size, device=self.dev)).shape[1])

    def encode(self, bgr: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """(N, D) float16 L2-normalised embeddings of the normalised xyxy box crops of one frame."""
        if not len(boxes):
            return np.zeros((0, self.dim), np.float16)
        H, W = bgr.shape[:2]
        rgb = torch.from_numpy(bgr[:, :, ::-1].copy()).to(self.dev).permute(2, 0, 1).float() / 255
        crops = []
        for bb in boxes:
            x1, y1, x2, y2 = (bb * [W, H, W, H]).round().astype(int)
            x1, y1 = max(x1, 0), max(y1, 0); x2, y2 = min(max(x2, x1 + 2), W), min(max(y2, y1 + 2), H)
            crop = torch.nn.functional.interpolate(rgb[None, :, y1:y2, x1:x2], size=(self.size, self.size),
                                                   mode="bilinear", align_corners=False)[0]
            crops.append((crop - self.mean) / self.std)
        with torch.no_grad():
            e = self.model.encode_image(torch.stack(crops)).float()
        return torch.nn.functional.normalize(e, dim=1).cpu().numpy().astype(np.float16)


@register("encoder", "timm")
class TimmEncoder(OpenClipEncoder):
    """Same crop rule as OpenClipEncoder with a timm image backbone (e.g. DINOv2 ViT-S/14), pooled features."""

    def __init__(self, model: str = "vit_small_patch14_dinov2.lvd142m", size: int = 224, device: str | None = None):
        import timm
        self.dev = default_device(device)
        kw = {"img_size": size} if "dinov2" in model else {}
        net = timm.create_model(model, pretrained=True, num_classes=0, **kw).to(self.dev).eval()
        cfg = timm.data.resolve_data_config({}, model=net)
        self.mean = torch.tensor(cfg["mean"], device=self.dev)[:, None, None]
        self.std = torch.tensor(cfg["std"], device=self.dev)[:, None, None]
        self.size = size
        self.model = type("Wrap", (), {"encode_image": staticmethod(lambda x: net(x))})()
        with torch.no_grad():
            self.dim = int(net(torch.zeros(1, 3, size, size, device=self.dev)).shape[1])
