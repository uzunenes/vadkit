"""Params / GFLOPs / weight sizes / peak GPU memory of VideoPatchCore and AED-MAE components, with the authors' weights."""
import json, os, sys
import torch
from torch.utils.flop_counter import FlopCounterMode

def gfl(fn):
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            fn()
    return fc.get_total_flops() / 1e9

which = sys.argv[1]; out = {}
if which == "vpc":
    os.chdir("/workspace/m3/vpc/weights")
    yolo = torch.hub.load("/workspace/m3/vpc/yolov5", "yolov5l", source="local", pretrained=True).eval().cuda()
    core = yolo.model  # DetectMultiBackend inside AutoShape
    x = torch.zeros(1, 3, 640, 640, device="cuda")
    out["yolov5l"] = {"params_M": sum(p.numel() for p in yolo.parameters()) / 1e6,
                      "gflops_640": gfl(lambda: core(x)), "file_MB": os.path.getsize("/workspace/m3/vpc/weights/yolov5l.pt") / 1e6}
    import clip
    model, _ = clip.load("RN101", device="cuda")
    v = model.visual
    xi = torch.zeros(1, 3, 224, 224, device="cuda", dtype=next(v.parameters()).dtype)
    out["clip_rn101_visual"] = {"params_M": sum(p.numel() for p in v.parameters()) / 1e6,
                                "params_total_incl_text_M": sum(p.numel() for p in model.parameters()) / 1e6,
                                "gflops_224_per_crop": gfl(lambda: v(xi)),
                                "file_MB_full_clip": os.path.getsize(os.path.expanduser("~/.cache/clip/RN101.pt")) / 1e6}
    banks = {}
    for root, _, files in os.walk("/workspace/m3/vpc/wd/0"):
        for f in files:
            if "memory_bank" in f:
                banks[os.path.join(root, f).split("/wd/0/")[1]] = os.path.getsize(os.path.join(root, f)) / 1e6
    out["memory_banks_MB"] = banks
    out["memory_banks_total_MB"] = {ds: sum(v for k, v in banks.items() if f"/{ds}/" in k) for ds in ("avenue", "shanghai")}
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        core(x); v(torch.zeros(64, 3, 224, 224, device="cuda", dtype=xi.dtype))
    out["peak_allocated_MB_yolo_plus_clip_batch64"] = torch.cuda.max_memory_allocated() / 2**20
else:
    sys.path.insert(0, "/workspace/m3/aedmae/scripts")
    import aedmae_speed as s
    for ds in ("avenue", "shanghai"):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model, args = s.build(ds)
        H, W = args.input_size
        x = torch.randn(1, 9, H, W, device="cuda").clamp(-1, 1)
        g = torch.rand(1, 3, H, W, device="cuda") * 255
        t = torch.cat([x[:, 3:6], -torch.ones(1, 1, H, W, device="cuda")], 1)
        f = gfl(lambda: model(x, targets=t, grad_mask=g, mask_ratio=args.mask_ratio))
        ck = f"/workspace/m3/aedmae/checkpoints-aed-mae/{ds}"
        out[ds] = {"input_size": [H, W], "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
                   "gflops_per_frame_batch1": f, "peak_allocated_MB_batch1": torch.cuda.max_memory_allocated() / 2**20,
                   "checkpoint_MB": {n: os.path.getsize(f"{ck}/{n}") / 1e6 for n in os.listdir(ck)}}
        del model
print(json.dumps(out, indent=1))
json.dump(out, open(f"/workspace/m3/size_{which}.json", "w"), indent=1)
