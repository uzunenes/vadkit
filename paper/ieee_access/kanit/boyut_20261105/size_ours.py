"""Size, compute, memory and ONNX portability of our r2 base components (run on the speed instance, repo root)."""
import json, os, time
from pathlib import Path
import numpy as np, torch
from torch.utils.flop_counter import FlopCounterMode

out = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
W = Path("/workspace/sizes"); W.mkdir(exist_ok=True)

def flops(fn, *a):
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            fn(*a)
    return fc.get_total_flops() / 1e9

from ultralytics import YOLO
for w in ["yolo11n.pt", "yolo11n-pose.pt"]:
    m = YOLO(w)
    from ultralytics.utils.torch_utils import get_flops
    npar = sum(q.numel() for q in m.model.parameters())
    gfl = get_flops(m.model, imgsz=640)
    x640 = torch.zeros(1, 3, 640, 640)
    out[w] = {"params_M": npar / 1e6, "gflops_640_ultralytics": gfl, "file_MB": os.path.getsize(m.ckpt_path) / 1e6}
    try:
        f = m.export(format="onnx", imgsz=640, dynamic=False, simplify=False)
        out[w]["onnx_MB"] = os.path.getsize(f) / 1e6; out[w]["onnx_export"] = "ok"
    except Exception as e:
        out[w]["onnx_export"] = f"failed: {e!r}"[:300]

import open_clip
model, _, prep = open_clip.create_model_and_transforms("MobileCLIP2-S0", pretrained="dfndr2b", device="cuda")
model.eval()
vis = model.visual
S = vis.image_size if hasattr(vis, "image_size") else 224
S = S[0] if isinstance(S, (tuple, list)) else int(S)
x = torch.zeros(1, 3, S, S, device="cuda")
nvis = sum(p.numel() for p in vis.parameters()); nall = sum(p.numel() for p in model.parameters())
torch.save({k: v.half() for k, v in vis.state_dict().items()}, W / "mc2s0_visual_fp16.pt")
out["MobileCLIP2-S0"] = {"image_size": S, "visual_params_M": nvis / 1e6, "total_params_incl_text_M": nall / 1e6,
                         "visual_gflops_per_crop": flops(model.encode_image, x),
                         "visual_fp16_MB": os.path.getsize(W / "mc2s0_visual_fp16.pt") / 1e6}
try:
    vc = vis.float().cpu().eval()
    torch.onnx.export(vc, torch.zeros(1, 3, S, S), str(W / "mc2s0_visual.onnx"), opset_version=17,
                      input_names=["image"], output_names=["emb"], dynamic_axes={"image": {0: "n"}, "emb": {0: "n"}})
    import onnxruntime as ort
    sess = ort.InferenceSession(str(W / "mc2s0_visual.onnx"), providers=["CPUExecutionProvider"])
    xx = torch.randn(4, 3, S, S)
    with torch.no_grad():
        ref = vc(xx).numpy()
    got = sess.run(None, {"image": xx.numpy()})[0]
    out["MobileCLIP2-S0"].update(onnx_export="ok", onnx_MB=os.path.getsize(W / "mc2s0_visual.onnx") / 1e6,
                                 onnx_max_abs_diff=float(np.abs(ref - got).max()))
except Exception as e:
    out["MobileCLIP2-S0"]["onnx_export"] = f"failed: {e!r}"[:300]

# fitted per-scene state (densities + calibration references)
for ds in ["avenue", "ped2"]:
    d = Path(f"/workspace/speed/state_{ds}_gpu") if ds == "avenue" else Path(f"/workspace/speed/state_{ds}_cuda")
    out[f"state_{ds}"] = {"files": {p.name: os.path.getsize(p) / 1e6 for p in d.iterdir()},
                          "total_MB": sum(os.path.getsize(p) for p in d.iterdir()) / 1e6}

# peak GPU memory of the streaming perception (Avenue test 01, 300 frames)
del model, vis
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
from vadkit.pipeline.offline import Perception, load_config
from vadkit.io.frames import frame_paths, read
cfg = load_config("vadkit/configs/r2/base.yaml")
p = Perception(cfg, 2, "cuda")
paths = frame_paths("/workspace/frames/avenue", "testing")["01"][:300]
nobj = []
for q in paths:
    o = p(read(q, 640)); nobj.append(len(o))
torch.cuda.synchronize()
import subprocess
smi = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True).stdout
out["runtime"] = {"peak_allocated_MB": torch.cuda.max_memory_allocated() / 2**20,
                  "peak_reserved_MB": torch.cuda.max_memory_reserved() / 2**20,
                  "nvidia_smi_process": smi.strip(), "mean_objects_per_frame": float(np.mean(nobj))}
Path("/workspace/sizes/size_ours.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
