import numpy as np, torch, glob, time, tensorrt as trt, onnxruntime as ort
from vadkit.io.frames import frame_paths, read
from vadkit.perception.embed import OpenClipEncoder
from ultralytics import YOLO
n = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
onnx = sorted(glob.glob("/workspace/speed/trt/cache/*fp32*.onnx"))[0]
det = YOLO("yolo11n.pt"); enc = OpenClipEncoder()
xs = []
for p in frame_paths("/workspace/frames/avenue", "testing")["01"][100:130]:
    im = read(p, 640); bx = det.predict(im, conf=0.25, verbose=False)[0].boxes.xyxyn.cpu().numpy()
    H, W = im.shape[:2]; rgb = torch.from_numpy(im[:, :, ::-1].copy()).cuda().permute(2, 0, 1).float() / 255
    for bb in bx:
        x1, y1, x2, y2 = (bb * [W, H, W, H]).round().astype(int); x1, y1 = max(x1, 0), max(y1, 0); x2, y2 = min(max(x2, x1 + 2), W), min(max(y2, y1 + 2), H)
        c = torch.nn.functional.interpolate(rgb[None, :, y1:y2, x1:x2], size=(enc.size, enc.size), mode="bilinear", align_corners=False)[0]
        xs.append((c - enc.mean) / enc.std)
x = torch.stack(xs[:32]).contiguous()
with torch.no_grad(): ref = enc.model.encode_image(x).float().cpu().numpy()
log = trt.Logger(trt.Logger.ERROR)
def build(flags):
    b = trt.Builder(log); net = b.create_network(0); p = trt.OnnxParser(net, log); assert p.parse(open(onnx, "rb").read())
    cfg = b.create_builder_config()
    for f in flags.get("set", []): cfg.set_flag(f)
    for f in flags.get("clear", []): cfg.clear_flag(f)
    pr = b.create_optimization_profile(); pr.set_shape("image", (1, 3, 256, 256), (8, 3, 256, 256), (64, 3, 256, 256)); cfg.add_optimization_profile(pr)
    eng = trt.Runtime(log).deserialize_cuda_engine(bytes(b.build_serialized_network(net, cfg))); return eng
def run(eng, xb):
    ctx = eng.create_execution_context(); out = torch.empty((len(xb), 512), device="cuda")
    ctx.set_input_shape("image", tuple(xb.shape)); ctx.set_tensor_address("image", xb.data_ptr()); ctx.set_tensor_address("emb", out.data_ptr())
    ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream); torch.cuda.synchronize(); return out.cpu().numpy(), ctx
for name, flags in [("default fp32 (tf32 on)", {}), ("tf32 off", {"clear": [trt.BuilderFlag.TF32]}), ("fp16", {"set": [trt.BuilderFlag.FP16]})]:
    eng = build(flags); r, ctx = run(eng, x)
    xb = x[:7].contiguous(); out = torch.empty((7, 512), device="cuda")
    ctx.set_input_shape("image", (7, 3, 256, 256)); ctx.set_tensor_address("image", xb.data_ptr()); ctx.set_tensor_address("emb", out.data_ptr())
    for _ in range(20): ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(200): ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize(); ms = (time.perf_counter() - s) / 200 * 1e3
    c = (n(ref) * n(r)).sum(1)
    print("%-24s cos min %.6f mean %.6f | 7 crops %.2f ms" % (name, c.min(), c.mean(), ms))
