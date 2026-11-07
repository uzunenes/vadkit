import numpy as np, torch, open_clip, onnxruntime as ort, glob
from vadkit.perception.embed import TensorRTEncoder
enc = TensorRTEncoder(engine_dir="/workspace/speed/trt/cache", fp16=False)
model, _, _ = open_clip.create_model_and_transforms("MobileCLIP2-S0", pretrained="dfndr2b", device="cuda"); model.eval()
torch.manual_seed(0)
n = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
onnx = sorted(glob.glob("/workspace/speed/trt/cache/*fp32*.onnx"))[0]
sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
for bs in (1, 3, 7, 8, 16):
    x = torch.randn(bs, 3, 256, 256, device="cuda")
    with torch.no_grad():
        t = model.encode_image(x).float().cpu().numpy()
    r = enc.model.encode_image(x); torch.cuda.synchronize(); r = r.cpu().numpy()
    o = sess.run(None, {"image": x.cpu().numpy()})[0]
    print(bs, "trt-vs-torch cos min %.6f" % (n(t) * n(r)).sum(1).min(), "| onnx-vs-torch %.6f" % (n(t) * n(o)).sum(1).min(),
          "| trt-vs-onnx %.6f" % (n(o) * n(r)).sum(1).min(), "| norms", np.linalg.norm(t, axis=1)[:2], np.linalg.norm(r, axis=1)[:2])
