import numpy as np, torch, open_clip, onnxruntime as ort, glob
from vadkit.io.frames import frame_paths, read
from vadkit.perception.embed import OpenClipEncoder
from ultralytics import YOLO
n = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
onnx = sorted(glob.glob("/workspace/speed/trt/cache/*fp32*.onnx"))[0]
sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
det = YOLO("yolo11n.pt"); enc = OpenClipEncoder()
xs = []
for p in frame_paths("/workspace/frames/avenue", "testing")["01"][100:130]:
    im = read(p, 640); bx = det.predict(im, conf=0.25, verbose=False)[0].boxes.xyxyn.cpu().numpy()
    if not len(bx): continue
    H, W = im.shape[:2]; rgb = torch.from_numpy(im[:, :, ::-1].copy()).cuda().permute(2, 0, 1).float() / 255
    for bb in bx:
        x1, y1, x2, y2 = (bb * [W, H, W, H]).round().astype(int); x1, y1 = max(x1, 0), max(y1, 0); x2, y2 = min(max(x2, x1 + 2), W), min(max(y2, y1 + 2), H)
        c = torch.nn.functional.interpolate(rgb[None, :, y1:y2, x1:x2], size=(enc.size, enc.size), mode="bilinear", align_corners=False)[0]
        xs.append((c - enc.mean) / enc.std)
x = torch.stack(xs)
o = sess.run(None, {"image": x.cpu().numpy()})[0]
for tf32 in (True, False):
    torch.backends.cudnn.allow_tf32 = tf32; torch.backends.cuda.matmul.allow_tf32 = tf32
    with torch.no_grad(): t = enc.model.encode_image(x).float().cpu().numpy()
    print("cuda tf32=%s vs exact fp32 (onnx cpu): cos min %.6f mean %.6f  (crops %d)" % (tf32, (n(t)*n(o)).sum(1).min(), (n(t)*n(o)).sum(1).mean(), len(x)))
