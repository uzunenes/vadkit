#!/usr/bin/env bash
# Reviewer-driven measurements (docs/tez_notlari/hakem_duzeltmeleri_on_kayit_2026-11-06.md), ON the vast machine
# (image pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel): packages pinned to the 4-5 Nov runs, TensorRT 10.3.0 as in
# kanit/hiz_20261105/c2_step2/export.log. Writes /workspace/logs/env.json.
set -euo pipefail
apt-get update -qq && apt-get install -y -qq unzip rsync libxcb1 libgl1 libglib2.0-0 tmux > /dev/null
pip install -q "ultralytics==8.4.201" lap scipy scikit-learn timm open_clip_torch==3.3.0 huggingface_hub pyyaml \
  faiss-cpu "tensorrt==10.3.0" "onnx>=1.12.0,<2.0.0" "onnxslim>=0.1.82"
pip uninstall -y -q opencv-python opencv-python-headless || true
pip install -q opencv-python-headless==4.11.0.86 opencv-python==4.11.0.86
cd /workspace/cctv_aean
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); YOLO('yolo11n-pose.pt')" > /dev/null
python -c "import open_clip; open_clip.create_model_and_transforms('MobileCLIP2-S0', pretrained='dfndr2b')" > /dev/null
python - <<'EOF'
import cv2, torch, torchvision, ultralytics, json, platform, subprocess, tensorrt, sklearn, faiss
info = {"cv2": cv2.__version__, "torch": torch.__version__, "torchvision": torchvision.__version__,
        "ultralytics": ultralytics.__version__, "tensorrt": tensorrt.__version__, "sklearn": sklearn.__version__,
        "faiss": faiss.__version__, "gpu": torch.cuda.get_device_name(0), "python": platform.python_version(),
        "cpu": subprocess.run("lscpu | grep 'Model name' | sed 's/.*: *//'", shell=True, capture_output=True, text=True).stdout.strip(),
        "nproc": subprocess.run("nproc", capture_output=True, text=True).stdout.strip()}
assert info["cv2"].startswith("4.11"), info
json.dump(info, open("/workspace/logs/env.json", "w"), indent=1)
print(info)
EOF
pip freeze > /workspace/logs/pip_freeze.txt
echo SETUP_OK
