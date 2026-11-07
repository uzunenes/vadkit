#!/usr/bin/env bash
# Run ON the vast machine (image pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel): packages pinned to the 3-4 Nov runs.
set -euo pipefail
apt-get update -qq && apt-get install -y -qq unzip rsync libxcb1 libgl1 libglib2.0-0 tmux git > /dev/null
pip install -q "ultralytics==8.4.201" lap scipy scikit-learn
pip uninstall -y -q opencv-python opencv-python-headless || true
pip install -q opencv-python-headless==4.11.0.86 opencv-python==4.11.0.86
mkdir -p /workspace/{data,cache,frames,feat2,pilot/logs,repos}
cd /workspace/repos && [ -d NeuFlow_v2 ] || git clone -q https://github.com/neufieldrobotics/NeuFlow_v2
(cd NeuFlow_v2 && git log -1 --format='NeuFlow_v2 %h %ad' --date=short)
cd /workspace/cctv_aean
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); YOLO('yolo11x.pt')" > /dev/null
python -c "from torchvision.models.optical_flow import raft_large, Raft_Large_Weights; raft_large(weights=Raft_Large_Weights.C_T_V2)" > /dev/null
python - <<'EOF'
import cv2, torch, torchvision, ultralytics, json, platform, subprocess
info = {"cv2": cv2.__version__, "torch": torch.__version__, "torchvision": torchvision.__version__,
        "ultralytics": ultralytics.__version__, "gpu": torch.cuda.get_device_name(0), "python": platform.python_version(),
        "cpu": subprocess.run("lscpu | grep 'Model name' | sed 's/.*: *//'", shell=True, capture_output=True, text=True).stdout.strip(),
        "nproc": subprocess.run("nproc", capture_output=True, text=True).stdout.strip()}
assert info["cv2"].startswith("4.11"), info
json.dump(info, open("/workspace/pilot/env.json", "w"), indent=1)
print(info)
EOF
pip freeze > /workspace/pilot/pip_freeze.txt
echo SETUP_OK
