#!/usr/bin/env bash
# Write env.json: hardware and software of the machine a result was produced on.
set -euo pipefail
out=${1:-env.json}
python3 - "$out" <<'PY'
import json, platform, subprocess, sys
def run(c):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e: return f"ERR {e}"
info = {
  "host": platform.node(), "kernel": platform.release(), "python": platform.python_version(),
  "cpu_model": run("lscpu | sed -n 's/^Model name:[ \\t]*//p' | head -1"),
  "cpu_count": run("nproc"), "cpu_flags_avx": run("grep -o 'avx[0-9a-z_]*' /proc/cpuinfo | sort -u | tr '\\n' ' '"),
  "mem": run("free -g | sed -n 2p"),
  "gpu": run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
  "cuda": run("nvcc --version | tail -1"),
  "torch": run("python3 -c 'import torch;print(torch.__version__, torch.version.cuda)'"),
}
json.dump(info, open(sys.argv[1], "w"), indent=1); print(json.dumps(info, indent=1))
PY
