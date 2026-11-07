#!/usr/bin/env bash
# Run on the GPU machine: dependencies + UCSD dataset + cache.
set -euo pipefail
cd "$(dirname "$0")"
pip install -q scikit-learn scipy matplotlib pillow
mkdir -p /workspace/data && cd /workspace/data
[ -f UCSD_Anomaly_Dataset.tar.gz ] || wget -q http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz
[ -d UCSD_Anomaly_Dataset.v1p2 ] || tar xzf UCSD_Anomaly_Dataset.tar.gz
cd - >/dev/null
python prepare_ucsd.py --root /workspace/data/UCSD_Anomaly_Dataset.v1p2 --out /workspace/cache
