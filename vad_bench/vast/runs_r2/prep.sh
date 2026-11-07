#!/usr/bin/env bash
# Round 2, ON the vast machine after runs_velpilot/setup.sh and dl.sh: gt + JPEG frames (time order) for Ped1, Ped2, Avenue,
# ShanghaiTech, and the archived grid-channel scores (checkpoints). Markers: prep_small.done, prep_sht.done.
set -euo pipefail
cd /workspace/data
R=/workspace/cctv_aean; M=/workspace/r2; mkdir -p $M/logs
pip install -q timm open_clip_torch==3.3.0 huggingface_hub pyyaml
[ -d UCSD_Anomaly_Dataset.v1p2 ] || tar xzf UCSD_Anomaly_Dataset.tar.gz
mkdir -p avenue; [ -f avenue/Avenue_Dataset.zip ] || mv Avenue_Dataset.zip ground_truth_demo.zip avenue/
[ -d /workspace/checkpoints ] || tar xf checkpoints_2026-11-02.tar -C /workspace
python $R/prepare_ucsd.py --root UCSD_Anomaly_Dataset.v1p2 --out /workspace/cache --peds UCSDped2 UCSDped1
python $R/prepare_avenue.py --zip avenue/Avenue_Dataset.zip --gt-zip avenue/ground_truth_demo.zip --out /workspace/cache/avenue --splits test
for d in ped1 ped2; do
  python $R/vad_bench/vast/hazirla_frames.py --ds $d --src UCSD_Anomaly_Dataset.v1p2 --out /workspace/frames --gt /workspace/cache/$d/gt.json &
done
python $R/vad_bench/vast/hazirla_frames.py --ds avenue --src avenue/Avenue_Dataset.zip --out /workspace/frames --gt /workspace/cache/avenue/gt.json &
mkdir -p shanghaitech; [ -n "$(ls -A shanghaitech)" ] || (cat shanghaitech.tar.gz.a? | tar xz -C shanghaitech)
wait
touch $M/prep_small.done
python $R/vad_bench/vast/runs_velpilot/gt_sht.py --root /workspace/data/shanghaitech --out /workspace/cache/shanghaitech/gt.json
# training from the .avi files, test from the archive's testing/frames (107 videos, numeric names)
python $R/vad_bench/vast/hazirla_frames.py --ds shanghaitech --src shanghaitech --test-src shanghaitech --out /workspace/frames \
  --gt /workspace/cache/shanghaitech/gt.json
touch $M/prep_sht.done
echo PREP_OK
