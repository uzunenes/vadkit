#!/usr/bin/env bash
# ON the vast machine after setup.sh and dl.sh: gt, JPEG frames (time order) of the four datasets, the grid caches of
# Ped2 and Avenue (training + test, for tools/bench_pipeline.py) and the archived checkpoints (grid AE + scores).
# Same steps as runs_r2/prep.sh, except that Avenue's cache also gets the training split. Markers in /workspace/logs.
set -euo pipefail
cd /workspace/data
R=/workspace/cctv_aean; L=/workspace/logs
[ -d UCSD_Anomaly_Dataset.v1p2 ] || tar xzf UCSD_Anomaly_Dataset.tar.gz
mkdir -p avenue; [ -f avenue/Avenue_Dataset.zip ] || mv Avenue_Dataset.zip ground_truth_demo.zip avenue/
[ -d /workspace/checkpoints ] || tar xf checkpoints_2026-11-02.tar -C /workspace
cd $R
python prepare_ucsd.py --root /workspace/data/UCSD_Anomaly_Dataset.v1p2 --out /workspace/cache --peds UCSDped2 UCSDped1
python prepare_avenue.py --zip /workspace/data/avenue/Avenue_Dataset.zip --gt-zip /workspace/data/avenue/ground_truth_demo.zip \
  --out /workspace/cache/avenue --splits train,test
touch $L/gt_small.done
for d in ped1 ped2; do
  python vad_bench/vast/hazirla_frames.py --ds $d --src /workspace/data/UCSD_Anomaly_Dataset.v1p2 --out /workspace/frames \
    --gt /workspace/cache/$d/gt.json &
done
python vad_bench/vast/hazirla_frames.py --ds avenue --src /workspace/data/avenue/Avenue_Dataset.zip --out /workspace/frames \
  --gt /workspace/cache/avenue/gt.json &
mkdir -p /workspace/data/shanghaitech
[ -n "$(ls -A /workspace/data/shanghaitech)" ] || (cat /workspace/data/shanghaitech.tar.gz.a? | tar xz -C /workspace/data/shanghaitech)
wait
touch $L/prep_small.done
python vad_bench/vast/runs_velpilot/gt_sht.py --root /workspace/data/shanghaitech --out /workspace/cache/shanghaitech/gt.json
touch $L/gt_sht.done
python vad_bench/vast/hazirla_frames.py --ds shanghaitech --src /workspace/data/shanghaitech --test-src /workspace/data/shanghaitech \
  --out /workspace/frames --gt /workspace/cache/shanghaitech/gt.json
touch $L/prep_sht.done
echo PREP_OK
