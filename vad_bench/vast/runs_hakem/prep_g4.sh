#!/usr/bin/env bash
# E1 addendum (docs/tez_notlari/e1_pozsuz_hiz_olcumu_on_kayit_2026-11-06.md), ON the GPU machine after setup.sh and
# runs_velpilot/dl.sh (UCSD + ShanghaiTech tarballs): test frames of Ped1, Ped2 and ShanghaiTech for `vadkit bench`.
# ShanghaiTech: test frames only (an empty training source), checked against gt.json like prep.sh.
set -euo pipefail
CACHE=${CACHE:-/workspace/cache}      # gt.json per dataset
cd /workspace/data
R=/workspace/cctv_aean; L=/workspace/logs
[ -d UCSD_Anomaly_Dataset.v1p2 ] || tar xzf UCSD_Anomaly_Dataset.tar.gz
mkdir -p shanghaitech empty
[ -n "$(ls -A shanghaitech)" ] || (cat shanghaitech.tar.gz.a? | tar xz -C shanghaitech)
cd $R
for d in ped1 ped2; do
  python vad_bench/vast/hazirla_frames.py --ds $d --src /workspace/data/UCSD_Anomaly_Dataset.v1p2 --out /workspace/frames \
    --gt $CACHE/$d/gt.json &
done
python vad_bench/vast/hazirla_frames.py --ds shanghaitech --src /workspace/data/empty --test-src /workspace/data/shanghaitech \
  --out /workspace/frames --gt $CACHE/shanghaitech/gt.json &
wait
touch $L/prep_g4.done
echo PREP_OK
