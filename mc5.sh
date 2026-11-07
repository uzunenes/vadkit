#!/bin/sh
# Round 5: better optical flow (DIS 'medium' preset) and DINOv2 ViT-S/14 cell features, then LODO fusion over all channels.
cd /workspace/cctv_aean
LOG=/workspace/mc5.log
ev() { python evaluate.py --pool 48 --tag pool48_mc5 --no-state --channels flowm,flowrxm,dino --workers 6 "$@" 2>&1 | tail -1 | cut -c1-100 >> $LOG; }
ev --data /workspace/cache/ped2 --run runs/ped2_2d_noadv --flow-gap 1 &
ev --data /workspace/cache/ped1 --run runs/ped1_2d_noadv --flow-gap 1 &
ev --data /workspace/cache/avenue --run runs/avenue_2d_noadv --flow-gap 2 &
wait
python tools/fuse_runs.py ped2 ped1 avenue > /workspace/mc5_fusion.json 2>&1
echo MC5_DONE >> $LOG
