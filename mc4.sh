#!/bin/sh
# Round 4: same evaluations as rounds 2-3, now saving cross-fitted normal-frame reference scores of every channel,
# then leave-one-dataset-out selection of the channel combination and normalisation (tools/fuse_runs.py).
cd /workspace/cctv_aean
LOG=/workspace/mc4.log
lane() {
  ds=$1; gap=$2
  ev() { python evaluate.py --pool 48 --tag pool48_mc4 --no-state "$@" 2>&1 | tail -1 | cut -c1-90 >> $LOG; }
  ev --data /workspace/cache/$ds --run runs/${ds}_2d_noadv --flow-gap $gap --workers 6
  ev --data /workspace/cache/$ds --run runs/${ds}_3d_noadv --channels ""
  ev --data /workspace/cache/${ds}_flowmag --run runs/${ds}_flowmag_2d_noadv --channels ""
  ev --data /workspace/cache/${ds}_flowmag --run runs/${ds}_flowmag_3d_noadv --channels ""
  echo "$ds lane done" >> $LOG
}
lane ped2 1 &
lane ped1 1 &
lane avenue 2 &
wait
python tools/fuse_runs.py ped2 ped1 avenue >> $LOG 2>&1
echo MC4_DONE >> $LOG
