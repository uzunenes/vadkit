#!/bin/sh
# CUHK Avenue with the protocol fixed on UCSD (no tuning on Avenue): no-adv 2d/3d, 20k steps,
# 48 px window (chosen on Ped2) as the main result, 16 px for reference.
# Avenue is 25 fps (UCSD 10 fps), so 3d clips use a frame step of 2: 16 frames cover 1.24 s (UCSD 1.5 s).
cd /workspace/cctv_aean
LOG=/workspace/avenue.log
D=/workspace/data/avenue
until [ -f $D/dl.done ]; do sleep 30; done
[ -f /workspace/cache/avenue/gt.json ] || python prepare_avenue.py --zip $D/Avenue_Dataset.zip \
    --gt-zip $D/ground_truth_demo.zip --out /workspace/cache/avenue >> $LOG 2>&1
for v in 2d 3d; do
  out=runs/avenue_${v}_noadv
  [ -f $out/model.pt ] || python train.py --data /workspace/cache/avenue --variant $v --no-adv --tstride 2 \
      --out $out > $out.log 2>&1
  python evaluate.py --data /workspace/cache/avenue --run $out --pool 48 --tag pool48 2>&1 | tail -1 | sed "s|^|$out pool48: |" >> $LOG
  python evaluate.py --data /workspace/cache/avenue --run $out 2>&1 | tail -1 | sed "s|^|$out pool16: |" >> $LOG
done
echo AVENUE_DONE >> $LOG
