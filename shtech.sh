#!/bin/sh
# ShanghaiTech Campus with the final method, per scene (fixed protocol: 180x320, pool 48, cell 8, flow step 2 at ~24 fps).
cd /workspace/cctv_aean
LOG=/workspace/shtech.log
# $1 = the zip, or the folder holding the official split archive shanghaitech.tar.gz.a? (extracted to /workspace/data)
if [ ! -f /workspace/cache/shtech_01/gt.json ]; then
  if [ -d "$1" ]; then
    [ -d /workspace/data/shanghaitech/testing ] || cat "$1"/shanghaitech.tar.gz.a? | tar xz -C /workspace/data
    python prepare_shanghaitech.py --root /workspace/data/shanghaitech --out /workspace/cache >> $LOG 2>&1 \
      || { echo "PREPARE_FAILED" >> $LOG; exit 1; }
  else
    python prepare_shanghaitech.py --zip "$1" --out /workspace/cache >> $LOG 2>&1 \
      || { echo "PREPARE_FAILED" >> $LOG; exit 1; }
  fi
fi
# the official test set covers scenes 01-12 (scene 13 has training videos only)
[ "$(ls /workspace/cache/shtech_*/gt.json 2>/dev/null | wc -l)" -ge 12 ] || { echo "PREPARE_FAILED" >> $LOG; exit 1; }
lane() {
  for d in "$@"; do
    sc=${d##*_}; out=runs/shtech_${sc}_2d_noadv
    [ -f $out/model.pt ] || python train.py --data $d --variant 2d --no-adv --out $out > $out.log 2>&1
    python evaluate.py --data $d --run $out --pool 48 --tag pool48_mc5 --no-state --channels flowm,flowrxm \
        --flow-gap 2 --workers 6 2>&1 | tail -1 | cut -c1-90 | sed "s|^|$sc: |" >> $LOG
  done
}
S=$(ls /workspace/cache/shtech_*/gt.json | xargs -n1 dirname | sort)
lane $(echo $S | tr ' ' '\n' | awk 'NR%2==1') &
lane $(echo $S | tr ' ' '\n' | awk 'NR%2==0') &
wait
python tools/shtech_pooled.py >> $LOG 2>&1
echo SHTECH_DONE >> $LOG
