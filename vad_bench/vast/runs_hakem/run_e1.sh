#!/usr/bin/env bash
# Machine 2: E1 checks (a) and (c) on the four datasets; Avenue waits for its S1 run (needs s1/avenue.json).
set -uo pipefail
cd /workspace/cctv_aean
export PYTHONPATH=. CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
L=/workspace/logs; O=/workspace/out; F=/workspace/feat/base; C=/workspace/checkpoints; S1=/workspace/s1_laptop
mkdir -p $O/e1
grid() {
  if [ "$1" = shanghaitech ]; then
    echo "$C/final_2026-10-07/runs/shtech_{sc}_2d_noadv/eval_shtech_{sc}_pool48_mc5/scores.npz:frx_bayes=I,flow_ubm_m=Fs,flow_rx_m=Fd"
  else
    echo "$C/scores/runs/$1_2d_noadv/eval_$1_pool48_mc4/scores.npz:frx_bayes=I $C/scores/runs/$1_2d_noadv/eval_$1_pool48_mc5/scores.npz:flow_ubm_m=Fs,flow_rx_m=Fd"
  fi
}
run() { python vad_bench/vast/runs_hakem/e1.py --dataset $1 --features $F/$1 --gt /workspace/cache/$1/gt.json --s1 $2 \
          --extra-channels $(grid $1) --out $O/e1/$1.json > $L/e1_$1.log 2>&1; }
for ds in ped1 ped2 shanghaitech; do run $ds $S1/$ds.json & done
( until grep -q RUN_DONE $L/s1_avenue.log 2>/dev/null; do sleep 30; done; run avenue $O/s1/avenue.json ) &
wait
echo E1_ALL_DONE >> $L/run_e1.log
