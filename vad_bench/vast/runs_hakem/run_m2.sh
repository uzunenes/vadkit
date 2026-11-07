#!/usr/bin/env bash
# Machine 2 (CPU only): S1 ablation of Avenue (moved from machine 1, where sklearn's full-covariance GMM on 124k rows was
# too slow with 8 threads; same code, same features, same pre-registration) and KISS-1 on the four datasets
# (docs/tez_notlari/kiss1_hiz_buyuklugu_on_kayit_2026-11-06.md). Data copied from machine 1 with `vastai copy`.
#   nohup setsid bash vad_bench/vast/runs_hakem/run_m2.sh > /workspace/logs/run_m2.log 2>&1 < /dev/null &
set -uo pipefail
cd /workspace/cctv_aean
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=""
L=/workspace/logs; O=/workspace/out; F=/workspace/feat/base; C=/workspace/checkpoints
mkdir -p $O/s1 $O/kiss1
stamp() { echo "$(date +%T) $*"; }
ready() { [ "$(find $F -name '*.npz' | wc -l)" -eq 572 ] && grep -q SETUP_OK $L/setup.log 2>/dev/null \
          && for d in ped1 ped2 avenue shanghaitech; do [ -s /workspace/cache/$d/gt.json ] || return 1; done \
          && [ -d $C/scores/runs/avenue_2d_noadv ] && [ -d $C/final_2026-10-07/runs ]; }
until ready; do sleep 20; done
sleep 60                                   # let the last copies settle
stamp "data ready: $(find $F -name '*.npz' | wc -l) feature files"
env OMP_NUM_THREADS=64 MKL_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 python vad_bench/vast/runs_hakem/ablation.py run \
  --dataset avenue --features $F/avenue --gt /workspace/cache/avenue/gt.json --out $O/s1/avenue.json > $L/s1_avenue.log 2>&1 &
grid() {
  if [ "$1" = shanghaitech ]; then
    echo "$C/final_2026-10-07/runs/shtech_{sc}_2d_noadv/eval_shtech_{sc}_pool48_mc5/scores.npz:frx_bayes=I,flow_ubm_m=Fs,flow_rx_m=Fd"
  else
    echo "$C/scores/runs/$1_2d_noadv/eval_$1_pool48_mc4/scores.npz:frx_bayes=I $C/scores/runs/$1_2d_noadv/eval_$1_pool48_mc5/scores.npz:flow_ubm_m=Fs,flow_rx_m=Fd"
  fi
}
for ds in ped1 ped2 avenue shanghaitech; do
  env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 python vad_bench/vast/runs_hakem/kiss1.py --dataset $ds \
    --features $F/$ds --gt /workspace/cache/$ds/gt.json --extra-channels $(grid $ds) --out $O/kiss1/$ds.json \
    > $L/kiss1_$ds.log 2>&1 &
done
wait
stamp M2_DONE
