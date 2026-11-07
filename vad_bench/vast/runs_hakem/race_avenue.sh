#!/usr/bin/env bash
# Machine 2: the Avenue S1 run (PID $1, 64 BLAS threads) spent > 2 h in its D2 row, probably from thread oversubscription
# on small per-component matrix products. Same command with 8 threads, in parallel. Rule fixed before either finishes:
# the run that finishes first (RUN_DONE in its log) is used, the other is killed; no comparison of their AUCs decides it.
# If the 8-thread run is first: kill the 64-thread run, copy its json to s1/avenue.json and append RUN_DONE to
# s1_avenue.log, so that run_m2.sh (M2_DONE) and run_e1.sh (Avenue) proceed as before.
set -uo pipefail
cd /workspace/cctv_aean
export PYTHONPATH=. CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
L=/workspace/logs; O=/workspace/out; F=/workspace/feat/base
python vad_bench/vast/runs_hakem/ablation.py run --dataset avenue --features $F/avenue \
  --gt /workspace/cache/avenue/gt.json --out $O/s1/avenue_t8.json > $L/s1_avenue_t8.log 2>&1 &
T8=$!
while :; do
  if grep -q RUN_DONE $L/s1_avenue.log; then kill $T8; echo "$(date +%T) 64-thread run first; 8-thread run killed"; break; fi
  if grep -q RUN_DONE $L/s1_avenue_t8.log && [ -s $O/s1/avenue_t8.json ]; then
    kill $1; sleep 2; cp $O/s1/avenue_t8.json $O/s1/avenue.json
    echo "$(date +%T) avenue RUN_DONE (8-thread rerun, race_avenue.sh)" >> $L/s1_avenue.log
    echo "$(date +%T) 8-thread run first; 64-thread run killed"; break
  fi
  if ! kill -0 $T8 2>/dev/null && ! grep -q RUN_DONE $L/s1_avenue_t8.log; then echo "$(date +%T) 8-thread run died"; break; fi
  sleep 30
done
