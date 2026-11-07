#!/usr/bin/env bash
# Speed + fit-time measurements on a quiet RTX 3090 (speed instance). Logs in /workspace/speed.
set -uo pipefail
cd /workspace/cctv_aean
O=/workspace/speed; mkdir -p $O
nvidia-smi --query-gpu=name,driver_version,utilization.gpu --format=csv > $O/gpu.txt; uptime >> $O/gpu.txt
C=vadkit/configs/r2/base.yaml
# 2) fit time on the quiet machine (GPU and CPU), Avenue and Ped2; the Avenue GPU state is reused by bench
for ds in avenue ped2; do
  t0=$(date +%s.%N); python -m vadkit fit --config $C --features /workspace/r2/feat/base/$ds --out $O/state_${ds}_gpu --device cuda > $O/fit_${ds}_gpu.log 2>&1; echo "ELAPSED_S $(echo "$(date +%s.%N) - $t0" | python -c "import sys; print(eval(sys.stdin.read()))")" >> $O/fit_${ds}_gpu.log
  t0=$(date +%s.%N); python -m vadkit fit --config $C --features /workspace/r2/feat/base/$ds --out $O/state_${ds}_cpu --device cpu > $O/fit_${ds}_cpu.log 2>&1; echo "ELAPSED_S $(echo "$(date +%s.%N) - $t0" | python -c "import sys; print(eval(sys.stdin.read()))")" >> $O/fit_${ds}_cpu.log
done
# 3) latency, batch 1, Avenue test frames in time order (20 warm-up + 3 x 100)
for par in "" "--parallel-flow"; do
  tag=seq; [ -n "$par" ] && tag=par
  python -m vadkit bench --config $C --frames /workspace/frames/avenue --dataset avenue --state $O/state_avenue_gpu \
    --out $O/bench_${tag}_allcores.json $par > $O/bench_${tag}_allcores.log 2>&1
  OMP_NUM_THREADS=4 VADKIT_CV2_THREADS=4 python -m vadkit bench --config $C --frames /workspace/frames/avenue --dataset avenue \
    --state $O/state_avenue_gpu --out $O/bench_${tag}_4cores.json $par > $O/bench_${tag}_4cores.log 2>&1
done
uptime >> $O/gpu.txt
echo SPEED_DONE > $O/done
