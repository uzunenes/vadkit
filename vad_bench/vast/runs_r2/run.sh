#!/usr/bin/env bash
# Round 2 component study (docs/tez_notlari/tur2_on_kayit_2026-11-05.md), ON the vast machine:
#   setsid nohup bash vad_bench/vast/runs_r2/run.sh > /workspace/r2/logs/run.log 2>&1 < /dev/null &
# For each config: vadkit extract (N shards), then eval of the object branch alone and with the grid branch (I, Fs, Fd
# archived cell channels). Latency benchmarks run at the end on a quiet machine.
set -uo pipefail
cd /workspace/cctv_aean
R=/workspace/r2; L=$R/logs; N=${N:-12}
CFGS=${CFGS:-"base enc_mc2s2 enc_mc2b enc_dinov2s enc_clipb16 pose_y11s det_y11s det_y11m"}
DSS=${DSS:-"ped1 ped2 avenue shanghaitech"}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $L $R/res $R/bench
stamp() { echo "$(date +%T) $*"; }
grid() {
  local C=/workspace/checkpoints
  if [ "$1" = shanghaitech ]; then
    echo "$C/final_2026-10-07/runs/shtech_{sc}_2d_noadv/eval_shtech_{sc}_pool48_mc5/scores.npz:frx_bayes=I,flow_ubm_m=Fs,flow_rx_m=Fd"
  else
    echo "$C/scores/runs/$1_2d_noadv/eval_$1_pool48_mc4/scores.npz:frx_bayes=I $C/scores/runs/$1_2d_noadv/eval_$1_pool48_mc5/scores.npz:flow_ubm_m=Fs,flow_rx_m=Fd"
  fi
}
for cfg in $CFGS; do
  for ds in $DSS; do
    m=$R/prep_small.done; [ $ds = shanghaitech ] && m=$R/prep_sht.done
    while [ ! -f $m ]; do sleep 30; done
    C=vadkit/configs/r2/$cfg.yaml; X=""; [ $ds = shanghaitech ] && { C=vadkit/configs/r2/${cfg}_sht.yaml; X="--scenes"; }
    te=1; [ $ds = shanghaitech ] && te=5
    stamp "$cfg $ds extract"
    for i in $(seq 0 $((N - 1))); do
      VADKIT_CV2_THREADS=1 OMP_NUM_THREADS=2 python -m vadkit extract --config $C --frames /workspace/frames/$ds --dataset $ds \
        --out $R/feat/$cfg/$ds --train-every $te --shard $i/$N > $L/${cfg}_${ds}_x$i.log 2>&1 &
    done
    wait
    mkdir -p $R/res/$cfg
    ( python -m vadkit eval --config $C --features $R/feat/$cfg/$ds --gt /workspace/cache/$ds/gt.json $X \
        --out $R/res/$cfg/${ds}_obj.json > $L/${cfg}_${ds}_eval_obj.log 2>&1
      python -m vadkit eval --config $C --features $R/feat/$cfg/$ds --gt /workspace/cache/$ds/gt.json $X \
        --extra-channels $(grid $ds) --out $R/res/$cfg/${ds}_grid.json > $L/${cfg}_${ds}_eval_grid.log 2>&1 ) &
  done
done
wait
stamp "latency (quiet machine)"
for cfg in $CFGS; do
  python -m vadkit bench --config vadkit/configs/r2/$cfg.yaml --frames /workspace/frames/avenue --dataset avenue \
    --out $R/bench/$cfg.json > $L/bench_$cfg.log 2>&1
done
stamp R2_DONE
