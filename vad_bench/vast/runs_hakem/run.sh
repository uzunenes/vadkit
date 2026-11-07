#!/usr/bin/env bash
# Reviewer-driven measurements (docs/tez_notlari/hakem_duzeltmeleri_on_kayit_2026-11-06.md), ON the vast machine:
#   setsid nohup bash vad_bench/vast/runs_hakem/run.sh > /workspace/logs/run.log 2>&1 < /dev/null &
# 1. S1 scoring ablation on the CPU (archived base features in /workspace/feat/base, uploaded from the laptop), the four
#    datasets in parallel while prep.sh finishes;
# 2. on the then quiet machine (GPU idle state written before every measurement): TensorRT engines, timed fit (G2,
#    also the states of the benches), G1 grid-branch latency, G3 latency spread, S1 cost, G2 timed feature extraction.
set -uo pipefail
cd /workspace/cctv_aean
L=/workspace/logs; O=/workspace/out; F=/workspace/feat/base
mkdir -p $O/s1 $O/g1 $O/g2 $O/g3 $O/cost
export PYTHONPATH=.
stamp() { echo "$(date +%T) $*"; }
idle() { { date -Is; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader;
           nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader; uptime; } > "$1" 2>&1; }
elapsed() { python -c "import sys; print(float(sys.argv[2]) - float(sys.argv[1]))" "$1" "$2"; }
cfg() { case "$1/$2" in shanghaitech/pt) echo vadkit/configs/r2/base_sht.yaml;; shanghaitech/trt) echo vadkit/configs/r2/base_trt_sht.yaml;;
        */trt) echo vadkit/configs/r2/base_trt.yaml;; *) echo vadkit/configs/r2/base.yaml;; esac; }
T8="OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8"

while [ ! -f $L/feat.done ] || [ ! -f $L/gt_small.done ]; do sleep 20; done
stamp "S1 start"
for ds in ped2 ped1 avenue; do
  env $T8 CUDA_VISIBLE_DEVICES="" nice -n 5 python vad_bench/vast/runs_hakem/ablation.py run --dataset $ds \
    --features $F/$ds --gt /workspace/cache/$ds/gt.json --out $O/s1/$ds.json > $L/s1_$ds.log 2>&1 &
done
( while [ ! -f $L/gt_sht.done ]; do sleep 20; done
  env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 CUDA_VISIBLE_DEVICES="" nice -n 5 \
    python vad_bench/vast/runs_hakem/ablation.py run --dataset shanghaitech --features $F/shanghaitech \
    --gt /workspace/cache/shanghaitech/gt.json --out $O/s1/shanghaitech.json > $L/s1_shanghaitech.log 2>&1 ) &
wait
stamp "S1 done"
while [ ! -f $L/prep_sht.done ]; do sleep 20; done
stamp "quiet phase"

# TensorRT engines (one-time per GPU type), as kanit/hiz_20261105/c2_step2/export.log
t0=$(date +%s.%N)
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='engine', half=True, dynamic=True); YOLO('yolo11n-pose.pt').export(format='engine', half=True, dynamic=True)" > $O/g2/export.log 2>&1
echo "ELAPSED_S $(elapsed $t0 $(date +%s.%N))" >> $O/g2/export.log

# G2a: timed fit on the GPU (its states serve the benches)
for ds in ped2 ped1 avenue shanghaitech; do
  X=""; [ $ds = shanghaitech ] && X="--scenes"
  idle $O/g2/idle_fit_$ds.txt; t0=$(date +%s.%N)
  python -m vadkit fit --config $(cfg $ds pt) --features $F/$ds --out $O/g2/state_$ds --device cuda $X > $O/g2/fit_$ds.log 2>&1
  echo "ELAPSED_S $(elapsed $t0 $(date +%s.%N))" >> $O/g2/fit_$ds.log
done
stamp "G2a done"

# G1: grid branch latency (GPU and 4 CPU threads), 3 x 100 frames
for ds in ped2 avenue; do
  gap=1; [ $ds = avenue ] && gap=2
  for dev in cuda cpu; do
    idle $O/g1/idle_${ds}_$dev.txt
    python tools/bench_pipeline.py --data /workspace/cache/$ds --run /workspace/checkpoints/runs/${ds}_2d_noadv \
      --reference-dir /workspace/checkpoints/scores/runs/${ds}_2d_noadv --device $dev --threads 4 --frames 100 \
      --repeats 3 --gap $gap --out $O/g1/${ds}_$dev.json > $O/g1/${ds}_$dev.log 2>&1
  done
done
stamp "G1 done"

# G3: latency spread, parallel flow, TensorRT YOLO and PyTorch, the four datasets
for ds in ped2 ped1 avenue shanghaitech; do
  for be in trt pt; do
    idle $O/g3/idle_${ds}_$be.txt
    python -m vadkit bench --config $(cfg $ds $be) --frames /workspace/frames/$ds --dataset $ds --state $O/g2/state_$ds \
      --parallel-flow --out $O/g3/${ds}_$be.json > $O/g3/${ds}_$be.log 2>&1
  done
done
stamp "G3 done"

# S1 cost: NIW vs 1-NN scoring, 4 CPU threads, alone
for ds in ped2 ped1 avenue shanghaitech; do
  python vad_bench/vast/runs_hakem/ablation.py cost --dataset $ds --features $F/$ds --threads 4 --out $O/cost/$ds.json \
    > $O/cost/$ds.log 2>&1
done
stamp "cost done"

# G2b: timed feature extraction of the training split, one process, PyTorch back-end (as the archived features)
for ds in ped2 ped1 avenue shanghaitech; do
  te=1; [ $ds = shanghaitech ] && te=5
  idle $O/g2/idle_extract_$ds.txt; t0=$(date +%s.%N)
  python -m vadkit extract --config $(cfg $ds pt) --frames /workspace/frames/$ds --dataset $ds --splits training \
    --train-every $te --out $O/g2/feat/$ds > $O/g2/extract_$ds.log 2>&1
  echo "ELAPSED_S $(elapsed $t0 $(date +%s.%N))" >> $O/g2/extract_$ds.log
done
stamp "G2b done"
nvidia-smi > $O/nvidia_smi_end.txt 2>&1; uptime >> $O/nvidia_smi_end.txt
stamp ALL_DONE
