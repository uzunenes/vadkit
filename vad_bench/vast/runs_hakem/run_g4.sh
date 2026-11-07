#!/usr/bin/env bash
# E1 addendum (docs/tez_notlari/e1_pozsuz_hiz_olcumu_on_kayit_2026-11-06.md), ON the GPU machine after prep_g4.sh:
# with-pose (base*) and pose-off (base_nopose*) configurations in the same session on Ped1, Ped2 and ShanghaiTech:
# TensorRT engines, timed `vadkit fit` on the GPU (states for the benches), then `vadkit bench` as G3 (parallel flow,
# 20 warm-up frames, 3 x 100 timed frames, TensorRT YOLO and PyTorch), with-pose and pose-off alternating.
#   nohup setsid bash vad_bench/vast/runs_hakem/run_g4.sh > /workspace/logs/run_g4.log 2>&1 < /dev/null &
set -uo pipefail
cd /workspace/cctv_aean
export PYTHONPATH=.
L=/workspace/logs; O=/workspace/out/g4; F=${FEAT:-/workspace/feat/base}
mkdir -p $O
stamp() { echo "$(date +%T) $*"; }
idle() { { date -Is; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader;
           nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader; uptime; } > "$1" 2>&1; }
elapsed() { python -c "import sys; print(float(sys.argv[2]) - float(sys.argv[1]))" "$1" "$2"; }
cfg() {   # cfg <dataset> <pose|nopose> <pt|trt>
  local n=base; [ "$2" = nopose ] && n=${n}_nopose; [ "$3" = trt ] && n=${n}_trt; [ "$1" = shanghaitech ] && n=${n}_sht
  echo vadkit/configs/r2/$n.yaml; }
until [ -f $L/prep_g4.done ]; do sleep 20; done
stamp start
t0=$(date +%s.%N)
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='engine', half=True, dynamic=True); YOLO('yolo11n-pose.pt').export(format='engine', half=True, dynamic=True)" > $O/export.log 2>&1
echo "ELAPSED_S $(elapsed $t0 $(date +%s.%N))" >> $O/export.log
stamp "engines done"
for ds in ped2 ped1 shanghaitech; do
  X=""; [ $ds = shanghaitech ] && X="--scenes"
  for p in pose nopose; do
    idle $O/idle_fit_${ds}_$p.txt; t0=$(date +%s.%N)
    python -m vadkit fit --config $(cfg $ds $p pt) --features $F/$ds --out $O/state_${ds}_$p --device cuda $X > $O/fit_${ds}_$p.log 2>&1
    echo "ELAPSED_S $(elapsed $t0 $(date +%s.%N))" >> $O/fit_${ds}_$p.log
  done
done
stamp "fits done"
for ds in ped2 ped1 shanghaitech; do
  for be in trt pt; do
    for p in pose nopose; do
      idle $O/idle_${ds}_${p}_$be.txt
      python -m vadkit bench --config $(cfg $ds $p $be) --frames /workspace/frames/$ds --dataset $ds \
        --state $O/state_${ds}_$p --parallel-flow --out $O/${ds}_${p}_$be.json > $O/${ds}_${p}_$be.log 2>&1
    done
  done
done
stamp "benches done"
nvidia-smi > $O/nvidia_smi_end.txt 2>&1; uptime >> $O/nvidia_smi_end.txt
stamp ALL_DONE
