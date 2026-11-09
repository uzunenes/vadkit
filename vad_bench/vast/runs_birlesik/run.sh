#!/usr/bin/env bash
# Combined streaming detector (docs/tez_notlari/birlesik_hiz_on_kayit_2026-11-08.md), ON the vast machine after
# runs_hakem/setup.sh and dl.sh (UCSD, Avenue + gt, checkpoints):
#   setsid nohup bash vad_bench/vast/runs_birlesik/run.sh > /workspace/logs/run.log 2>&1 < /dev/null &
# 1. prep: Ped2 / Avenue grey caches (grid) and JPEG frames; 2. object features (training + testing, r2/base) and fit;
# 3. grid fit + check against the archive (GPU); 4. TensorRT engines; 5. Avenue streamed through the combined detector
# (accuracy check + Figure 3 data); 6. end-to-end latency, object vs object + grid, ABBAAB, PyTorch and TensorRT.
set -uo pipefail
cd /workspace/cctv_aean
L=/workspace/logs; O=/workspace/out; CK=/workspace/checkpoints; D=/workspace/data
mkdir -p $L $O/feat $O/state $O/grid $O/bench $O/stream
export PYTHONPATH=.
stamp() { echo "$(date +%T) $*"; }
cfg() { [ "$1" = trt ] && echo vadkit/configs/r2/base_trt.yaml || echo vadkit/configs/r2/base.yaml; }

stamp prep
cd $D
[ -d UCSD_Anomaly_Dataset.v1p2 ] || tar xzf UCSD_Anomaly_Dataset.tar.gz
mkdir -p avenue; [ -f avenue/Avenue_Dataset.zip ] || mv Avenue_Dataset.zip ground_truth_demo.zip avenue/
[ -d $CK ] || tar xf checkpoints_2026-11-02.tar -C /workspace
cd /workspace/cctv_aean
python prepare_ucsd.py --root $D/UCSD_Anomaly_Dataset.v1p2 --out /workspace/cache --peds UCSDped2 > $L/prep_ucsd.log 2>&1
python prepare_avenue.py --zip $D/avenue/Avenue_Dataset.zip --gt-zip $D/avenue/ground_truth_demo.zip \
  --out /workspace/cache/avenue --splits train,test > $L/prep_avenue.log 2>&1
python vad_bench/vast/hazirla_frames.py --ds ped2 --src $D/UCSD_Anomaly_Dataset.v1p2 --out /workspace/frames \
  --gt /workspace/cache/ped2/gt.json > $L/frames_ped2.log 2>&1 &
python vad_bench/vast/hazirla_frames.py --ds avenue --src $D/avenue/Avenue_Dataset.zip --out /workspace/frames \
  --gt /workspace/cache/avenue/gt.json > $L/frames_avenue.log 2>&1 &
wait
stamp "prep done"

for ds in ped2 avenue; do
  python -m vadkit extract --config vadkit/configs/r2/base.yaml --frames /workspace/frames/$ds --dataset $ds \
    --out $O/feat/$ds --device cuda > $L/extract_$ds.log 2>&1
  python -m vadkit fit --config vadkit/configs/r2/base.yaml --features $O/feat/$ds --out $O/state/$ds --device cuda \
    > $L/fit_$ds.log 2>&1
  python -m vadkit eval --config vadkit/configs/r2/base.yaml --features $O/feat/$ds --state $O/state/$ds \
    --gt /workspace/cache/$ds/gt.json --out $O/state/eval_obj_$ds.json > $L/eval_obj_$ds.log 2>&1
  stamp "object $ds done"
done

for ds in ped2 avenue; do
  python tools/grid_check.py --cache /workspace/cache/$ds --dataset $ds --run $CK/runs/${ds}_2d_noadv \
    --archive $CK/scores/runs/${ds}_2d_noadv --state $O/grid/state_$ds --out $O/grid/check_$ds.json --device cuda \
    --workers 8 > $L/grid_$ds.log 2>&1
  stamp "grid $ds done"
done

python -c "from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='engine', half=True, dynamic=True); YOLO('yolo11n-pose.pt').export(format='engine', half=True, dynamic=True)" > $L/export.log 2>&1
stamp "trt done"

python tools/stream_eval.py --config vadkit/configs/r2/base.yaml --frames /workspace/frames/avenue --dataset avenue \
  --state $O/state/avenue --grid-state $O/grid/state_avenue --grid-run $CK/runs/avenue_2d_noadv \
  --gt /workspace/cache/avenue/gt.json --out $O/stream/avenue > $L/stream_avenue.log 2>&1
stamp "stream done"

nvidia-smi > $O/bench/nvidia_smi_start.txt 2>&1; uptime >> $O/bench/nvidia_smi_start.txt
for ds in avenue ped2; do
  for be in trt pt; do
    i=0
    for mode in obj grid grid obj obj grid; do
      i=$((i+1))
      G=""; [ $mode = grid ] && G="--grid-state $O/grid/state_$ds --grid-run $CK/runs/${ds}_2d_noadv --parallel-grid"
      python tools/bench_e2e.py --config $(cfg $be) --frames /workspace/frames/$ds --dataset $ds --state $O/state/$ds \
        --parallel-flow $G --out $O/bench/${ds}_${be}_${mode}_$i.json > $L/bench_${ds}_${be}_${mode}_$i.log 2>&1
    done
  done
done
nvidia-smi > $O/bench/nvidia_smi_end.txt 2>&1; uptime >> $O/bench/nvidia_smi_end.txt
cp $L/*.log $O/ 2>/dev/null
stamp ALL_DONE
touch $L/ALL_DONE
