#!/usr/bin/env bash
# HB-POT stage 1: the single pre-registered test evaluation (docs/tez_notlari/hiyerarsik_kalibrasyon_on_kayit_2026-11-05.md
# §4-§5; gate committed in c91bfe4 before it), ON the vast machine, CPU only, at most 8 threads:
#   setsid nohup bash /workspace/hb/stage/run_hb.sh > /workspace/hb/test/run.log 2>&1 < /dev/null &
# 1. waits for R2_DONE (round-2 B3 job), 2. installs the staged committed files (git HEAD of vadkit/cli.py,
# vadkit/grid/precomputed.py, vadkit/plugins.py, and boot_hb.py) into the repo copy after checking their md5,
# 3. runs boot_hb.py per dataset (ShT first; any failure, e.g. the T0 reproduction check, stops the run),
# 4. combines the verdicts into /workspace/hb/test/results.json.
set -uo pipefail
STAGE=/workspace/hb/stage; O=/workspace/hb/test; REPO=/workspace/cctv_aean
mkdir -p $O
stamp() { echo "$(date +%T) $*"; }
while ! grep -q R2_DONE /workspace/r2/logs/run_b3.log 2>/dev/null; do sleep 30; done
stamp "R2_DONE seen; installing staged files"
( cd $STAGE/files && md5sum -c $STAGE/manifest.md5 ) || { stamp "staged files fail their md5"; exit 1; }
while read -r sum f; do
  mkdir -p "$(dirname $REPO/$f)"
  cp "$STAGE/files/$f" "$REPO/$f.hbtmp" && mv "$REPO/$f.hbtmp" "$REPO/$f"
done < $STAGE/manifest.md5
( cd $REPO && md5sum -c $STAGE/manifest.md5 ) || { stamp "installed files fail their md5"; exit 1; }
cd $REPO
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
C=/workspace/checkpoints; F=/workspace/r2/feat/base; I=/workspace/hb/refids
run() {
  local ds=$1; shift
  stamp "$ds start"
  nice -n 10 python vad_bench/vast/runs_hb/boot_hb.py --dataset $ds --features $F/$ds --gt /workspace/cache/$ds/gt.json \
    --out $O/results_$ds.json "$@" > $O/boot_$ds.log 2>&1 || { stamp "$ds FAILED (see $O/boot_$ds.log)"; exit 1; }
  stamp "$ds done"
}
run shanghaitech --config vadkit/configs/r2/base_sht.yaml --scenes \
  --extra-channels "$C/final_2026-10-07/runs/shtech_{sc}_2d_noadv/eval_shtech_{sc}_pool48_mc5/scores.npz:frx_bayes=I,flow_ubm_m=Fs,flow_rx_m=Fd" \
  --extra-ref-ids "$I/shanghaitech/{sc}.npz"
for ds in ped1 ped2 avenue; do
  run $ds --config vadkit/configs/r2/base.yaml \
    --extra-channels "$C/scores/runs/${ds}_2d_noadv/eval_${ds}_pool48_mc4/scores.npz:frx_bayes=I" \
                     "$C/scores/runs/${ds}_2d_noadv/eval_${ds}_pool48_mc5/scores.npz:flow_ubm_m=Fs,flow_rx_m=Fd" \
    --extra-ref-ids $I/$ds/mc4.npz $I/$ds/mc5.npz
done
python vad_bench/vast/runs_hb/boot_hb.py --combine $O/results_shanghaitech.json $O/results_ped1.json \
  $O/results_ped2.json $O/results_avenue.json --out $O/results.json > $O/combine.log 2>&1
stamp HB_DONE
