#!/usr/bin/env bash
# Main grid: dataset x variant (AEAN), plus no-adversarial ablation. Logs -> runs/<ped>_<variant>[_noadv]/
set -uo pipefail
cd "$(dirname "$0")"
STEPS=${STEPS:-20000}
for ped in ${PEDS:-ped2 ped1}; do
  for v in ${VARIANTS:-2d 3d 1d}; do
    for adv in ${ADVS:-adv noadv}; do
      name=${ped}_${v}$([ "$adv" = noadv ] && echo _noadv)
      flag=$([ "$adv" = noadv ] && echo --no-adv)
      echo "=== $name ===" 
      python train.py --data /workspace/cache/$ped --variant $v --steps $STEPS $flag --out runs/$name > runs_$name.log 2>&1 \
        && python evaluate.py --data /workspace/cache/$ped --run runs/$name >> runs_$name.log 2>&1
      tail -1 runs_$name.log
    done
  done
done
