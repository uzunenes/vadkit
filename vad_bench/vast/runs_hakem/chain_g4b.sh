#!/usr/bin/env bash
# ON the G4 machine (setup.sh already done): wait for the copied features, then prep_g4 -> run_g4.
# /workspace/cache is a broken entry left by a failed `vastai copy`, so gt and features live under new paths.
export CACHE=/workspace/gtm2/cache FEAT=/workspace/featm2/base
cd /workspace/cctv_aean
until [ "$(find $FEAT -name '*.npz' 2>/dev/null | wc -l)" -eq 572 ] && [ -s $CACHE/shanghaitech/gt.json ]; do sleep 20; done
sleep 30
bash vad_bench/vast/runs_hakem/prep_g4.sh > /workspace/logs/prep_g4.log 2>&1 || { echo PREP_FAILED; exit 1; }
bash vad_bench/vast/runs_hakem/run_g4.sh > /workspace/logs/run_g4.log 2>&1
echo CHAIN_DONE
