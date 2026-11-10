#!/usr/bin/env bash
set -uo pipefail
cd /workspace/cctv_aean
export PYTHONPATH=.
python vad_bench/vast/runs_birlesik/stage2_runner.py > /workspace/logs/stage2_execution.log 2>&1
