#!/usr/bin/env bash
set -u
cd /workspace/nlp-hw2 || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export ACCELERATE_DYNAMO_USE_DYNAMIC=true PYTHONUNBUFFERED=1
root=artifacts/sweeps/hpo_006_t5
.venv/bin/python -u train_sweep.py --config configs/hpo_006_t5/sweep.json > "$root/sweep.log" 2>&1
code=$?
printf '%s\n' "$code" > "$root/sweep.exitcode"
exit "$code"
