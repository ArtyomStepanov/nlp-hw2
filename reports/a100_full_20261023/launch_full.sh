#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/home/artyom-step/workspace/nlp-hw2/artifacts/a100_full_20261023
TASK_OLD=/home/artyom-step/workspace/nlp-hw2/artifacts/a100_sequential_20261022
TASK_CACHE=/home/artyom-step/workspace/nlp-hw2/artifacts/experiments/transformer_002_8x8_d768_lr1e4/preprocessed/59ebc525571791733f851050d49d4a25a2ba5330ff775d360d868ec0bc6f35e0/dataset
exec 9>"$TASK_ROOT/pipeline.lock"
flock -n 9
test ! -e "$TASK_ROOT/full_pipeline.pid"
trap 'TASK_RESULT=$?; printf "%s\n" "$TASK_RESULT" > "$TASK_ROOT/full_pipeline.exitcode"; if [ "$TASK_RESULT" -ne 0 ]; then printf "failed\n" > "$TASK_ROOT/full_pipeline.phase"; fi' EXIT
printf '%s\n' "$$" > "$TASK_ROOT/full_pipeline.pid"
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE MASTER_ADDR MASTER_PORT ACCELERATE_USE_CPU ACCELERATE_MIXED_PRECISION
export MPLBACKEND=Agg TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$TASK_ROOT/code"
cd "$TASK_ROOT/code"
python3 - <<'PY'
import torch, transformers, datasets, accelerate, shutil
assert transformers.__version__ == '5.10.2', transformers.__version__
assert datasets.__version__ == '4.6.0', datasets.__version__
assert accelerate.__version__ == '1.15.0', accelerate.__version__
assert torch.__version__ == '2.9.1+cu128', torch.__version__
assert shutil.disk_usage('.').free > 12 * 1024**3, 'Less than 12 GiB disk free'
print('Pinned dependency versions verified', flush=True)
PY
printf 'preparing_data\n' > "$TASK_ROOT/full_pipeline.phase"
python3 tools/prepare_a100_full.py --full-cache "$TASK_CACHE" --original-bundle "$TASK_OLD/bundle" --stage2-data-dir "$TASK_OLD/stage2_data" --output-dir "$TASK_ROOT/bundle" > "$TASK_ROOT/logs/data_prepare.log" 2>&1
printf 'waiting_stage4\n' > "$TASK_ROOT/full_pipeline.phase"
while [ ! -f "$TASK_OLD/stage4_pipeline.exitcode" ]; do
  sleep 10
done
test "$(cat "$TASK_OLD/stage4_pipeline.exitcode")" = 0
printf 'training\n' > "$TASK_ROOT/full_pipeline.phase"
python3 tools/run_a100_full.py --bundle-root "$TASK_ROOT/bundle" --output-dir "$TASK_ROOT/training" --stage2-dir "$TASK_OLD/stage2_recovery" --previous-stage4-dir "$TASK_OLD/stage4_decoding" > "$TASK_ROOT/logs/full_training.log" 2>&1 &
TASK_DRIVER=$!
printf '%s\n' "$TASK_DRIVER" > "$TASK_ROOT/full_driver.pid"
wait "$TASK_DRIVER"
printf 'complete\n' > "$TASK_ROOT/full_pipeline.phase"
