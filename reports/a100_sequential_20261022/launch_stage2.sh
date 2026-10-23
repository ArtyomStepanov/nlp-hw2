#!/usr/bin/env bash
# Execute only the already-authorized repetition/diversity comparison.
set -euo pipefail
TASK_ROOT=${1:?run root required}
TASK_CODE="$TASK_ROOT/stage2_code"
TASK_FULL_CACHE=${2:?full tokenized cache required}
exec 9>"$TASK_ROOT/stage2_pipeline.lock"
flock -n 9
trap 'result=$?; printf "%s\n" "$result" > "$TASK_ROOT/stage2_pipeline.exitcode"; if [ "$result" -ne 0 ]; then printf "failed\n" > "$TASK_ROOT/stage2_pipeline.phase"; fi' EXIT
printf '%s\n' "$$" > "$TASK_ROOT/stage2_pipeline.pid"
export MPLBACKEND=Agg TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export PYTHONPATH="$TASK_CODE"
cd "$TASK_CODE"
python3 -c 'import sys; from pathlib import Path; from tools.run_a100_stage2 import verify_stage1; print(verify_stage1(Path(sys.argv[1])/"stage1",Path(sys.argv[1])/"stage2_code/reference_identity.json"))' "$TASK_ROOT"
printf 'preparing_data\n' > "$TASK_ROOT/stage2_pipeline.phase"
python3 tools/prepare_a100_diversity.py \
  --bundle-root "$TASK_ROOT/bundle" \
  --full-cache "$TASK_FULL_CACHE" \
  --output-dir "$TASK_ROOT/stage2_data" \
  --expected-bundle-manifest-sha 7480c7d1778bffce50db987019f2ccf87c06f8643c1ce59ad811e4fd3c77d0e1 \
  > "$TASK_ROOT/logs/stage2_data_prepare.log" 2>&1
printf 'experiments\n' > "$TASK_ROOT/stage2_pipeline.phase"
python3 tools/run_a100_stage2.py \
  --stage1-dir "$TASK_ROOT/stage1" \
  --prepared-data-dir "$TASK_ROOT/stage2_data" \
  --output-dir "$TASK_ROOT/stage2" \
  --reference-identity "$TASK_CODE/reference_identity.json" \
  > "$TASK_ROOT/logs/stage2_driver.log" 2>&1
printf 'complete\n' > "$TASK_ROOT/stage2_pipeline.phase"
