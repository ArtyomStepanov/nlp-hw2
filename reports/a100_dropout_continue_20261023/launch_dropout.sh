#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/home/artyom-step/workspace/nlp-hw2/artifacts/a100_dropout_continue_20261023
TASK_FULL=/home/artyom-step/workspace/nlp-hw2/artifacts/a100_full_20261023
exec 9>"$TASK_ROOT/pipeline.lock"
flock -n 9
test ! -e "$TASK_ROOT/dropout_pipeline.pid"
trap 'TASK_RESULT=$?; printf "%s\n" "$TASK_RESULT" > "$TASK_ROOT/dropout_pipeline.exitcode"; if [ "$TASK_RESULT" -ne 0 ]; then printf "failed\n" > "$TASK_ROOT/dropout_pipeline.phase"; fi' EXIT
printf '%s\n' "$$" > "$TASK_ROOT/dropout_pipeline.pid"
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE MASTER_ADDR MASTER_PORT ACCELERATE_USE_CPU ACCELERATE_MIXED_PRECISION
export MPLBACKEND=Agg TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$TASK_ROOT/code"
cd "$TASK_ROOT/code"
printf 'environment_verification\n' > "$TASK_ROOT/dropout_pipeline.phase"
python3 - <<'PY'
import torch,transformers,datasets,accelerate,shutil,json,hashlib
from pathlib import Path
assert torch.__version__=='2.9.1+cu128'
assert transformers.__version__=='5.10.2'
assert datasets.__version__=='4.6.0'
assert accelerate.__version__=='1.15.0'
assert shutil.disk_usage('.').free>12*1024**3
assert torch.cuda.is_available()
for name,expected in json.loads(Path('../source_manifest.json').read_text())['files'].items():
 assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
print('Source and pinned environment verified; GPU:',torch.cuda.get_device_name(0),flush=True)
PY
printf 'preflight\n' > "$TASK_ROOT/dropout_pipeline.phase"
python3 tools/run_a100_dropout.py --bundle-root "$TASK_FULL/bundle" --parent-checkpoint "$TASK_FULL/training/trial/checkpoint-180000" --output-dir "$TASK_ROOT/preflight" --preflight > "$TASK_ROOT/logs/preflight.log" 2>&1
python3 - <<'PY'
import json,math
from pathlib import Path
root=Path('/home/artyom-step/workspace/nlp-hw2/artifacts/a100_dropout_continue_20261023')
result=json.loads((root/'preflight/trial/complete.json').read_text())
assert result['global_step']>=2
assert math.isfinite(result['metrics']['eval_token_loss'])
assert result['peak_allocated_gib']<70
assert result['effective_batch']==256
print('GPU preflight passed:',result['metrics']['eval_token_loss'],result['peak_allocated_gib'],flush=True)
PY
printf 'training\n' > "$TASK_ROOT/dropout_pipeline.phase"
python3 tools/run_a100_dropout.py --bundle-root "$TASK_FULL/bundle" --parent-checkpoint "$TASK_FULL/training/trial/checkpoint-180000" --output-dir "$TASK_ROOT/training" > "$TASK_ROOT/logs/training.log" 2>&1 &
TASK_DRIVER=$!
printf '%s\n' "$TASK_DRIVER" > "$TASK_ROOT/dropout_driver.pid"
wait "$TASK_DRIVER"
printf 'complete\n' > "$TASK_ROOT/dropout_pipeline.phase"
