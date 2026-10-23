#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/home/artyom-step/workspace/nlp-hw2/artifacts/a100_full_eval_20261023
TASK_FULL=/home/artyom-step/workspace/nlp-hw2/artifacts/a100_full_20261023
TASK_OLD=/home/artyom-step/workspace/nlp-hw2/artifacts/a100_sequential_20261022
exec 9>"$TASK_ROOT/evaluation.lock"
flock -n 9
test ! -e "$TASK_ROOT/evaluation.pid"
trap 'TASK_RESULT=$?; printf "%s\n" "$TASK_RESULT" > "$TASK_ROOT/evaluation.exitcode"; if [ "$TASK_RESULT" -ne 0 ]; then printf "failed\n" > "$TASK_ROOT/evaluation.phase"; fi' EXIT
printf '%s\n' "$$" > "$TASK_ROOT/evaluation.pid"
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE MASTER_ADDR MASTER_PORT ACCELERATE_USE_CPU ACCELERATE_MIXED_PRECISION
export MPLBACKEND=Agg TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$TASK_ROOT/code"
cd "$TASK_ROOT/code"
python3 - <<'PY'
import torch,transformers,datasets,accelerate,shutil,json,hashlib
from pathlib import Path
assert torch.__version__=='2.9.1+cu128'
assert transformers.__version__=='5.10.2'
assert datasets.__version__=='4.6.0'
assert accelerate.__version__=='1.15.0'
assert shutil.disk_usage('.').free>12*1024**3
assert torch.cuda.is_available()
for name,expected in json.loads(Path('../evaluation_source_manifest.json').read_text())['files'].items():
 assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
print('Source and pinned environment verified; GPU:',torch.cuda.get_device_name(0),flush=True)
PY
printf 'evaluating\n' > "$TASK_ROOT/evaluation.phase"
python3 tools/evaluate_fixed_bundle.py --bundle-root "$TASK_OLD/bundle" --checkpoint "$TASK_FULL/training/trial/final" --output "$TASK_ROOT/evaluation" --architecture bart --batch-size 64 --num-beams 4 --max-new-tokens 256 --length-penalty 1.0 --training-bundle-root "$TASK_FULL/bundle" --expected-training-manifest-sha 0f330f228599f4cff65ed6ce0ca11e2d667ea24e6b911b5f67f6fff7e24bcbe5 --reference-identity "$TASK_ROOT/code/reference_identity.json" > "$TASK_ROOT/logs/evaluation.log" 2>&1 &
TASK_DRIVER=$!
printf '%s\n' "$TASK_DRIVER" > "$TASK_ROOT/evaluation_driver.pid"
wait "$TASK_DRIVER"
printf 'complete\n' > "$TASK_ROOT/evaluation.phase"
