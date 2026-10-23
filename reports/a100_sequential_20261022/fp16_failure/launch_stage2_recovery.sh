#!/usr/bin/env bash
set -euo pipefail
root=/home/artyom-step/workspace/nlp-hw2/artifacts/a100_sequential_20261022
cd "$root/stage2_recovery_code"
export MPLBACKEND=Agg TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
printf '%s\n' "$$" > "$root/stage2_recovery_pipeline.pid"
printf '%s\n' running > "$root/stage2_recovery_pipeline.phase"
set +e
python3 /home/artyom-step/workspace/nlp-hw2/artifacts/a100_sequential_20261022/stage2_recovery_code/tools/recover_a100_stage2.py --stage1-dir /home/artyom-step/workspace/nlp-hw2/artifacts/a100_sequential_20261022/stage1 --failed-stage2-dir /home/artyom-step/workspace/nlp-hw2/artifacts/a100_sequential_20261022/stage2 --prepared-data-dir /home/artyom-step/workspace/nlp-hw2/artifacts/a100_sequential_20261022/stage2_data --output-dir /home/artyom-step/workspace/nlp-hw2/artifacts/a100_sequential_20261022/stage2_recovery --reference-identity /home/artyom-step/workspace/nlp-hw2/artifacts/a100_sequential_20261022/stage2_recovery_code/reference_identity.json > "$root/logs/stage2_recovery_driver.log" 2>&1
result=$?
set -e
printf '%s\n' "$result" > "$root/stage2_recovery_pipeline.exitcode"
if [ "$result" -eq 0 ]; then
    printf '%s\n' complete > "$root/stage2_recovery_pipeline.phase"
else
    printf '%s\n' failed > "$root/stage2_recovery_pipeline.phase"
fi
exit "$result"
