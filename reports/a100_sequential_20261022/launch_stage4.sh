#!/usr/bin/env bash
set -euo pipefail
root=/home/artyom-step/workspace/nlp-hw2/artifacts/a100_sequential_20261022
cd "$root/stage2_recovery_code"
export MPLBACKEND=Agg TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
printf '%s\n' "$$" > "$root/stage4_pipeline.pid"
set +e
python3 /home/artyom-step/workspace/nlp-hw2/artifacts/a100_sequential_20261022/stage2_recovery_code/tools/run_a100_decoding.py --bundle-root /home/artyom-step/workspace/nlp-hw2/artifacts/a100_sequential_20261022/stage2_data/distinct4m --checkpoint /home/artyom-step/workspace/nlp-hw2/artifacts/a100_sequential_20261022/stage2_recovery/trials/distinct4m/final --architecture bart --output-dir /home/artyom-step/workspace/nlp-hw2/artifacts/a100_sequential_20261022/stage4_decoding --reference-identity /home/artyom-step/workspace/nlp-hw2/artifacts/a100_sequential_20261022/stage2_recovery_code/reference_identity.json > "$root/logs/stage4_decoding.log" 2>&1 &
child=$!
printf '%s\n' "$child" > "$root/stage4_driver.pid"
wait "$child"
result=$?
printf '%s\n' "$result" > "$root/stage4_pipeline.exitcode"
exit "$result"
