#!/bin/bash
cd /home/artyom-step/workspace/nlp-hw2 || exit 1
/usr/bin/python3 -u /home/artyom-step/workspace/nlp-hw2/artifacts/evaluations/transformer_002_checkpoint90000_test10000_seed42_20261013/run_evaluation.py --output /home/artyom-step/workspace/nlp-hw2/artifacts/evaluations/transformer_002_checkpoint90000_test10000_seed42_20261013 > /home/artyom-step/workspace/nlp-hw2/artifacts/launches/evaluation_002_checkpoint90000_20261013/run.log 2>&1
code=$?
printf "%s\n" "$code" > /home/artyom-step/workspace/nlp-hw2/artifacts/launches/evaluation_002_checkpoint90000_20261013/run.exitcode
exit "$code"
