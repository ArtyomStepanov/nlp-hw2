#!/bin/bash
cd /home/artyom-step/workspace/nlp-hw2 || exit 1
/usr/bin/python3 -u /home/artyom-step/workspace/nlp-hw2/artifacts/launches/evaluation_002_checkpoint360000_20261014/run_evaluation.py --output /home/artyom-step/workspace/nlp-hw2/artifacts/evaluations/transformer_002_checkpoint360000_test10000_seed42_20261014 > /home/artyom-step/workspace/nlp-hw2/artifacts/launches/evaluation_002_checkpoint360000_20261014/run.log 2>&1
code=$?
printf "%s\n" "$code" > /home/artyom-step/workspace/nlp-hw2/artifacts/launches/evaluation_002_checkpoint360000_20261014/run.exitcode
exit "$code"
