#!/bin/bash
set -e
cd /workspace/nlp-hw2
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ACCELERATE_DYNAMO_USE_DYNAMIC=true
exec /workspace/nlp-hw2/.venv/bin/python -u /workspace/nlp-hw2/train_distributed.py --dataset /workspace/nlp-hw2/artifacts/datasets/tinysms_98_1_1_seed42 --tokenizer /workspace/nlp-hw2/artifacts/tokenizer.json --model-config /workspace/nlp-hw2/configs/transformer_003_10x10_d1024_h100/model_config.json --training-config /workspace/nlp-hw2/configs/transformer_003_10x10_d1024_h100/training_config.json --resume-from-checkpoint /workspace/nlp-hw2/artifacts/experiments/transformer_003_10x10_d1024_h100/checkpoint-15000
