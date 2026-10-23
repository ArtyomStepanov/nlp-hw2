#!/bin/bash
cd /home/artyom-step/workspace/nlp-hw2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
/usr/bin/python3 -u -m torch.distributed.run --master-addr=127.0.0.1 --master-port=52311 --nproc-per-node=2 /home/artyom-step/workspace/nlp-hw2/train_distributed.py --dataset /home/artyom-step/workspace/nlp-hw2/artifacts/datasets/tinysms_98_1_1_seed42 --tokenizer /home/artyom-step/workspace/nlp-hw2/artifacts/experiments/transformer_002_8x8_d768_lr1e4/tokenizer.json --model-config /home/artyom-step/workspace/nlp-hw2/artifacts/experiments/transformer_002_8x8_d768_lr1e4/model_config.json --training-config /home/artyom-step/workspace/nlp-hw2/artifacts/experiments/transformer_002_8x8_d768_lr1e4/training_config.json --resume-from-checkpoint /home/artyom-step/workspace/nlp-hw2/artifacts/experiments/transformer_002_8x8_d768_lr1e4/checkpoint-50000 > /home/artyom-step/workspace/nlp-hw2/artifacts/launches/transformer_002_8x8_d768_lr1e4/train.log 2>&1
status=$?
printf "%s\n" "$status" > /home/artyom-step/workspace/nlp-hw2/artifacts/launches/transformer_002_8x8_d768_lr1e4/train.exitcode
exit "$status"
