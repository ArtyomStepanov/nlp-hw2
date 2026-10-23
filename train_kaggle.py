"""Run from a Kaggle notebook via torchrun; GPU workers import only project code."""
import argparse
import json

from kaggle_experiments import KaggleConfig, run_experiment


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle-root',required=True)
    p.add_argument('--architecture',choices=['t5','copy','bart'],required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--micro-batch',type=int,default=16)
    p.add_argument('--global-batch',type=int,default=256)
    p.add_argument('--learning-rate',type=float,default=2e-4)
    p.add_argument('--eval-steps',type=int,default=1000)
    p.add_argument('--precision',choices=['auto','fp16','bf16','no'],default='auto')
    p.add_argument('--torch-compile',action='store_true')
    p.add_argument('--no-gradient-checkpointing',action='store_true')
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--tiny-model',action='store_true',help='Only for local software smoke checks, not model comparison.')
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--expected-manifest-sha')
    p.add_argument('--resume-from')
    args=p.parse_args()
    config=KaggleConfig(architecture=args.architecture,output_dir=args.output_dir,
        micro_batch=args.micro_batch,global_batch=args.global_batch,learning_rate=args.learning_rate,
        eval_steps=args.eval_steps,precision=args.precision,torch_compile=args.torch_compile,
        gradient_checkpointing=not args.no_gradient_checkpointing,preflight=args.preflight,
        tiny_model=args.tiny_model,use_cpu=args.cpu,expected_manifest_sha=args.expected_manifest_sha)
    result=run_experiment(args.bundle_root,config,resume_from_checkpoint=args.resume_from)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
