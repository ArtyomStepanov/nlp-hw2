"""Run the authorized, matched BART/Copy comparison sequentially on one A100."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kaggle_experiments import write_json


def verify_reference_results(path, expected):
    fields = ('sample_row','evaluable','reference_status','reference_result_json')
    with Path(path).open() as stream:
        rows = [{key:row[key] for key in fields} for row in map(json.loads,stream)]
    if len(rows) != len(expected):
        raise ValueError('reference mismatch: number of rows changed')
    for actual, reference in zip(rows,expected):
        if actual != reference:
            raise ValueError(f'reference mismatch at sample_row={reference["sample_row"]}: {actual}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle-root', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--micro-batch', type=int, default=128)
    args = p.parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'status':'running', 'stage':1, 'driver_pid':os.getpid(),
        'started_at':datetime.now(timezone.utc).isoformat(), 'completed':[]}
    env = os.environ | {'MPLBACKEND':'Agg','TOKENIZERS_PARALLELISM':'false',
        'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONUNBUFFERED':'1',
        'PYTHONPATH':str(ROOT),'CUDA_VISIBLE_DEVICES':'0'}

    def invoke(architecture, phase, command):
        state.update(architecture=architecture, phase=phase,
            updated_at=datetime.now(timezone.utc).isoformat())
        log = out/'logs'/f'{architecture}_{phase}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT)
            state.update(child_pid=process.pid, log=str(log))
            write_json(out/'queue_state.json',state)
            print(f'{architecture} {phase}: pid={process.pid}, log={log}',flush=True)
            code = process.wait()
        if code:
            raise RuntimeError(f'{architecture} {phase} failed with exit code {code}; see {log}')

    def train_command(architecture, trial):
        return [sys.executable,str(ROOT/'train_kaggle.py'),
            '--bundle-root',args.bundle_root,'--architecture',architecture,
            '--output-dir',str(trial),'--micro-batch',str(args.micro_batch),
            '--global-batch','256','--learning-rate','0.0002','--eval-steps','1000',
            '--precision','fp16','--no-gradient-checkpointing',
            '--expected-manifest-sha','7480c7d1778bffce50db987019f2ccf87c06f8643c1ce59ad811e4fd3c77d0e1']

    try:
        for architecture in ('bart','copy'):
            trial = out/'preflight'/architecture
            if not (trial/'complete.json').exists():
                invoke(architecture,'preflight',train_command(architecture,trial)+['--preflight'])
            result = json.loads((trial/'complete.json').read_text())
            if result['peak_allocated_gib'] > 70:
                raise RuntimeError(f'{architecture}: preflight leaves insufficient GPU memory headroom')
        for architecture in ('bart','copy'):
            trial = out/'trials'/architecture
            evaluation = out/'evaluations'/architecture
            if not (trial/'complete.json').exists():
                invoke(architecture,'training',train_command(architecture,trial))
            if not (evaluation/'metrics.json').exists():
                invoke(architecture,'evaluation',[sys.executable,str(ROOT/'tools/evaluate_fixed_bundle.py'),
                    '--bundle-root',args.bundle_root,'--checkpoint',str(trial/'final'),
                    '--output',str(evaluation),'--architecture',architecture,
                    '--batch-size','64','--num-beams','4','--max-new-tokens','256'])
            state['completed'].append(architecture)
            write_json(out/'queue_state.json',state)
        metrics = {name:json.loads((out/'evaluations'/name/'metrics.json').read_text())
            for name in ('bart','copy')}
        references = json.loads((ROOT/'reference_identity.json').read_text())
        for name, row in metrics.items():
            if row['validation_sha256'] != '282001d478bffb2d13ab2ca41e8232a0c99c3623f898dd7077128c266172c406':
                raise ValueError(f'{name}: validation identity changed')
            if row['num_evaluated'] != 19886:
                raise ValueError(f'{name}: reference eligibility changed; investigate before comparison')
            verify_reference_results(out/'evaluations'/name/'evaluation/evaluation.jsonl', references['rows'])
        winner = 'copy' if metrics['copy']['pass@1'] > metrics['bart']['pass@1'] + .005 else 'bart'
        write_json(out/'stage1_complete.json',{'models':metrics,'winner':winner,
            'selection_rule':'Prefer simpler BART unless Copy leads by >0.5 percentage points.',
            'finished_at':datetime.now(timezone.utc).isoformat()})
        state.update(status='awaiting_stage2',phase='complete',child_pid=None,winner=winner)
        write_json(out/'queue_state.json',state)
    except BaseException as error:
        state.update(status='failed',error=str(error),child_pid=None,
            updated_at=datetime.now(timezone.utc).isoformat())
        write_json(out/'queue_state.json',state)
        raise


if __name__ == '__main__':
    main()
