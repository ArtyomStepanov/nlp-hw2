"""Execute downloaded predictions using the existing evaluator, after row checks."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import datasets as ds
from tokenizers import Tokenizer
from evaluation import EvaluationConfig, evaluate_code

p=argparse.ArgumentParser()
p.add_argument('architecture',choices=['copy','t5'])
args=p.parse_args()
out=ROOT/'artifacts/evaluations/kaggle_pass_20261020'/args.architecture
rows=[json.loads(line) for line in (out/'predictions.jsonl').read_text().splitlines()]
assert len(rows)==20000
assert [row['sample_row'] for row in rows]==list(range(20000))
bundle=ROOT/'artifacts/kaggle_20261019/verified_bundle'
data=ds.load_from_disk(str(bundle/'data/val'))
tokenizer=Tokenizer.from_file(str(bundle/'tokenizer.json'))
digest=hashlib.sha256()
for i,(row, original) in enumerate(zip(rows,data)):
    assert row['question']==tokenizer.decode(original['input_ids'],skip_special_tokens=True),i
    assert row['code']==tokenizer.decode(original['labels'],skip_special_tokens=True),i
    digest.update((json.dumps(original,sort_keys=True,separators=(',',':'))+'\n').encode())
print('Verified all 20,000 input/reference pairs and original order.',flush=True)
started=time.perf_counter()
result=evaluate_code(ds.Dataset.from_list(rows),EvaluationConfig(workers=8,
    timeout_seconds=2,relative_tolerance=1e-6,absolute_tolerance=1e-6,
    output_dir=out/'evaluation'))
metrics=dict(result.metrics,architecture=args.architecture,num_beams=4,max_new_tokens=256,
    validation_sha256=digest.hexdigest(),evaluation_split='validation',
    num_reference_errors=20000-result.metrics['num_evaluated'],
    num_hit_token_limit=sum(row['hit_token_limit'] for row in rows),
    mean_generated_tokens=sum(row['generated_tokens'] for row in rows)/len(rows),
    scoring_environment=dict(python=sys.version,platform=platform.platform()),
    evaluator_sha256=hashlib.sha256((ROOT/'evaluation.py').read_bytes()).hexdigest(),
    code_runner_sha256=hashlib.sha256((ROOT/'_code_runner.py').read_bytes()).hexdigest(),
    wall_seconds=time.perf_counter()-started)
(out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
report=ROOT/'reports/kaggle_20261020_pass'/f'{args.architecture}_local_metrics.json'
report.write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(metrics,ensure_ascii=False,indent=2),flush=True)
