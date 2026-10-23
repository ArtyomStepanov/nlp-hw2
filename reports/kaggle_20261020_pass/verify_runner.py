"""Small real checkpoint/generation/execution check before spending GPU time."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
import datasets as ds
import numpy as np
from tokenizer import train_bpe_tokenizer
from kaggle_experiments import file_sha256, KaggleConfig, run_experiment
from evaluate_saved import run

with tempfile.TemporaryDirectory(prefix='nlp-pass-smoke-') as directory:
    root=Path(directory)/'bundle'
    (root/'selection').mkdir(parents=True)
    codes=['def solve():\n    return 1\n','def solve():\n    return 2\n','return 3','def solve():\n    return 4\n']
    raw=ds.Dataset.from_dict({'question':['one','two','three','four'],'code':codes})
    tokenizer=train_bpe_tokenizer(raw,vocab_size=280,min_frequency=1)
    tokenizer.save(str(root/'tokenizer.json'))
    def encoded(count):
        return ds.Dataset.from_list([dict(input_ids=tokenizer.encode(raw[i%4]['question']).ids,
            attention_mask=[1]*len(tokenizer.encode(raw[i%4]['question']).ids),
            labels=tokenizer.encode(raw[i%4]['code']).ids[1:],
            length=len(tokenizer.encode(raw[i%4]['code']).ids)) for i in range(count)])
    ds.DatasetDict({'train':encoded(8),'val':encoded(4)}).save_to_disk(str(root/'data'))
    details={}
    for name,count in [('train',8),('val',4)]:
        path=root/'selection'/f'{name}_indices.npy'
        np.save(path,np.arange(count))
        details[name]={'indices_sha256':file_sha256(path)}
    selection={'identity':{'train_rows':8,'validation_rows':4,'seed':42,
        'tokenizer_sha256':file_sha256(root/'tokenizer.json')},'splits':details}
    (root/'selection/data_manifest.json').write_text(json.dumps(selection))
    files=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=file_sha256(p))
        for p in sorted(root.rglob('*')) if p.is_file()]
    (root/'bundle_manifest.json').write_text(json.dumps(dict(schema=1,files=files,
        selection_sha256=file_sha256(root/'selection/data_manifest.json'))))
    digests=[]
    for architecture in ['copy','t5']:
        output=Path(directory)/architecture
        run_experiment(root,KaggleConfig(architecture=architecture,output_dir=output,
            micro_batch=2,global_batch=2,use_cpu=True,precision='no',tiny_model=True,
            expected_rows=(8,4),max_steps=4,eval_steps=10,gradient_checkpointing=False))
        metrics=run(root,output/'final',Path(directory)/f'eval-{architecture}',architecture=architecture,
            device='cpu',batch_size=2,num_beams=4,max_new_tokens=8,expected_rows=(8,4),
            expected_manifest_sha=file_sha256(root/'bundle_manifest.json'))
        assert metrics['num_examples']==4
        assert metrics['num_evaluated']==3, metrics
        assert metrics['num_reference_errors']==1
        assert metrics['num_beams']==4
        digests.append(metrics['validation_sha256'])
        print('VERIFIED',architecture,metrics['num_evaluated'],metrics['pass@1'])
    assert digests[0]==digests[1]
print('Both native HF checkpoints, beam4, identical rows/order, reference-error denominator: verified.')
