"""Build the private Kaggle upload from verified data and current project code."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from kaggle_experiments import file_sha256, write_json


def build_bundle(backup_root, output_zip):
    backup_root,output_zip=Path(backup_root),Path(output_zip)
    original=json.loads((backup_root/'artifacts/exports/kaggle_20261019/kaggle_core.manifest.json').read_text())
    original={entry['path']:entry for entry in original['files']}
    prepared=backup_root/'artifacts/sweeps/hpo_004_1m_1epoch'
    aliases={}
    for source in sorted((prepared/'data').rglob('*')):
        if source.is_file():
            aliases['data/'+source.relative_to(prepared/'data').as_posix()]=source
    for name in ['data_manifest.json','train_indices.npy','val_indices.npy']:
        aliases['selection/'+name]=prepared/name
    aliases['tokenizer.json']=backup_root/'artifacts/tokenizer.json'
    for source in aliases.values():
        entry=original[source.relative_to(backup_root).as_posix()]
        if source.stat().st_size!=entry['bytes'] or file_sha256(source)!=entry['sha256']:
            raise ValueError(f'Source integrity checksum mismatch: {source}')
    for name in ['training.py','utils.py','kaggle_experiments.py','train_kaggle.py',
        'plotting.py','tokenizer.py','requirements-kaggle.txt']:
        aliases[name]=PROJECT/name
    for source in sorted((PROJECT/'models').glob('*.py')):
        aliases['models/'+source.name]=source
    manifest={'schema':1,'created_at':datetime.now(timezone.utc).isoformat(),
        'selection_sha256':file_sha256(prepared/'data_manifest.json'),
        'files':[{'path':name,'bytes':source.stat().st_size,'sha256':file_sha256(source)}
            for name,source in sorted(aliases.items())]}
    manifest_bytes=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
    import hashlib
    manifest_sha=hashlib.sha256(manifest_bytes).hexdigest()
    output_zip.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output_zip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        z.writestr('bundle_manifest.json',manifest_bytes)
        for name,source in sorted(aliases.items()):
            z.write(source,name)
    result={'archive':str(output_zip.resolve()),'archive_bytes':output_zip.stat().st_size,
        'archive_sha256':file_sha256(output_zip),'bundle_manifest_sha256':manifest_sha,
        'selection_sha256':manifest['selection_sha256'],'file_count':len(aliases),
        'uncompressed_bytes':sum(entry['bytes'] for entry in manifest['files'])}
    write_json(output_zip.with_suffix('.json'),result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backup-root',type=Path,default=PROJECT/'artifacts/remote_backups/h100_20261019_kaggle/restored')
    p.add_argument('--output',type=Path,default=PROJECT/'artifacts/kaggle_20261019/nlp_hw2_kaggle.zip')
    args=p.parse_args()
    print(json.dumps(build_bundle(args.backup_root,args.output),ensure_ascii=False,indent=2))
