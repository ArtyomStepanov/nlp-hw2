"""Actually unpack the upload archive and consume it through the runtime loader."""
import importlib.util
import json
from pathlib import Path
import shutil
import zipfile

import pytest

from test_kaggle_experiments import bundle, digest


def test_upload_bundle_round_trip_and_source_integrity(bundle,tmp_path):
    assert importlib.util.find_spec('tools.prepare_kaggle_bundle'), 'Upload bundler is missing'
    from tools.prepare_kaggle_bundle import build_bundle
    from kaggle_experiments import load_bundle
    backup=tmp_path/'backup'
    prepared=backup/'artifacts/sweeps/hpo_004_1m_1epoch'
    shutil.copytree(bundle/'data',prepared/'data')
    for p in (bundle/'selection').iterdir():
        shutil.copyfile(p,prepared/p.name)
    shutil.copyfile(bundle/'tokenizer.json',backup/'artifacts/tokenizer.json')
    original={'files':[{'path':str(p.relative_to(backup)),'bytes':p.stat().st_size,'sha256':digest(p)}
        for p in backup.rglob('*') if p.is_file()]}
    manifest=backup/'artifacts/exports/kaggle_20261019/kaggle_core.manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(original))
    output=tmp_path/'upload.zip'
    metadata=build_bundle(backup,output)
    restored=tmp_path/'uploaded'
    with zipfile.ZipFile(output) as z:
        z.extractall(restored)
    data,tok,identity=load_bundle(restored,expected_rows=(8,4),
        expected_manifest_sha=metadata['bundle_manifest_sha256'])
    assert len(data['train'])==8 and len(data['val'])==4
    assert (restored/'models/copy.py').is_file() and (restored/'train_kaggle.py').is_file()
    assert (restored/'requirements-kaggle.txt').is_file()
    (prepared/'data/train/state.json').write_text('corrupt source')
    with pytest.raises(ValueError,match='checksum|integrity'):
        build_bundle(backup,tmp_path/'bad.zip')
