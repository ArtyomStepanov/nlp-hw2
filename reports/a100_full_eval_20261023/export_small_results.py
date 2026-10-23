"""Export only completed evaluation records, never model weights or datasets."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import tarfile

root = Path('/home/artyom-step/workspace/nlp-hw2/artifacts/a100_full_eval_20261023')
assert (root/'evaluation.exitcode').read_text().strip() == '0'
assert json.loads((root/'evaluation/progress.json').read_text())['phase'] == 'complete'
assert json.loads((root/'evaluation/metrics.json').read_text())['reference_identity_verified'] is True
paths = [p for folder in (root/'evaluation', root/'logs') for p in sorted(folder.rglob('*'))
         if p.is_file() and p.suffix in ('.json', '.jsonl', '.log')]
paths += [root/name for name in ('evaluation_source_manifest.json', 'evaluation.pid',
          'evaluation_driver.pid', 'evaluation.phase', 'evaluation.exitcode', 'launch_evaluation.sh')]
assert all(p.stat().st_size < 100*1024**2 for p in paths)
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
receipt = dict(created_at=datetime.now(timezone.utc).isoformat(), files={
    p.relative_to(root).as_posix(): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in paths})
manifest = root/'results_manifest.json'
assert not manifest.exists(), 'Refusing to overwrite an earlier export'
manifest.write_text(json.dumps(receipt, indent=2)+'\n')
archive = root/'evaluation_results.tar.gz'
with tarfile.open(archive, 'x:gz') as output:
    for path in [*paths, manifest]:
        output.add(path, arcname=path.relative_to(root))
print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size, sha256=sha(archive),
                     files=len(paths), manifest_sha256=sha(manifest))))
