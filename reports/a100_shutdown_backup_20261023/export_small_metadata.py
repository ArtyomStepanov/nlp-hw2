from pathlib import Path
import json,shlex,subprocess,tarfile
root=Path('reports/a100_shutdown_backup_20261023')
backup=Path('artifacts/a100_shutdown_backup_20261023')
selected=json.loads((root/'backup_selection.json').read_text())['selected']
paths=[e['path']for e in selected if e['bytes']<=100_000_000]
remote_script=r'''
from pathlib import Path
import gzip,json,sys,tarfile
root=Path('/home/artyom-step/workspace/nlp-hw2')
paths=json.load(sys.stdin)
with gzip.GzipFile(fileobj=sys.stdout.buffer,mode='wb',compresslevel=1)as compressed, tarfile.open(fileobj=compressed,mode='w|')as archive:
 for name in paths:
  p=root/name
  if p.is_symlink()or not p.is_file()or not p.resolve().is_relative_to(root):raise ValueError('Unsafe path '+name)
  archive.add(p,arcname=name,recursive=False)
'''
archive=backup/'small_metadata.tar.gz'
partial=archive.with_suffix('.gz.partial')
with partial.open('xb')as stream:
 subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','nlp-bart.ml-contest-mla.ws.deep.yandex.net','python3 -c '+shlex.quote(remote_script)],input=json.dumps(paths).encode(),stdout=stream,check=True)
partial.rename(archive)
restored=backup/'restored'
restored.mkdir(exist_ok=True)
with tarfile.open(archive,'r:gz')as source:
 for entry in source.getmembers():
  if not(entry.isfile()or entry.islnk())or not(restored/entry.name).resolve().is_relative_to(restored.resolve()):raise ValueError('Unsafe archive member')
  if entry.islnk()and not(restored/entry.linkname).resolve().is_relative_to(restored.resolve()):raise ValueError('Unsafe archive hardlink')
 source.extractall(restored,filter='data')
print(json.dumps(dict(status='small_metadata_downloaded',files=len(paths),archive_bytes=archive.stat().st_size,restored=str(restored)),indent=2))
