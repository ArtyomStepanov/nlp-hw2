"""Verify the selected off-machine backup locally, without loading model tensors."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path

REPORT=Path(__file__).resolve().parent
PROJECT=REPORT.parents[1]
RESTORED=PROJECT/'artifacts/a100_shutdown_backup_20261023/restored'


def digest(path,cache):
 stat=path.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
 if key not in cache:
  h=hashlib.sha256()
  with path.open('rb')as stream:
   for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
  cache[key]=h.hexdigest()
 return cache[key]


def main():
 source=json.loads((REPORT/'remote_sha256_manifest.json').read_text())
 raw=json.loads((REPORT/'raw_source_backup_receipt.json').read_text())
 local=json.loads((REPORT/'local_source_snapshot.json').read_text())
 remote_final=json.loads((REPORT/'remote_snapshot_final_check.json').read_text())
 assert remote_final['status']=='unchanged'and not remote_final['changed']
 expected=[dict(entry,local_path=entry['path'])for entry in source['files']]
 expected += [dict(entry,local_path=entry['path'])for entry in raw['files']]
 expected += [dict(entry,local_path='local_source/'+entry['path'])for entry in local['files']]
 cache={};failures=[]
 for entry in expected:
  path=RESTORED/entry['local_path']
  if not path.is_file()or path.is_symlink():failures.append(dict(path=entry['local_path'],error='missing or not a regular file'));continue
  if path.stat().st_size!=entry['bytes']:failures.append(dict(path=entry['local_path'],error='size mismatch'));continue
  if digest(path,cache)!=entry['sha256']:failures.append(dict(path=entry['local_path'],error='SHA256 mismatch'))
 if failures:
  print(json.dumps(dict(status='incomplete_or_invalid',files_checked=len(expected),failures=failures),indent=2))
  raise SystemExit(1)
 required=['model.safetensors','optimizer.pt','scheduler.pt','scaler.pt','rng_state.pth','trainer_state.json','run_manifest.json','config.json','tokenizer.json']
 checkpoints=[]
 for step in (180000,181144):
  relative=f'artifacts/a100_full_20261023/training/trial/checkpoint-{step}'
  path=RESTORED/relative
  for name in required:
   assert(path/name).is_file()and(path/name).stat().st_size>0,(step,name)
  state=json.loads((path/'trainer_state.json').read_text())
  assert state['global_step']==step
  manifest=json.loads((path/'run_manifest.json').read_text())
  assert manifest['precision']=='fp16'and manifest['config']['architecture']=='bart'
  assert(manifest['train_rows'],manifest['validation_rows'])==(11593060,118302)
  checkpoints.append(dict(path=relative,global_step=step,all_required_exact_state_files_verified=True))
 fullroot=RESTORED/'artifacts/a100_full_20261023/training/trial'
 assert digest(fullroot/'final/model.safetensors',cache)==digest(fullroot/'checkpoint-180000/model.safetensors',cache)=='2059268173416c77cd8c901c791eb86e7cbcea0ede756b2bb830ec23638b8165'
 assert digest(fullroot/'checkpoint-181144/model.safetensors',cache)=='1a0a46502e93933c0beabd1224ab7ebef050f32ab9a5401f7e5a250ba96c8602'
 partials=[p.relative_to(RESTORED).as_posix()for p in RESTORED.rglob('*.partial')]
 assert not partials,partials
 physical={}
 for path in RESTORED.rglob('*'):
  if path.is_file():
   stat=path.stat();physical[(stat.st_dev,stat.st_ino)]=stat.st_size
 result=dict(schema=1,status='verified',verified_at=datetime.now(timezone.utc).isoformat(),
  restored_root=str(RESTORED),repository_revision=json.loads((REPORT/'restore_git_revision.json').read_text())['repository_revision'],remote_files_verified=len(source['files']),raw_files_verified=len(raw['files']),
  local_source_files_verified=len(local['files']),total_files_verified=len(expected),unique_hashed_inodes=len(cache),
  selected_remote_logical_bytes=sum(e['bytes']for e in source['files']),raw_logical_bytes=sum(e['bytes']for e in raw['files']),
  restored_unique_inode_bytes=sum(physical.values()),full_checkpoints=checkpoints,
  final_matches_best=True,latest_distinct_from_best=True,missing_or_corrupt_files=0,
  scope='Every selected source byte verified using fresh remote source SHA256 and local SHA256. Raw inputs and source snapshot also verified. No model tensor loading or GPU use.',
  derived_arrow_caches_excluded=True,restoration_limit='Derived caches must be rebuilt with recorded semantics and reviewed provenance before strict pinned-manifest training continuation.',
  remote_source_final_check=remote_final,remote_stop_command_issued=False,safe_to_stop_a100=False,
  shutdown_limit='A separately authorized dropout continuation is running; its new outputs are outside this immutable-snapshot backup. Do not stop that training because this backup completes.')
 (REPORT/'backup_complete.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))


if __name__=='__main__':main()
