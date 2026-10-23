"""One authorized backup transfer; partial files support a later explicit rerun."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib,json,os,shlex,subprocess,threading,time
from pathlib import Path

REPORT=Path('reports/a100_shutdown_backup_20261023')
BACKUP=Path('artifacts/a100_shutdown_backup_20261023')/'restored'
HOST='nlp-bart.ml-contest-mla.ws.deep.yandex.net'
manifest=json.loads((REPORT/'remote_sha256_manifest.json').read_text())
entries=[e for e in manifest['files']if e['bytes']>100_000_000]

def digest(path):
 value=hashlib.sha256()
 with Path(path).open('rb')as stream:
  for chunk in iter(lambda:stream.read(8*1024*1024),b''):value.update(chunk)
 return value.hexdigest()

def verified(path,entry):
 return path.is_file()and path.stat().st_size==entry['bytes']and digest(path)==entry['sha256']

known=[Path('artifacts/a100_full_eval_20261023/t4_bundle/checkpoint/model.safetensors'),Path('artifacts/remote_backups/nlp-bart_20261019_161656/nlp-hw2/artifacts/experiments/transformer_002_8x8_d768_lr1e4/model.pt')]
reuse={}
for path in known:
 if path.is_file():reuse[(digest(path),path.stat().st_size)]=path.resolve()
initial_external_keys=set(reuse)
groups={}
for e in entries:groups.setdefault((e['sha256'],e['bytes']),[]).append(e)
receipts=[];pending=[]
for key,group in groups.items():
 source=reuse.get(key)
 if source is None:
  for e in group:
   path=BACKUP/e['path']
   if verified(path,e):source=path.resolve();break
 if source:
  for e in group:
   dest=BACKUP/e['path'];dest.parent.mkdir(parents=True,exist_ok=True)
   if not dest.exists():os.link(source,dest)
   elif not verified(dest,e):raise ValueError('Conflicting existing backup '+str(dest))
   receipts.append(dict(path=e['path'],bytes=e['bytes'],sha256=e['sha256'],method='reused_verified_local_hardlink',source=str(source)))
 else:pending.append(group)
pending.sort(key=lambda group:-group[0]['bytes'])
progress={};mutex=threading.Lock();stop=threading.Event();started=time.perf_counter()
total=sum(key[1]for key in groups if key not in initial_external_keys)
completed_before_restart=total-sum(group[0]['bytes']for group in pending)
unique_download_files=len(groups)-len(initial_external_keys)
for group in pending:
 entry=group[0];partial=(BACKUP/entry['path']).with_name(Path(entry['path']).name+'.partial')
 progress[entry['path']]=dict(status='pending',bytes_received=partial.stat().st_size if partial.exists()else 0,bytes_total=entry['bytes'])
initial_received=completed_before_restart+sum(value['bytes_received']for value in progress.values())

def snapshot(status='running'):
 with mutex:
  done=completed_before_restart+sum(v['bytes_received']for v in progress.values())
  active=[dict(path=k,**v)for k,v in progress.items()]
 elapsed=time.perf_counter()-started
 speed=(done-initial_received)/max(elapsed,1e-6)
 value=dict(status=status,updated_at=datetime.now(timezone.utc).isoformat(),unique_download_bytes=total,
  bytes_received=done,percent=100*done/max(total,1),elapsed_seconds=elapsed,
  throughput_mib_per_second=speed/2**20,eta_seconds=(total-done)/speed if speed else None,
  completed_files=unique_download_files-len(pending)+sum(v['status']=='verified'for v in active),unique_download_files=unique_download_files,active=active,
  reused_logical_bytes=sum(e['bytes']for e in entries if(e['sha256'],e['bytes'])in initial_external_keys),verified_existing_backup_bytes=completed_before_restart)
 temporary=REPORT/'transfer_progress.json.tmp'
 temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(REPORT/'transfer_progress.json')
 return value

def monitor():
 while not stop.wait(10):
  value=snapshot()
  print(json.dumps({k:value[k]for k in ('percent','bytes_received','throughput_mib_per_second','eta_seconds')}),flush=True)

remote_script=r'''
from pathlib import Path
import hashlib,json,sys
entry=json.loads(sys.argv[1]);offset=int(sys.argv[2]);root=Path('/home/artyom-step/workspace/nlp-hw2');p=root/entry['path']
if p.is_symlink()or not p.resolve().is_relative_to(root):raise ValueError('Unsafe source')
s=p.stat()
if(s.st_size,s.st_mtime_ns)!=(entry['bytes'],entry['mtime_ns']):raise ValueError('Source changed before transfer')
with p.open('rb')as stream:
 if offset:
  remaining=offset;prefix=hashlib.sha256()
  while remaining:
   block=stream.read(min(8*1024*1024,remaining))
   if not block:raise ValueError('Source prefix shorter than saved partial')
   prefix.update(block);remaining-=len(block)
  if prefix.hexdigest()!=sys.argv[3]:raise ValueError('Local partial prefix does not match pinned remote source')
 stream.seek(offset)
 while True:
  block=stream.read(8*1024*1024)
  if not block:break
  sys.stdout.buffer.write(block)
sys.stdout.buffer.flush();s=p.stat()
if(s.st_size,s.st_mtime_ns)!=(entry['bytes'],entry['mtime_ns']):raise ValueError('Source changed during transfer')
'''

def download(group):
 if stop.is_set():raise RuntimeError('Backup transfer cancelled after another failure')
 entry=group[0];dest=BACKUP/entry['path'];dest.parent.mkdir(parents=True,exist_ok=True)
 partial=dest.with_name(dest.name+'.partial')
 offset=partial.stat().st_size if partial.exists()else 0
 if offset>entry['bytes']:raise ValueError('Oversized local partial '+str(partial))
 prefix_sha=digest(partial)if offset else hashlib.sha256(b'').hexdigest()
 with mutex:progress[entry['path']]=dict(status='downloading',bytes_received=offset,bytes_total=entry['bytes'])
 command='python3 -c '+shlex.quote(remote_script)+' '+shlex.quote(json.dumps(entry,separators=(',',':')))+' '+str(offset)+' '+prefix_sha
 logs=BACKUP.parent/'transfer_logs';logs.mkdir(exist_ok=True)
 log_path=logs/(entry['sha256']+'.ssh.log');diagnostics=log_path.open('ab')
 process=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','-o','LogLevel=ERROR',HOST,command],stdout=subprocess.PIPE,stderr=diagnostics)
 try:
  with partial.open('ab')as stream:
   while True:
    if stop.is_set():raise RuntimeError('Transfer interrupted after another failure')
    block=process.stdout.read(8*1024*1024)
    if not block:break
    stream.write(block)
    with mutex:progress[entry['path']]['bytes_received']+=len(block)
  code=process.wait()
  if code:raise RuntimeError(f'SSH transfer failed ({code}); diagnostics retained privately at {log_path}')
  if not verified(partial,entry):raise ValueError('Downloaded file SHA256/size mismatch '+entry['path'])
  partial.rename(dest)
  result=[dict(path=entry['path'],bytes=entry['bytes'],sha256=entry['sha256'],method='downloaded_and_sha256_verified',resumed_prefix_bytes=offset,resumed_prefix_sha256=prefix_sha)]
  for duplicate in group[1:]:
   target=BACKUP/duplicate['path'];target.parent.mkdir(parents=True,exist_ok=True)
   if target.exists():
    if not verified(target,duplicate):raise ValueError('Conflicting duplicate '+str(target))
   else:os.link(dest,target)
   result.append(dict(path=duplicate['path'],bytes=duplicate['bytes'],sha256=duplicate['sha256'],method='deduplicated_local_hardlink',source=str(dest)))
  with mutex:progress[entry['path']]['status']='verified'
  print('VERIFIED '+entry['path'],flush=True)
  return result
 finally:
  if process.poll()is None:process.terminate();process.wait()
  diagnostics.close()

thread=threading.Thread(target=monitor,daemon=True);thread.start()
print(json.dumps(dict(status='started',unique_download_files=len(pending),download_gib=total/2**30,reused_files=len(receipts),parallel_streams=3)),flush=True)
try:
 with ThreadPoolExecutor(max_workers=3)as pool:
  futures=[pool.submit(download,group)for group in pending]
  try:
   for future in as_completed(futures):receipts.extend(future.result())
  except BaseException:
   stop.set()
   for future in futures:future.cancel()
   raise
 stop.set();thread.join();value=snapshot('complete')
 (REPORT/'large_file_receipts.json').write_text(json.dumps(dict(schema=1,status='complete',files=receipts,progress=value),indent=2)+'\n')
 print(json.dumps(dict(status='complete',download_gib=total/2**30,elapsed_seconds=value['elapsed_seconds'],throughput_mib_per_second=value['throughput_mib_per_second'])),flush=True)
except BaseException:
 stop.set();thread.join();snapshot('failed');raise
