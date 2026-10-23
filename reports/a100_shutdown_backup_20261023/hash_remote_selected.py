from pathlib import Path
import json,shlex,subprocess
remote_script=r'''
from pathlib import Path
import json,hashlib,sys,time
root=Path('/home/artyom-step/workspace/nlp-hw2')
paths=json.load(sys.stdin)
files=[];cache={}
for name in paths:
 p=root/name
 if not p.resolve().is_relative_to(root) or p.is_symlink() or not p.is_file():raise ValueError('Unsafe source '+name)
 stat=p.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
 if key not in cache:
  h=hashlib.sha256()
  with p.open('rb')as f:
   for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
  after=p.stat()
  if(stat.st_size,stat.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Source changed '+name)
  cache[key]=h.hexdigest()
 files.append(dict(path=name,bytes=stat.st_size,sha256=cache[key],dev=stat.st_dev,inode=stat.st_ino,links=stat.st_nlink,mtime_ns=stat.st_mtime_ns))
print(json.dumps(dict(schema=1,remote_root=str(root),hashed_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),files=files,unique_source_inodes=len(cache)),indent=2))
'''
root=Path('reports/a100_shutdown_backup_20261023')
paths=(root/'selected_paths.json').read_text()
result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','nlp-bart.ml-contest-mla.ws.deep.yandex.net','python3 -c '+shlex.quote(remote_script)],input=paths,text=True,stdout=subprocess.PIPE,check=True)
(root/'remote_sha256_manifest.json').write_text(result.stdout)
value=json.loads(result.stdout)
print(json.dumps(dict(files=len(value['files']),unique_source_inodes=value['unique_source_inodes'],bytes=sum(e['bytes']for e in value['files'])),indent=2))
