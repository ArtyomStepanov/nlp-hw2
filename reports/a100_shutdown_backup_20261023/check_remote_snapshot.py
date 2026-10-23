from pathlib import Path
import json,shlex,subprocess
report=Path('reports/a100_shutdown_backup_20261023')
source=json.loads((report/'remote_sha256_manifest.json').read_text())
script=r'''
from pathlib import Path
import json,sys,time,ctypes,ctypes.util
manifest=json.load(sys.stdin);root=Path(manifest['remote_root']);changed=[]
for entry in manifest['files']:
 p=root/entry['path']
 if not p.is_file()or p.is_symlink():changed.append(dict(path=entry['path'],error='missing or symbolic'));continue
 stat=p.stat()
 if(stat.st_size,stat.st_mtime_ns)!=(entry['bytes'],entry['mtime_ns']):changed.append(dict(path=entry['path'],error='size or modification time changed'))
gpu={}
try:
 lib=ctypes.CDLL(ctypes.util.find_library('nvidia-ml')or'libnvidia-ml.so.1');gpu['init_status']=lib.nvmlInit_v2()
 if gpu['init_status']==0:
  count=ctypes.c_uint();lib.nvmlDeviceGetCount_v2(ctypes.byref(count));gpu['devices']=[]
  for index in range(count.value):
   handle=ctypes.c_void_p();lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index),ctypes.byref(handle))
   class Util(ctypes.Structure):_fields_=[('gpu',ctypes.c_uint),('memory',ctypes.c_uint)]
   util=Util();lib.nvmlDeviceGetUtilizationRates(handle,ctypes.byref(util))
   processes=ctypes.c_uint();status=lib.nvmlDeviceGetComputeRunningProcesses_v3(handle,ctypes.byref(processes),None)
   gpu['devices'].append(dict(index=index,utilization_percent=util.gpu,compute_process_count=processes.value,count_query_status=status))
  lib.nvmlShutdown()
except Exception as error:gpu['error']=str(error)
print(json.dumps(dict(schema=1,status='unchanged'if not changed else'changed',checked_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),files_checked=len(manifest['files']),changed=changed,verification='Source size/mtime unchanged after fresh initial SHA256; each transferred local file separately verified by full SHA256',gpu=gpu,concurrent_new_training_reported_pid=268935,concurrent_new_training_tmux='nlp-a100-dropout-20261023',remote_stop_authorized=False),indent=2))
'''
result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','nlp-bart.ml-contest-mla.ws.deep.yandex.net','python3 -c '+shlex.quote(script)],input=json.dumps(source),text=True,stdout=subprocess.PIPE,check=True)
value=json.loads(result.stdout)
(report/'remote_snapshot_final_check.json').write_text(json.dumps(value,indent=2)+'\n')
assert value['status']=='unchanged',value['changed']
print(json.dumps(value,indent=2))
