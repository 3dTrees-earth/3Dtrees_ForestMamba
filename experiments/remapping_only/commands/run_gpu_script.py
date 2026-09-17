import fcntl,json,os,shutil,subprocess,sys,time
from pathlib import Path
from docker_state import inspect_container
r=Path(__file__).resolve().parents[1];label=sys.argv[1];script=sys.argv[2];m=json.loads((r/'manifest.json').read_text());name='fm-small-'+label+'-'+r.name.rsplit('-',1)[-1].lower()
slot=(r/'gpu2.lock').open('a');fcntl.flock(slot,fcntl.LOCK_EX|fcntl.LOCK_NB)
source=r/script.removeprefix('/run/')
snapshot=r/'commands'/f'{label}.snapshot.py';shutil.copy2(source,snapshot)
request=r/'provenance'/f'{label}.request.json'
request_source=Path(sys.argv[3]) if len(sys.argv)>3 else r/'provenance/replay-request.json'
if request_source.resolve()!=request.resolve():shutil.copy2(request_source,request)
script='/run/commands/'+snapshot.name
cmd=['docker','run','--pull=never','--name',name,'--cpus','10','--memory','50000000000','--memory-swap','50000000000','--network','none','--shm-size','4g','--user',f'{os.getuid()}:{os.getgid()}','--gpus','device='+m['gpu_uuid']]
for src,dst in [(r,'/run'),(r/'work','/scratch'),(Path(m['frozen_outputs']),'/fixtures:ro')]:cmd+=['-v',f'{src}:{dst}']
for k,v in {'FM_REPLAY_REQUEST':'/run/provenance/'+request.name,'FM_REPLAY_LABEL':label,'CUDA_VISIBLE_DEVICES':'0','PYTHONPATH':'/workspace:/run/commands','HOME':'/scratch','TMPDIR':'/scratch','OMP_NUM_THREADS':'10','OPENBLAS_NUM_THREADS':'10','MKL_NUM_THREADS':'10','PYTHONDONTWRITEBYTECODE':'1','PYTHONHASHSEED':'1838'}.items():cmd+=['-e',f'{k}={v}']
cmd+=['--workdir','/workspace','--entrypoint','/opt/conda/bin/python',m['image'],'-u',script]
(r/'commands'/f'{label}.docker.json').write_text(json.dumps(cmd,indent=2))
with (r/'logs'/f'{label}.log').open('w') as log:
 client=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
 while client.poll() is None:
  try:info=inspect_container(name)
  except (OSError,TimeoutError):
   time.sleep(2);continue
  if info and info['State']['Status']=='exited':client.kill();client.wait();break
  time.sleep(2)
while True:
 try:
  info=inspect_container(name);break
 except (OSError,TimeoutError):time.sleep(2)
if info is None:raise RuntimeError('Replay container was not created; inspect the preserved Docker log')
(r/'logs'/f'{label}.container.json').write_text(json.dumps({k:info[k] for k in ['State','HostConfig','Image']},indent=2));raise SystemExit(info['State']['ExitCode'])
