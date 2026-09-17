"""Sequential, capped model execution with GPU telemetry and retained failure evidence."""
import argparse, fcntl, json, os, pathlib, subprocess, time
from docker_state import inspect_container
p=argparse.ArgumentParser();p.add_argument('--trace',action='store_true');p.add_argument('--detail-window',type=int,default=0);p.add_argument('label');p.add_argument('--variant',choices=['main','remap'],default='main');p.add_argument('--input',type=pathlib.Path);p.add_argument('--gpu-index',type=int,choices=[0,2],required=True);args=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[1]
lock=(root/('gpu'+str(args.gpu_index)+'.lock')).open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
m=json.loads((root/'manifest.json').read_text());label=args.label
m['gpu_index']=args.gpu_index
assert args.gpu_index == 2
m['gpu_uuid']=m['gpu_uuid']
assert label.replace('_','').isalnum(), 'Use an alphanumeric stage label'
assert not any(s['status']=='failed' for s in m['stages']), 'A stage failed: investigate before extending this run'
(root/'work'/label).mkdir();(root/'outputs'/label).mkdir()
source=root/'sources'/args.variant
inp=(args.input or pathlib.Path(m['input'])).resolve(strict=True)
assert not args.trace, 'Clean image benchmark must not trace the model'
name='fmopt-'+label+'-'+root.name.rsplit('-',1)[-1].lower()
cmd=['docker','run','--pull=never','--name',name,'--cpus','10','--memory','50000000000','--memory-swap','50000000000','--shm-size','4g','--network','none','--user',f'{os.getuid()}:{os.getgid()}','--gpus','device='+m['gpu_uuid']]
env={'CUDA_VISIBLE_DEVICES':'0','GALAXY_SLOTS':'10','OMP_NUM_THREADS':'10','OPENBLAS_NUM_THREADS':'10','MKL_NUM_THREADS':'10','NUMEXPR_NUM_THREADS':'10','HOME':'/scratch','TMPDIR':'/scratch','NUMBA_CACHE_DIR':'/scratch/numba','MPLCONFIGDIR':'/scratch/matplotlib','WANDB_MODE':'disabled','PYTHONDONTWRITEBYTECODE':'1','PYTHONHASHSEED':'1838','XDG_CACHE_HOME':'/scratch/.cache','HF_HOME':'/scratch/huggingface','TRANSFORMERS_CACHE':'/scratch/huggingface/transformers','PYTHONPATH':'/workspace:/run/commands','FM_BENCH_LABEL':label,'FM_DETAIL_WINDOW':str(args.detail_window)}
for key,val in env.items():cmd+=['-e',f'{key}={val}']
mounts=[(root,'/run'),(root/'work'/label,'/scratch'),(inp,'/input/input.laz:ro')]
for src,dst in mounts:cmd+=['-v',f'{src}:{dst}']
cmd+=['--workdir','/scratch','--entrypoint','python',m['images'][args.variant],'/run/commands/run_stage.py',label,'python','-u','/workspace/src/run.py','--dataset-path','/input/input.laz','--output-dir','/run/outputs/'+label,'--work-dir','/scratch','--repo-dir','/workspace','--checkpoint','/workspace/work_dirs/forestmamba_chm_radius16_qp300_2many_v6/v6_epoch_1500_fix.pth','--config-base','/workspace/configs/benchmark_v6.py','--preprocess-workers','10','--chunk-size','1000000','--semantic-dim','PredSemantic_FM','--instance-dim','PredInstance_FM','--score-dim','PredScore_FM','--benchmark-stage-log','/run/logs/'+label+'.stages.jsonl','--bluepoint-iterations','0','--bluepoint-second-pass-threshold','0.01','--spatial-match-tolerance','0.01']
(root/'commands'/f'{label}.json').write_text(json.dumps(cmd,indent=2))
telemetry=['nvidia-smi','-i',m['gpu_uuid'],'--query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw','--format=csv','-lms','200']
(root/'commands'/f'{label}.gpu.json').write_text(json.dumps(telemetry,indent=2))
stage={'name':label,'variant':args.variant,'image':m['images'][args.variant],'input':str(inp),'started_at':time.time(),'container':name,'status':'running','gpu_index':args.gpu_index,'gpu_uuid':m['gpu_uuid'],'cpu_limit':10,'memory_bytes':50000000000}
(root/'logs'/f'{label}.stage.json').write_text(json.dumps(stage,indent=2))
print(json.dumps(stage),flush=True)
with (root/'logs'/f'{label}.gpu.csv').open('w') as gpu, (root/'logs'/f'{label}.log').open('w') as log:
    monitor=subprocess.Popen(telemetry,stdout=gpu,stderr=subprocess.STDOUT)
    client=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
    client_cleanup=False
    try:
        while client.poll() is None:
            # Docker log attachment has sometimes remained open after exit.
            # Only terminate our client after the authoritative container state
            # confirms completion; preserve the actual container exit code.
            if (root/'logs'/f'{label}.resources.json').exists():
                try: info=inspect_container(name)
                except (OSError, TimeoutError):
                    time.sleep(1);continue
                if info is not None:
                    state=info['State']
                    if state['Status']=='exited':
                        client.kill();client.wait()
                        result=subprocess.CompletedProcess(cmd,state['ExitCode'])
                        client_cleanup=True
                        break
            time.sleep(1)
        else:
            result=subprocess.CompletedProcess(cmd,client.returncode)
        stage['docker_client_cleanup_after_confirmed_exit']=client_cleanup
    finally:
        monitor.terminate()
        try:monitor.wait(timeout=5)
        except subprocess.TimeoutExpired:monitor.kill();monitor.wait()
stage.update(status='completed' if result.returncode==0 else 'failed',exit_code=result.returncode,host_elapsed_seconds=time.time()-stage['started_at'])
info=inspect_container(name)
if info is not None:
    (root/'logs'/f'{label}.container.json').write_text(json.dumps({k:info[k] for k in ['Id','Image','State','HostConfig']},indent=2))
(root/'logs'/f'{label}.stage.json').write_text(json.dumps(stage,indent=2))
print(json.dumps(stage),flush=True)
raise SystemExit(result.returncode)
