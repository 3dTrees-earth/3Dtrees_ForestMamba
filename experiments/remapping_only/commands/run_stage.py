import json, os, pathlib, subprocess, sys, time
label=sys.argv[1]; argv=sys.argv[2:]
start=time.time()
result=subprocess.run(argv)
metrics={'label':label,'argv':argv,'elapsed_seconds':time.time()-start,'exit_code':result.returncode}
for name in ['memory.peak','memory.max','memory.swap.max','cpu.max','cpu.stat','memory.events']:
 p=pathlib.Path('/sys/fs/cgroup')/name
 if p.exists(): metrics[name]=p.read_text().strip()
pathlib.Path('/run/logs',label+'.resources.json').write_text(json.dumps(metrics,indent=2))
print('STAGE_RESULT',json.dumps(metrics),flush=True)
sys.exit(result.returncode)
