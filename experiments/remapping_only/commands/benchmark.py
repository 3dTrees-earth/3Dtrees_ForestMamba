import json,subprocess,sys,traceback
from pathlib import Path
r=Path(__file__).resolve().parents[1]
def read():return json.loads((r/'manifest.json').read_text())
def save(m):(r/'manifest.json').write_text(json.dumps(m,indent=2))
def command(args,label):
 with (r/'logs'/(label+'.launcher.log')).open('w') as f:
  p=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError(label+' failed; inspect preserved log')
def ensure_gpu_free():
 gpu=read()['gpu_uuid']
 active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 assert not any(line.split(',')[0].strip()==gpu for line in active.splitlines()),'GPU2 is occupied; stop without disturbing its owner'
def main():
 m=read();assert m['status']=='images_verified';m['status']='testing_packaged_helper';save(m)
 ensure_gpu_free()
 command([sys.executable,str(r/'commands/run_gpu_script.py'),'image_replay','/run/commands/frozen_replay.py',str(r/'provenance/image_replay.request.json')],'image_replay')
 print('PASSED packaged-helper exact replay',flush=True)
 m=read();m['status']='benchmarking';save(m)
 for i in range(1,6):
  # Alternate pair order to avoid giving one variant every earlier position.
  for variant in (['main','remap'] if i%2 else ['remap','main']):
   label=f'tree68_10m_{variant}_{i:02d}';print('START',label,flush=True)
   ensure_gpu_free()
   command([sys.executable,str(r/'commands/launch.py'),label,'--variant',variant,'--gpu-index','2','--input',m['input']],label)
   command([sys.executable,str(r/'commands/run_cpu.py'),str(r/'commands/validate_output.py'),m['input'],str(r/'outputs'/label/'input_forestmamba.laz'),str(r/'logs'/(label+'.validation.json'))],label+'_validation')
   m=read();m['stages'].append(json.loads((r/'logs'/(label+'.stage.json')).read_text()));m['cohorts'][0]['labels'].append(label);save(m)
   print('DONE',label,flush=True)
 # Existing comparator assumes its anchor is first; chronological order still starts main_01.
 command([sys.executable,str(r/'commands/run_cpu.py'),str(r/'commands/compare_cohort.py'),str(r),'tree68_10m'],'analysis')
 m=read();m['status']='benchmark_complete';save(m);print('BENCHMARK_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  m=read();m['status']='failed';m['failure']=traceback.format_exc();save(m);raise
