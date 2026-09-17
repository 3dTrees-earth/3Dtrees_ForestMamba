"""Initialize the exact small-crop replay/5+5 protocol in a fresh directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--frozen-outputs', required=True, type=Path,
                   help='Directory containing tree68_10m_capture/frozen from the original capture')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--gpu-timezone', default='Europe/Berlin', help='Host nvidia-smi timestamp timezone')
    a = p.parse_args()
    here = Path(__file__).resolve().parent
    evidence = here/'results/2026-09-17'
    m = json.loads((evidence/'manifest.json').read_text())
    if hashlib.sha256(a.input.read_bytes()).hexdigest() != m['input_sha256']:
        raise ValueError('This recorded protocol requires the original tree68_10m crop (input hash mismatch)')
    fixture = a.frozen_outputs.resolve()/'tree68_10m_capture/frozen'
    done = json.loads((fixture/'complete.json').read_text())
    if not done['completed'] or done['windows_captured'] != 6:
        raise ValueError('Incomplete six-window frozen fixture')
    for image in m['images'].values():
        actual = subprocess.check_output(['docker','image','inspect',image,'--format','{{.Id}}'],text=True).strip()
        if actual != image:
            raise ValueError('The measured immutable images must be available locally')
    gpu = subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    cpu_python = os.environ.get('FM_CPU_PYTHON', 'python3')
    subprocess.run([cpu_python,'-c','import laspy, lazrs, numpy, scipy'],check=True)
    root = a.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    for folder in ['logs','outputs','work','provenance','inputs']:
        (root/folder).mkdir()
    shutil.copytree(here/'commands',root/'commands',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(a.input,root/'inputs/tree68_10m.laz')
    for name in ['original.expected-files.json','remap.expected-files.json']:
        shutil.copy2(evidence/name,root/'provenance'/name)
    # Independently verify the immutable image sources again; do not synthesize a pass.
    for variant, expected in [('main','original'),('remap','remap')]:
        command=['docker','run','--rm','--pull=never','--network','none',
                 '--cpus','10','--memory','50000000000','--memory-swap','50000000000',
                 '--user',f'{os.getuid()}:{os.getgid()}',
                 '-v',f'{root}:/run:ro','--entrypoint','python',m['images'][variant],
                 '/run/commands/verify_image.py',f'/run/provenance/{expected}.expected-files.json']
        result=subprocess.check_output(command,text=True)
        if not json.loads(result)['passed']:
            raise ValueError('Image verification failed')
        (root/'logs'/f'verify_{variant}.log').write_text(result)
    (root/'provenance/image_replay.request.json').write_text(json.dumps({'capture':'tree68_10m_capture'}))
    m.update(run_root=str(root),created_at=datetime.now(timezone.utc).isoformat(),status='images_verified',
             input=str(root/'inputs/tree68_10m.laz'),gpu_uuid=gpu,gpu_timezone=a.gpu_timezone,
             frozen_outputs=str(a.frozen_outputs.resolve()),stages=[],cohorts=[{'group':'tree68_10m','labels':[]}])
    for key in ['completed_at','failure','adoption_status','benchmark_summary','report','recovery','frozen_evidence']:
        m.pop(key,None)
    (root/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    print('Initialized',root)
    print('Next: python',root/'commands/benchmark.py')
    print('Then: python',root/'commands/report.py')


if __name__ == '__main__':
    main()
