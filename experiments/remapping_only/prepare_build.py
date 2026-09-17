"""Recreate the measured build context; never download weights or dependencies."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

SOURCE = '24a68f9c2c0bc407508a7b6990817509674af74b'
CHECKPOINT = 'a398d6e5e79af8ce9cf697fdd10b1887927dc838cdafcb5ae5c3352a2d56d96a'
ORIGINAL_IMAGE = 'sha256:00a1a1d93ccf2d94e4f2f770f3cfdf0f7c63e99047931df9b103e24170faadf7'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--wrapper-image', default=ORIGINAL_IMAGE,
                        help='Existing local image containing the measured common LAZ wrapper')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    repo = here.parents[1]
    expected = json.loads((here/'results/2026-09-17/original.expected-files.json').read_text())
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if digest != CHECKPOINT:
        raise ValueError('Checkpoint SHA-256 differs from the measured v6 checkpoint')
    archive = subprocess.check_output(['git', '-C', str(repo), 'archive', SOURCE, 'oneformer3d', 'configs'])
    # git archive uses only paths from the explicit immutable tree above.
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        if any(m.issym() or m.islnk() or m.name.startswith('/') or '..' in Path(m.name).parts for m in tar.getmembers()):
            raise ValueError('Unexpected archive entry')
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/'main').mkdir()
        tar.extractall(args.output/'main')
    container = subprocess.check_output(['docker', 'create', '--pull=never', args.wrapper_image], text=True).strip()
    try:
        for folder in ['src', 'scripts']:
            target = args.output/'tool'/folder
            target.mkdir(parents=True)
            subprocess.run(['docker', 'cp', f'{container}:/workspace/{folder}/.', str(target)], check=True)
        (args.output/'tool/tools').mkdir()
        subprocess.run(['docker', 'cp', f'{container}:/workspace/tools/test.py', str(args.output/'tool/tools/test.py')], check=True)
    finally:
        subprocess.run(['docker', 'rm', container], check=True)
    for path in (args.output/'tool').rglob('__pycache__'):
        shutil.rmtree(path)
    for path in (args.output/'tool').rglob('*'):
        if path.is_file():
            image_path = '/workspace/' + str(path.relative_to(args.output/'tool'))
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected.get(image_path):
                raise ValueError(f'Common wrapper differs from measured image: {image_path}')
    remap = args.output/'remap/oneformer3d'
    remap.mkdir(parents=True)
    for name in ['oneformer3d.py', 'remapping.py', 'hybrid_nearest.py']:
        shutil.copy2(repo/'oneformer3d'/name, remap/name)
    for name in ['Dockerfile', 'benchmark_v6.py']:
        shutil.copy2(here/name, args.output/name)
    shutil.copy2(args.checkpoint, args.output/'v6_epoch_1500_fix.pth')
    print(f'Prepared {args.output}; build original and remap targets with the pinned BASE_IMAGE.')


if __name__ == '__main__':
    main()
