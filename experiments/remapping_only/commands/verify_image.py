import hashlib,json,sys
from pathlib import Path
expected=json.loads(Path(sys.argv[1]).read_text());bad=[]
for name,wanted in expected.items():
 p=Path(name)
 actual=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
 if actual!=wanted:bad.append(name)
actual_model={str(p) for p in Path('/workspace/oneformer3d').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
expected_model={p for p in expected if p.startswith('/workspace/oneformer3d/')}
assert not bad,bad
assert actual_model==expected_model,sorted(actual_model^expected_model)
print(json.dumps({'passed':True,'files_verified':len(expected),'exact_model_file_set':True}))
