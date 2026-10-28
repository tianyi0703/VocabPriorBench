"""Immutable execution snapshot; no weights, corpus, or environment copied."""
import hashlib,json,shutil,time
from pathlib import Path
root=Path(__file__).resolve().parents[2];target=root/'runs/v01_20261022/snapshot'
if target.exists():raise SystemExit('Refuse to overwrite a sealed execution snapshot')
paths=[root/'CODEX_V01_VocabPriorBench.txt',root/'requirements.txt',root/'configs/v0.yaml',root/'configs/v01.yaml',root/'reports/v01/EXPERIMENT_CONTRACT.md']
for folder in ['src','scripts','tests']:
    paths.extend(p for p in (root/folder).rglob('*.py') if '__pycache__' not in str(p))
files={}
for path in sorted(set(paths)):
    name=path.relative_to(root).as_posix();data=path.read_bytes();out=target/name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
    files[name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
tree=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
(target/'SNAPSHOT_MANIFEST.json').write_text(json.dumps({'sealed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'tree_sha256':tree,'files':files},indent=2),encoding='utf-8')
print(json.dumps({'snapshot':str(target),'files':len(files),'tree_sha256':tree},indent=2))
