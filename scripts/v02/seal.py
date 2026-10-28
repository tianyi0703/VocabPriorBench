"""Seal V0.2 and byte-for-byte copies of the actual V0.1 model dependencies."""
import hashlib,json,os
from pathlib import Path
import yaml
root=Path(os.environ['VPB_WORKSPACE']).resolve();cfg=yaml.safe_load((root/'configs/v02.yaml').read_text())
target=root/'runs'/cfg['run_name']/'snapshot'
if target.exists():raise SystemExit('Existing snapshot; use a new run directory')
paths=[root/'CODEX_V02_VocabPriorBench.txt',root/'configs/v02.yaml',root/'reports/v02/EXPERIMENT_CONTRACT.md',root/'requirements.txt',root/'tests/test_v02_tables.py']
paths.extend((root/'scripts/v02').glob('*.py'))
copies={p.relative_to(root).as_posix():p for p in paths}
deps={}
old_manifest=json.loads((root/'runs/v01_20261022/snapshot/SNAPSHOT_MANIFEST.json').read_text(encoding='utf-8'))
for name in ['model.py','v01_model.py','__init__.py']:
    source=root/cfg['reuse']['model_source']/name;relative='src/vocabpriorbench/'+name
    expected=old_manifest['files'][relative]
    assert source.stat().st_size==expected['bytes'] and hashlib.sha256(source.read_bytes()).hexdigest()==expected['sha256'],relative
    copies[relative]=source;deps[relative]={'source':source.relative_to(root).as_posix(),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
files={}
for relative,source in sorted(copies.items()):
    data=source.read_bytes();path=target/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    files[relative]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
tree=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
(target/'SNAPSHOT_MANIFEST.json').write_text(json.dumps({'files':files,'tree_sha256':tree,'model_dependency_sources':deps},indent=2),encoding='utf-8')
print(json.dumps({'snapshot':str(target),'tree_sha256':tree,'files':len(files)},indent=2))
