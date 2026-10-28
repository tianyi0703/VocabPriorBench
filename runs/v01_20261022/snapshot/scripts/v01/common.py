"""Small shared helpers for the sealed V0.1 implementation, offline only."""
import hashlib,json,os,sys,time
from pathlib import Path
import yaml
CODE_ROOT=Path(__file__).resolve().parents[2]
ROOT=Path(os.environ.get('VPB_WORKSPACE',str(CODE_ROOT))).resolve()
CFG=yaml.safe_load((CODE_ROOT/'configs/v01.yaml').read_text(encoding='utf-8'))
RUN=ROOT/'runs'/CFG['run_name'];REPORT=ROOT/'reports/v01'
PRIVATE=RUN/'private'
for p in [RUN,REPORT,PRIVATE]:p.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(CODE_ROOT/'src'))
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)
def verify_snapshot():
    manifest=read_json(CODE_ROOT/'SNAPSHOT_MANIFEST.json')
    for name,spec in manifest['files'].items():assert sha(CODE_ROOT/name)==spec['sha256'],name
    return manifest
def verify_inputs():
    m=read_json(REPORT/'source_manifest.json')
    for name,spec in m['input_artifacts'].items():assert sha(ROOT/name)==spec['sha256'],name
    for name,spec in m.get('prepared_artifacts',{}).items():assert sha(ROOT/name)==spec['sha256'],name
    return m
