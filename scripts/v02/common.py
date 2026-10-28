"""V0.2 paths are explicit; never import scripts/v01/common.py."""
import hashlib,json,os,sys,time
from pathlib import Path
import yaml
CODE_ROOT=Path(__file__).resolve().parents[2]
if 'VPB_WORKSPACE' not in os.environ:raise RuntimeError('VPB_WORKSPACE must be set explicitly')
ROOT=Path(os.environ['VPB_WORKSPACE']).resolve()
assert ROOT==Path('F:/VocabPriorBench').resolve()
CFG=yaml.safe_load((CODE_ROOT/'configs/v02.yaml').read_text(encoding='utf-8'))
RUN=ROOT/'runs'/CFG['run_name'];REPORT=ROOT/CFG['report_directory'];PRIVATE=RUN/'private'
for path in [RUN,REPORT,PRIVATE]:path.mkdir(parents=True,exist_ok=True)
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
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)
def verify_snapshot():
    m=read_json(CODE_ROOT/'SNAPSHOT_MANIFEST.json')
    for name,spec in m['files'].items():assert sha(CODE_ROOT/name)==spec['sha256'],name
    return m
def input_specs():
    previous=read_json(ROOT/CFG['reuse']['v01_manifest'])
    specs={name.replace('\\','/'):spec for group in ['input_artifacts','prepared_artifacts'] for name,spec in previous[group].items()}
    names=[CFG['reuse'][key] for key in ['codes','packed','prior','selection']]
    for seed in CFG['seeds']:
        names.extend(CFG['reuse'][key].format(seed=seed) for key in ['common_init_template','block_order_template'])
    return {name:specs[name] for name in names}
def verify_inputs(include_derived=True):
    failures=[];specs=input_specs()
    if include_derived:
        m=read_json(REPORT/'source_manifest.json');specs.update(m.get('derived_artifacts',{}))
    for name,spec in specs.items():
        path=ROOT/name
        if not path.exists():failures.append({'path':name,'failure':'MISSING'});continue
        if path.stat().st_size!=spec['bytes'] or sha(path)!=spec['sha256']:failures.append({'path':name,'failure':'SIZE_OR_SHA_MISMATCH'})
    if failures:raise ValueError(json.dumps({'status':'BLOCKED_MISSING_OR_MISMATCHED_V01_INPUT','failures':failures}))
    return specs
