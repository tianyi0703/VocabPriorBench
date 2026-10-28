"""CPU-only independent audit of prepared provenance and completed run accounting."""
import hashlib,json,math,sys
from pathlib import Path
import torch,yaml
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from vocabpriorbench.model import next_token_labels,state_sha256
from provenance import sha
cfg=yaml.safe_load((root/'configs/v0.yaml').read_text());m=json.loads((root/'reports/source_manifest.json').read_text(encoding='utf-8'))
assert sha(root/'configs/v0.yaml')==m['config_sha256']
for item in m['donor_files']+[m['data']['file'],m['data']['card']]:assert sha(root/item['path'])==item['sha256']
for item in [m['codes'],m['transforms']]:assert sha(root/item['path'])==item['sha256']
assert sha(root/m['common_init']['path'])==m['common_init']['file_sha256']
for path,digest in m['data_artifact_hashes'].items():assert sha(root/path)==digest,path
common=torch.load(root/'artifacts/common_init.pt',map_location='cpu',weights_only=True)
assert state_sha256(common)==m['common_init']['state_sha256']
codes=torch.load(root/'artifacts/codes.pt',map_location='cpu',weights_only=True)
transforms=torch.load(root/'artifacts/transforms.pt',map_location='cpu',weights_only=True)
for side in ['input','output']:
    code=codes[side];assert code.shape==(50304,128) and torch.isfinite(code).all()
    assert abs(code.square().mean().sqrt().item()-1/math.sqrt(128))<1e-7
    basis=transforms[side]['basis'];torch.testing.assert_close(basis.T@basis,torch.eye(128,dtype=torch.float64),atol=1e-10,rtol=1e-10)
manifest=json.loads((root/'data/manifest.json').read_text(encoding='utf-8'));seen=set()
for split in ['train','dev','holdout']:
    docs=json.loads((root/f'data/{split}_documents.json').read_text(encoding='utf-8'))
    for doc,record in zip(docs,manifest['splits'][split]):
        h=hashlib.sha256(' '.join(doc['text'].split()).encode()).hexdigest()
        assert h==record['normalized_sha256'] and h not in seen;seen.add(h)
assert len(seen)==128 and not manifest['holdout_used']
packed=torch.load(root/'data/packed.pt',map_location='cpu',weights_only=True)
assert 'holdout' not in packed
tokens=int((next_token_labels(packed['train']['ids'][:400],packed['train']['mask'][:400])!=-100).sum())
assert tokens==50800
result={'prepared_provenance':'PASS','document_disjointness':'PASS','pca_orthogonality_and_rms':'PASS','common_init':'PASS','training_prefix_valid_tokens':tokens,'cpu_only':True}
resource=root/'reports/resource_profile.json'
if resource.exists():
    profile=json.loads(resource.read_text(encoding='utf-8'));assert profile['status']=='COMPLETED'
    rows=[json.loads(line) for line in (root/'runs/v0_20261021/events.jsonl').read_text(encoding='utf-8').splitlines()]
    for arm in ['A','B','C']:
        steps=[r for r in rows if r['event']=='step' and r['arm']==arm]
        assert [s['step'] for s in steps]==list(range(1,101)) and sum(s['valid_tokens'] for s in steps)==50800
        final=torch.load(root/f'runs/v0_20261021/final_{arm}.pt',map_location='cpu',weights_only=True)
        assert all(torch.isfinite(value).all() for value in final.values())
        for side,name in [('input','input_codes'),('output','output_codes')]:
            initial=codes[side] if arm!='C' else torch.where(codes[side]>=0,1.,-1.)/math.sqrt(128)
            assert bool(torch.equal(final[name],initial))==(arm!='A')
    assert profile['gpu_budget']['cumulative_seconds']<=7200 and profile['common_completed_steps']==100
    assert len({a['common_init_sha256'] for a in profile['arms']})==1
    assert len({a['training_prefix_sha256'] for a in profile['arms']})==1
    result.update(run_accounting='PASS',code_freeze_and_update='PASS',finite_final_states='PASS',gpu_budget='PASS')
output=root/'runs/artifact_audit.json';output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
