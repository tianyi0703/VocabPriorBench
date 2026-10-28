"""Download one pinned donor and one small data shard, then prepare CPU-only artifacts."""
import hashlib, json, math, os, re, sys, time, urllib.request
from pathlib import Path
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from transformers import AutoTokenizer, GPTNeoXForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vocabpriorbench.model import Student, common_state_dict, state_sha256
from provenance import fetch,sha

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def pca_codes(table, k):
    # O(V*d + d*d) memory; never materialize a V*V matrix.
    x=table.detach().cpu().to(torch.float64)
    mean=x.mean(0); x-=mean
    covariance=x.T @ x / (x.shape[0]-1)
    values,vectors=torch.linalg.eigh(covariance)
    basis=vectors[:, -k:].flip(1).contiguous()
    pivots=basis.abs().argmax(0)
    orientation=torch.sign(basis[pivots,torch.arange(k)])
    basis*=orientation
    z=x @ basis
    scale=(1/math.sqrt(k))/z.square().mean().sqrt()
    codes=(z*scale).float()
    return codes,{'mean':mean,'basis':basis,'scale':scale,'eigenvalues':values.flip(0),
                  'explained_variance_fraction':float(values[-k:].sum()/values.sum()),
                  'algorithm':'exact covariance eigh CPU float64; descending; max-abs pivot positive'}

def main():
    torch.set_num_threads(6)
    cfg=yaml.safe_load((ROOT/'configs/v0.yaml').read_text())
    torch.manual_seed(cfg['seed'])
    donor_dir=ROOT/'artifacts/donor'; donor_dir.mkdir(parents=True,exist_ok=True)
    files=[]
    for name in ['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','README.md','model.safetensors']:
        files.append(fetch(f"https://huggingface.co/{cfg['donor']['repo']}/resolve/{cfg['donor']['revision']}/{name}",donor_dir/name))
    tokenizer=AutoTokenizer.from_pretrained(donor_dir,local_files_only=True)
    model=GPTNeoXForCausalLM.from_pretrained(donor_dir,local_files_only=True,torch_dtype=torch.float32).cpu()
    inp=model.get_input_embeddings().weight; out=model.get_output_embeddings().weight
    inspection={'input_shape':list(inp.shape),'output_shape':list(out.shape),
                'same_parameter_object':inp is out,'same_storage':inp.data_ptr()==out.data_ptr(),
                'config_tie_word_embeddings':model.config.tie_word_embeddings,
                'unique_parameters':sum(p.numel() for p in {id(p):p for p in model.parameters()}.values()),
                'all_parameter_objects':sum(1 for _ in model.parameters()),'tokenizer_length':len(tokenizer),
                'eos_token_id':tokenizer.eos_token_id,'pad_token_id':tokenizer.pad_token_id,
                'output_bias':model.get_output_embeddings().bias is not None}
    assert inp is not out and inp.data_ptr()!=out.data_ptr()
    assert not model.config.tie_word_embeddings and inp.shape==out.shape==(50304,512)
    cin,tin=pca_codes(inp,cfg['code_width']); cout,tout=pca_codes(out,cfg['code_width'])
    transforms=ROOT/'artifacts/transforms.pt'; torch.save({'input':tin,'output':tout},transforms)
    codepath=ROOT/'artifacts/codes.pt'; torch.save({'input':cin,'output':cout},codepath)
    vocab=sorted(tokenizer.get_vocab().items(),key=lambda t:t[1])
    vocabpath=ROOT/'artifacts/vocab_order.json';write_json(vocabpath,vocab)
    inspection.update({'continuous_rms_input':float(cin.square().mean().sqrt()),
                       'continuous_rms_output':float(cout.square().mean().sqrt()),
                       'input_explained_variance':tin['explained_variance_fraction'],
                       'output_explained_variance':tout['explained_variance_fraction']})
    del model,inp,out
    revision=cfg['data']['revision']
    datapath=ROOT/'data/source_train.parquet'
    datafile=fetch(f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/{revision}/{cfg['data']['source_file']}",datapath)
    datacard=fetch(f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/{revision}/README.md',ROOT/'data/DATASET_CARD.md')
    lines=pq.read_table(datapath,columns=['text']).column('text').to_pylist()
    docs=[]; current=[]
    for text in lines:
        if re.match(r'^\s*= [^=].*[^=] =\s*$',text) and current:
            docs.append(''.join(current));current=[]
        current.append(text)
    if current: docs.append(''.join(current))
    unique=[];seen=set(); duplicates=0
    for text in docs:
        norm=' '.join(text.split()); h=hashlib.sha256(norm.encode()).hexdigest()
        if not norm: continue
        if h in seen: duplicates+=1;continue
        seen.add(h); unique.append((h,text))
    selected=unique[:cfg['data']['documents']]
    assert len(selected)==128, len(selected)
    order=np.random.default_rng(cfg['seed']).permutation(len(selected)).tolist()
    splits={'train':order[:96],'dev':order[96:112],'holdout':order[112:128]}
    records={s:[{'source_document_index':i,'normalized_sha256':selected[i][0],
                  'utf8_bytes':len(selected[i][1].encode())} for i in ids] for s,ids in splits.items()}
    assert len({r['normalized_sha256'] for rs in records.values() for r in rs})==128
    packed={}; counts={}; seq=cfg['training']['sequence_length']; pad=tokenizer.eos_token_id
    for split,indices in splits.items():
        write_json(ROOT/f'data/{split}_documents.json', [{'source_document_index':i,'text':selected[i][1]} for i in indices])
        if split=='holdout':continue # Independent holdout is neither tokenized nor evaluated in V0.
        tokens=[]
        for i in indices:tokens.extend(tokenizer.encode(selected[i][1],add_special_tokens=False)+[pad])
        original=len(tokens); blocks=[];masks=[]
        for start in range(0,len(tokens),seq):
            chunk=tokens[start:start+seq]; valid=len(chunk)
            if valid<2:break
            blocks.append(chunk+[pad]*(seq-valid));masks.append([1]*valid+[0]*(seq-valid))
        packed[split]={'ids':torch.tensor(blocks,dtype=torch.long),'mask':torch.tensor(masks,dtype=torch.bool)}
        counts[split]={'raw_tokens_including_eos':original,'blocks':len(blocks),
                      'valid_prediction_tokens':sum(sum(m)-1 for m in masks)}
    assert len(packed['train']['ids'])>=400
    packedpath=ROOT/'data/packed.pt';torch.save(packed,packedpath)
    student=Student(cin,cout,True,**cfg['student'])
    common=common_state_dict(student);commonpath=ROOT/'artifacts/common_init.pt';torch.save(common,commonpath)
    write_json(ROOT/'data/manifest.json',{'source_revision':revision,'selection':'first 128 nonempty normalized-unique full articles from original training shard, then RNG1729 permutation before tokenization',
       'source_articles':len(docs),'source_unique_articles':len(unique),'normalized_duplicate_articles_removed':duplicates,
       'splits':records,'token_counts':counts,'packed_sha256':sha(packedpath),'holdout_used':False,
       'donor_evaluation_overlap':'UNKNOWN','deduplication_scope':'whole-document normalized exact hashes, not near-duplicate or donor decontamination'})
    manifest={'prepared_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'donor':cfg['donor'],
        'donor_files':files,'donor_inspection':inspection,
        'tokenizer_order_sha256':sha(vocabpath),'tokenizer_unused_output_rows':list(range(len(tokenizer),cin.shape[0])),
        'data':{'repo':'Salesforce/wikitext','revision':revision,
                'license':'Pinned card YAML: cc-by-sa-3.0 + gfdl; card Licensing Information prose: CC-BY-SA-4.0',
                'license_version_status':'SOURCE_INTERNAL_CONFLICT; public share-alike source, exact version not independently resolved',
                'file':datafile,'card':datacard},
        'transforms':{'path':str(transforms.relative_to(ROOT)),'sha256':sha(transforms),'seed':cfg['seed'],'fit_data':'donor tables only; independent input/output PCA'},
        'codes':{'path':str(codepath.relative_to(ROOT)),'sha256':sha(codepath),'dtype':'float32','binary_bit_packed':False},
        'common_init':{'path':str(commonpath.relative_to(ROOT)),'file_sha256':sha(commonpath),'state_sha256':state_sha256(common)},
        'config_sha256':sha(ROOT/'configs/v0.yaml'),
        'data_artifact_hashes':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'data').glob('*') if p.is_file()},
        'code_version':'V0 content-addressed source snapshot (workspace was not a Git repository)',
        'source_files':{str(p.relative_to(ROOT)):sha(p) for folder in ['src','scripts','tests','configs'] for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)}}
    write_json(ROOT/'reports/source_manifest.json',manifest)
    print(json.dumps({'event':'prepared','inspection':inspection,'data_counts':counts,'common_hash':state_sha256(common)},indent=2),flush=True)

if __name__=='__main__':main()
