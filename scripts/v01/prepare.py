"""One train-count pass, fixed block set, independent saved initializations. CPU only."""
import math,time
import torch
from common import CFG,ROOT,CODE_ROOT,RUN,REPORT,PRIVATE,sha,read_json,write_json,utc,verify_snapshot
from vocabpriorbench.model import Student,common_state_dict,state_sha256,next_token_labels
from vocabpriorbench.v01_model import OutputStudent,load_common_state_dict

def main():
    start=time.perf_counter();torch.set_num_threads(4);snapshot=verify_snapshot()
    v0=read_json(ROOT/'reports/source_manifest.json')
    # Explicit allowlist: never open the holdout file, even for hashing.
    paths=['artifacts/codes.pt','artifacts/transforms.pt','artifacts/common_init.pt','data/packed.pt','data/manifest.json',
           'artifacts/donor/config.json','artifacts/donor/tokenizer.json','artifacts/donor/tokenizer_config.json','artifacts/donor/special_tokens_map.json']
    expected={v0['codes']['path'].replace('\\','/'):v0['codes']['sha256'],v0['transforms']['path'].replace('\\','/'):v0['transforms']['sha256'],
              v0['common_init']['path'].replace('\\','/'):v0['common_init']['file_sha256'],
              **{p.replace('\\','/'):h for p,h in v0['data_artifact_hashes'].items() if 'holdout' not in p}}
    expected.update({item['path'].replace('\\','/'):item['sha256'] for item in v0['donor_files']})
    artifacts={}
    for name in paths:
        digest=sha(ROOT/name);assert digest==expected[name],name
        artifacts[name]={'sha256':digest,'bytes':(ROOT/name).stat().st_size}
    packed=torch.load(ROOT/'data/packed.pt',map_location='cpu',weights_only=True)
    assert set(packed)=={'train','dev'}
    full=(packed['train']['mask'].sum(1)==128).nonzero().flatten().tolist()
    count=(len(full)//4)*4;selected=full[:count]
    assert count==CFG['data']['selected_blocks']==3068
    assert count//4==CFG['training']['optimizer_steps']==767
    discarded=[{'block':i,'valid_input_tokens':int(packed['train']['mask'][i].sum()),
                'reason':'incomplete_block' if i not in full else 'complete_tail_not_divisible_by_effective_batch_4'}
               for i in range(len(packed['train']['ids'])) if i not in set(selected)]
    ts=time.perf_counter();labels=next_token_labels(packed['train']['ids'][selected],packed['train']['mask'][selected])
    counts=torch.bincount(labels[labels!=-100],minlength=CFG['vocabulary'])
    N=int(counts.sum());assert N==count*127==389636
    alpha=CFG['prior']['alpha'];pi=(counts.double()+alpha)/(N+alpha*CFG['vocabulary'])
    count_seconds=time.perf_counter()-ts
    devlabels=next_token_labels(packed['dev']['ids'],packed['dev']['mask'])
    dev_counts=torch.bincount(devlabels[devlabels!=-100],minlength=CFG['vocabulary'])
    assert int(dev_counts.sum())==62601
    codes=torch.load(ROOT/'artifacts/codes.pt',map_location='cpu',weights_only=True)
    assert codes['input'].shape==codes['output'].shape==(50304,128)
    torch.save({'counts':counts,'pi':pi,'dev_counts':dev_counts},PRIVATE/'prepared.pt')
    selection={'selected_block_indices':selected,'discarded_blocks':discarded,'selected_blocks':count,
               'train_valid_targets':N,'dev_blocks':len(packed['dev']['ids']),'dev_valid_targets':int(dev_counts.sum()),
               'prior_training_statistics_passes':1,'prior_training_statistics_targets':N,'prior_count_seconds':count_seconds,
               'prior_available_at_step':0,'prior_alpha':alpha,'holdout_opened':False,
               'document_resampling_supported':False,'document_mapping_reason':'existing packed.pt has no reliable per-target document mapping; documents concatenated with EOS',
               'seeds':{}}
    for seed in CFG['seeds']:
        torch.manual_seed(seed)
        model=Student(codes['input'],codes['output'],False,**CFG['student'])
        common=common_state_dict(model);path=PRIVATE/f'common_init_{seed}.pt';torch.save(common,path)
        generator=torch.Generator().manual_seed(seed)
        order=[selected[i] for i in torch.randperm(count,generator=generator).tolist()]
        order_path=RUN/f'block_order_{seed}.json';write_json(order_path,order)
        selection['seeds'][str(seed)]={'common_init_state_sha256':state_sha256(common),'common_init_file_sha256':sha(path),
                                      'block_order_sha256':sha(order_path),'order_file':str(order_path.relative_to(ROOT))}
        checks={}
        for bias_name,bias in [('zero',torch.zeros(50304)),('prior',pi.log().float())]:
            a=OutputStudent(codes['input'],codes['output'],True,bias,**CFG['student'])
            b=OutputStudent(codes['input'],codes['output'],False,bias,**CFG['student'])
            load_common_state_dict(a,common);load_common_state_dict(b,common)
            with torch.no_grad():difference=float((a(packed['train']['ids'][:1,:8])-b(packed['train']['ids'][:1,:8])).abs().max())
            assert difference==0;checks[bias_name]=difference
            del a,b
        selection['seeds'][str(seed)]['actual_vocabulary_L_F_initial_forward_max_difference']=checks
        del model
    assert selection['seeds']['1729']['common_init_state_sha256']!=selection['seeds']['2718']['common_init_state_sha256']
    write_json(REPORT/'data_selection.json',selection)
    prep_paths=[PRIVATE/'prepared.pt',PRIVATE/'common_init_1729.pt',PRIVATE/'common_init_2718.pt',REPORT/'data_selection.json',RUN/'block_order_1729.json',RUN/'block_order_2718.json']
    write_json(REPORT/'source_manifest.json',{'version':'V0.1','created_utc':utc(),'instruction_sha256':sha(CODE_ROOT/'CODEX_V01_VocabPriorBench.txt'),
        'snapshot_tree_sha256':snapshot['tree_sha256'],'snapshot_manifest':str((CODE_ROOT/'SNAPSHOT_MANIFEST.json').relative_to(ROOT)),
        'config_sha256':sha(CODE_ROOT/'configs/v01.yaml'),'contract_sha256':sha(CODE_ROOT/'reports/v01/EXPERIMENT_CONTRACT.md'),
        'input_artifacts':artifacts,'prepared_artifacts':{str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in prep_paths},
        'donor':v0['donor'],'tokenizer_order_sha256':v0['tokenizer_order_sha256'],'source_data':v0['data'],
        'no_downloads':True,'holdout_accessed':False,'cpu_prepare_wall_seconds':time.perf_counter()-start})
    print({'event':'prepared','selected_blocks':count,'optimizer_steps':count//4,'valid_targets':N,'dev_targets':int(dev_counts.sum()),'discarded':discarded,'seconds':time.perf_counter()-start},flush=True)

if __name__=='__main__':main()
