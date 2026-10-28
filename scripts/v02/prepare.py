"""CPU verification and row-norm controls only; never regenerate V0.1 state."""
import hashlib,json,math,time,traceback
import torch
from common import CFG,ROOT,CODE_ROOT,RUN,REPORT,PRIVATE,utc,sha,read_json,write_json,verify_snapshot,verify_inputs,input_specs
from tables import build_tables,tables_check,describe_tables
from vocabpriorbench.model import state_sha256,tensor_sha256,next_token_labels,unique_parameter_count
from vocabpriorbench.v01_model import OutputStudent,common_state_dict,load_common_state_dict

def main():
    start=time.perf_counter();torch.set_num_threads(4);snapshot=verify_snapshot()
    try:
        specs=verify_inputs(False)
        previous_environment=read_json(ROOT/'runs/v01_20261022/environment.json')
        assert torch.__version__==previous_environment['torch'] and torch.version.cuda==previous_environment['cuda_build'],'PyTorch/CUDA environment differs from V0.1'
        codes=torch.load(ROOT/CFG['reuse']['codes'],map_location='cpu',weights_only=True)
        packed=torch.load(ROOT/CFG['reuse']['packed'],map_location='cpu',weights_only=True)
        prior=torch.load(ROOT/CFG['reuse']['prior'],map_location='cpu',weights_only=True)
        selection=read_json(ROOT/CFG['reuse']['selection'])
        assert set(packed)=={'train','dev'} and set(prior)=={'counts','pi','dev_counts'}
        assert codes['input'].shape==codes['output'].shape==(50304,128)
        assert codes['input'].dtype==codes['output'].dtype==torch.float32
        selected=selection['selected_block_indices'];assert len(selected)==len(set(selected))==3068
        assert packed['train']['mask'][selected].all()
        labels=next_token_labels(packed['train']['ids'][selected],packed['train']['mask'][selected])
        counts=torch.bincount(labels[labels!=-100],minlength=50304)
        assert torch.equal(counts,prior['counts']) and int(counts.sum())==389636
        expected_pi=(counts.double()+.1)/(389636+.1*50304)
        assert torch.equal(prior['pi'],expected_pi)
        devlabels=next_token_labels(packed['dev']['ids'],packed['dev']['mask'])
        devcounts=torch.bincount(devlabels[devlabels!=-100],minlength=50304)
        assert torch.equal(devcounts,prior['dev_counts']) and int(devcounts.sum())==62601
        tables,r=build_tables(codes['output']);check=tables_check(tables,r,source_C=codes['output'])
        assert check['passed'],check
        description=describe_tables(tables,r,pi=prior['pi'],counts=counts,special_token_ids={'eos':0},tokenizer_length=50277,source_C=codes['output'])
        # Both anchors must match the actual V0.1 initial table hashes, for both seeds.
        seed_proofs={}
        for seed in CFG['seeds']:
            init_path=ROOT/CFG['reuse']['common_init_template'].format(seed=seed)
            initial=torch.load(init_path,map_location='cpu',weights_only=True)
            initial_hash=state_sha256(initial);expected=selection['seeds'][str(seed)]
            assert initial_hash==expected['common_init_state_sha256'],f'{init_path}: common state SHA mismatch'
            order_path=ROOT/CFG['reuse']['block_order_template'].format(seed=seed)
            order=read_json(order_path);assert sha(order_path)==expected['block_order_sha256'],f'{order_path}: order SHA mismatch'
            assert len(order)==len(set(order))==3068 and set(order)==set(selected),f'{order_path}: not a complete selected-set permutation'
            logpi=prior['pi'].log().float();models=[];checks={}
            for condition in CFG['conditions']:
                model=OutputStudent(codes['input'],tables[condition],False,logpi,**CFG['student']);load_common_state_dict(model,initial)
                assert unique_parameter_count(model,True)==444928
                before={name:tensor_sha256(getattr(model,name)) for name in ['input_codes','output_codes','output_bias']}
                assert before['output_codes']==tensor_sha256(tables[condition])
                assert state_sha256(common_state_dict(model))==initial_hash
                if condition in ['C','S']:
                    old=read_json(ROOT/f"runs/v01_20261022/resources_{seed}_{'Fp' if condition=='C' else 'Sp'}.json")
                    assert before==old['before_hashes']
                    assert initial_hash==old['common_initial_state_sha256']
                    assert sha(order_path)==old['block_order_sha256']
                checks[condition]={'initial_hashes':before,'common_state_sha256':initial_hash,'trainable_parameters':444928}
                models.append(model)
            for name in ['input_codes','output_codes','output_bias']:
                assert len({getattr(m,name).data_ptr() for m in models})==4
            assert len({checks[c]['initial_hashes']['input_codes'] for c in checks})==1
            assert len({checks[c]['initial_hashes']['output_bias'] for c in checks})==1
            seed_proofs[str(seed)]={'common_init_path':str(init_path.relative_to(ROOT)),'common_state_sha256':initial_hash,'order_path':str(order_path.relative_to(ROOT)),
                                    'order_sha256':sha(order_path),'same_selected_set':True,'independent_storage':True,'conditions':checks}
            del models
        table_path=PRIVATE/'tables.pt';torch.save(tables,table_path)
        description.update(source_C_file_sha256=sha(ROOT/CFG['reuse']['codes']),row_norm_source='CPU float64 L2 norm of original initial C only',
                           all_50304_classes_retained=True,prior_scaled_by_r=False,seed_proofs=seed_proofs)
        write_json(REPORT/'TABLE_CONSTRUCTION.json',description)
        write_json(RUN/'input_proof.json',{'status':'PASS','seeds':seed_proofs,'selected_indices_sha256':hashlib.sha256(json.dumps(selected,separators=(',',':')).encode()).hexdigest(),
                                        'train_targets':389636,'dev_targets':62601,'prior_recomputed_for_validation_only':True,'reused_original_prior_file':CFG['reuse']['prior'],
                                        'no_new_random_initialization_or_permutation':True,'holdout_opened_or_hashed':False})
        derived=[table_path,REPORT/'TABLE_CONSTRUCTION.json',RUN/'input_proof.json']
        write_json(REPORT/'source_manifest.json',{'version':'V0.2','run_name':CFG['run_name'],'created_utc':utc(),'snapshot_tree_sha256':snapshot['tree_sha256'],
            'instruction_sha256':sha(CODE_ROOT/'CODEX_V02_VocabPriorBench.txt'),'config_sha256':sha(CODE_ROOT/'configs/v02.yaml'),
            'contract_sha256':sha(CODE_ROOT/'reports/v02/EXPERIMENT_CONTRACT.md'),'input_artifacts':specs,
            'derived_artifacts':{p.relative_to(ROOT).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in derived},
            'model_dependencies_copied_verbatim':snapshot['model_dependency_sources'],'no_final_checkpoint_loaded':True,
            'no_holdout_access_or_hash':True,'no_download_or_environment_change':True,'prepare_elapsed_seconds':time.perf_counter()-start})
        write_json(RUN/'prepare_result.json',{'status':'PASS','seconds':time.perf_counter()-start,'train_targets':389636,'dev_targets':62601,'seed_proofs':seed_proofs})
        print({'status':'PASS','seconds':time.perf_counter()-start,'tables_file_sha256':sha(table_path)},flush=True)
    except BaseException as exc:
        failure={'status':'BLOCKED_MISSING_OR_MISMATCHED_V01_INPUT','type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'gpu_seconds':0}
        write_json(REPORT/'input_failure.json',failure);print(failure,flush=True);raise

if __name__=='__main__':main()
