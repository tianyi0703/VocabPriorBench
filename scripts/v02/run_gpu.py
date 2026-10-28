"""Eight frozen-output norm/direction trajectories; unchanged V0.1 numerical loop."""
import gc,json,math,os,platform,subprocess,sys,threading,time,traceback
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['HF_HUB_OFFLINE']='1'
import psutil,torch
from torch.nn import functional as F
from common import CFG,ROOT,CODE_ROOT,RUN,REPORT,PRIVATE,utc,sha,read_json,write_json,verify_snapshot,verify_inputs
from vocabpriorbench.model import next_token_labels,state_sha256,tensor_sha256,unique_parameter_count
from vocabpriorbench.v01_model import OutputStudent,common_state_dict,load_common_state_dict

def smi():
    try:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total,memory.used,memory.free,driver_version','--format=csv,noheader,nounits'],text=True,timeout=5)
        return dict(zip(['name','uuid','total_mib','used_mib','free_mib','driver'],raw.strip().splitlines()[0].split(', ')))
    except Exception as exc:return {'error':str(exc)}

class Monitor:
    def __init__(self):self.done=threading.Event();self.samples=[];self.rss_peak=0;self.device_peak=0
    def sample(self):
        rss=psutil.Process().memory_info().rss;device=smi();self.rss_peak=max(self.rss_peak,rss)
        self.device_peak=max(self.device_peak,float(device.get('used_mib',0)));self.samples.append({'utc':utc(),'rss_bytes':rss,'device_global':device})
    def work(self):
        while not self.done.is_set():self.sample();self.done.wait(1)
    def start(self):self.thread=threading.Thread(target=self.work,daemon=True);self.thread.start()
    def close(self):self.done.set();self.thread.join(timeout=7);self.sample()

def ce(logits,labels):return F.cross_entropy(logits.flatten(0,1),labels.flatten(),ignore_index=-100,reduction='sum')

@torch.no_grad()
def evaluate(model,dev):
    model.eval();numerator=0.;denominator=0;torch.cuda.synchronize();start=time.perf_counter()
    for offset in range(0,len(dev['ids']),2):
        ids=dev['ids'][offset:offset+2].cuda();mask=dev['mask'][offset:offset+2].cuda();labels=next_token_labels(ids,mask)
        with torch.autocast('cuda',dtype=torch.bfloat16):value=ce(model(ids,mask),labels)
        assert torch.isfinite(value);numerator+=float(value);denominator+=int((labels!=-100).sum())
    torch.cuda.synchronize();model.train();assert denominator==62601
    return {'nll':numerator/denominator,'nll_sum':numerator,'valid_targets':denominator,'seconds':time.perf_counter()-start,'blocks':len(dev['ids'])}

def main():
    if (RUN/'events.jsonl').exists():raise RuntimeError('Refuse overwrite or automatic rerun')
    if not (RUN/'numerics.log').exists() or '\nOK' not in (RUN/'numerics.log').read_text(encoding='utf-8',errors='replace'):raise RuntimeError('No passing sealed numerical checks')
    start=time.perf_counter();log=open(RUN/'events.jsonl','x',encoding='utf-8',buffering=1)
    def event(kind,**values):
        row={'utc':utc(),'event':kind,**values};log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');print(json.dumps(row,ensure_ascii=False,allow_nan=False),flush=True)
    deadline=float(os.environ['V02_GPU_DEADLINE_EPOCH'])
    def guard():
        if time.time()>deadline:raise TimeoutError('GPU cleanup reserve reached')
    profiles=[];metrics=[];errors=[];active=None;monitor=None
    result={'status':'RUNNING','runner_start_utc':utc(),'trajectories':profiles,'errors':errors,
        'device_memory_scope':'nvidia-smi entire device includes desktop and other apps','cpu_memory_scope':'process RSS sampled about 1s plus nvidia-smi latency',
        'trajectory_throughput_scope':'observed short-run synchronized step timings, no sustained thermal claim',
        'new_gpu_process_wall_seconds':None,'previous_project_gpu_process_wall_seconds':332.12303610004624}
    try:
        verify_snapshot();verify_inputs();torch.set_num_threads(4)
        assert torch.cuda.is_available() and '4060' in torch.cuda.get_device_name(0) and torch.cuda.device_count()==1
        assert torch.cuda.is_bf16_supported();torch.backends.cuda.matmul.allow_tf32=False
        env={'utc':utc(),'platform':platform.platform(),'python':sys.version,'executable':sys.executable,'torch':torch.__version__,
             'cuda_build':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'device':torch.cuda.get_device_name(0),'device_initial':smi(),'memory':psutil.virtual_memory()._asdict()}
        write_json(RUN/'environment.json',env);event('environment',**env)
        assert float(env['device_initial']['free_mib'])>=1500,'Insufficient headroom for safe small preflight'
        codes=torch.load(ROOT/'artifacts/codes.pt',map_location='cpu',weights_only=True)
        packed=torch.load(ROOT/'data/packed.pt',map_location='cpu',weights_only=True)
        prior=torch.load(ROOT/CFG['reuse']['prior'],map_location='cpu',weights_only=True);logpi=prior['pi'].log().float()
        tables=torch.load(PRIVATE/'tables.pt',map_location='cpu',weights_only=True)
        proof=read_json(RUN/'input_proof.json')
        assert set(tables)=={'C','U','RS','S'}
        dev_unigram_nll=float(-(prior['dev_counts'].double()*prior['pi'].log()).sum()/prior['dev_counts'].sum())
        selection=read_json(ROOT/CFG['reuse']['selection'])
        event('job_start',config=CFG,config_sha256=sha(CODE_ROOT/'configs/v02.yaml'),snapshot=read_json(CODE_ROOT/'SNAPSHOT_MANIFEST.json')['tree_sha256'],dev_unigram_nll=dev_unigram_nll)
        # Exactly one synthetic shape check, no optimizer and no corpus replay.
        guard();probe=OutputStudent(codes['input'],tables['C'],False,logpi,**CFG['student']).cuda()
        ids=torch.arange(32,device='cuda').remainder(50304)[None];labels=next_token_labels(ids)
        with torch.autocast('cuda',dtype=torch.bfloat16):value=ce(probe(ids),labels)/31
        value.backward();torch.cuda.synchronize();assert torch.isfinite(value)
        event('gpu_preflight',output_type='C',microbatch=1,sequence_length=32,optimizer_steps=0,synthetic_targets=31,allocated_bytes=torch.cuda.memory_allocated())
        del probe,ids,labels,value;gc.collect();torch.cuda.empty_cache()
        for seed in CFG['seeds']:
            common=torch.load(ROOT/CFG['reuse']['common_init_template'].format(seed=seed),map_location='cpu',weights_only=True)
            order_path=ROOT/CFG['reuse']['block_order_template'].format(seed=seed)
            order=read_json(order_path);assert len(order)==len(set(order))==3068
            assert set(order)==set(selection['selected_block_indices'])
            assert state_sha256(common)==selection['seeds'][str(seed)]['common_init_state_sha256']
            for condition in CFG['run_order'][str(seed)]:
                guard();trajectory_start_utc=utc();trajectory_start=time.perf_counter()
                monitor=Monitor();monitor.start()
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                out=tables[condition]
                bias=logpi
                model=OutputStudent(codes['input'],out,False,bias,**CFG['student']);load_common_state_dict(model,common)
                assert state_sha256(common_state_dict(model))==state_sha256(common)
                before={name:tensor_sha256(getattr(model,name)) for name in ['input_codes','output_codes','output_bias']}
                assert before==proof['seeds'][str(seed)]['conditions'][condition]['initial_hashes']
                model.cuda();torch.cuda.reset_peak_memory_stats()
                parameters=[p for p in model.parameters() if p.requires_grad]
                optimizer=torch.optim.AdamW(parameters,lr=3e-4,weight_decay=.01,betas=(.9,.999),eps=1e-8,fused=True)
                active={'seed':seed,'condition':condition,'status':'RUNNING','start_utc':trajectory_start_utc,'start_monotonic_seconds':trajectory_start,'steps':0,'train_valid_targets':0,
                    'common_initial_state_sha256':state_sha256(common),'block_order_sha256':sha(order_path),
                    'parameters_unique':unique_parameter_count(model),'parameters_trainable':unique_parameter_count(model,True),
                    'before_hashes':before,'code_dtype':str(model.input_codes.dtype),'bit_packed':False,'code_bytes':2*model.input_codes.numel()*4,'bias_bytes':model.output_bias.numel()*4,
                    'evaluations':[]}
                profiles.append(active);event('trajectory_start',**active)
                total_loss=0.;total_tokens=0;train_seconds=0.;eval_seconds=0.;pointer=0;grad_seen={}
                def evaluation(step):
                    nonlocal eval_seconds
                    value=evaluate(model,packed['dev']);eval_seconds+=value['seconds']
                    row={'seed':seed,'condition':condition,'step':step,'train_valid_targets':total_tokens,'train_online_nll_sum':total_loss,
                         'train_online_nll':total_loss/total_tokens if total_tokens else None,'nll':value['nll'],'nll_sum':value['nll_sum'],
                         'dev_valid_targets':value['valid_targets'],'dev_unigram_nll':dev_unigram_nll,'context_gain':dev_unigram_nll-value['nll'],
                         'eval_seconds':value['seconds']}
                    metrics.append(row);active['evaluations'].append(row);event('evaluation',**row);write_json(RUN/'metrics_raw.json',metrics)
                evaluation(0)
                for step in range(1,768):
                    guard();torch.cuda.synchronize();ts=time.perf_counter();optimizer.zero_grad(set_to_none=True);step_sum=0.;step_targets=0
                    for _ in range(2):
                        indices=order[pointer:pointer+2];assert len(indices)==2;pointer+=2
                        ids=packed['train']['ids'][indices].cuda();mask=packed['train']['mask'][indices].cuda();labels=next_token_labels(ids,mask)
                        n=int((labels!=-100).sum());assert n==254
                        with torch.autocast('cuda',dtype=torch.bfloat16):value=ce(model(ids,mask),labels)
                        assert torch.isfinite(value);(value/508).backward();step_sum+=float(value);step_targets+=n
                    for name,p in model.named_parameters():
                        if p.requires_grad:
                            assert p.grad is not None and torch.isfinite(p.grad).all(),name
                            if step==1:grad_seen[name]=float(p.grad.abs().max())
                        else:assert p.grad is None,name
                    torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step()
                    assert all(torch.isfinite(p).all() for p in parameters)
                    torch.cuda.synchronize();seconds=time.perf_counter()-ts;train_seconds+=seconds;total_loss+=step_sum;total_tokens+=step_targets
                    active.update(steps=step,train_valid_targets=total_tokens)
                    event('step',seed=seed,condition=condition,step=step,nll_sum=step_sum,valid_targets=step_targets,nll=step_sum/step_targets,cumulative_targets=total_tokens,seconds=seconds)
                    if step in [100,767]:evaluation(step)
                assert pointer==3068 and total_tokens==389636
                after={name:tensor_sha256(getattr(model,name)) for name in before}
                assert before['input_codes']==after['input_codes'] and before['output_bias']==after['output_bias']
                assert before['output_codes']==after['output_codes']
                final_common=common_state_dict(model);updated={name:not torch.equal(common[name],value.cpu()) for name,value in final_common.items()}
                assert all(updated.values()) and all(x>0 for x in grad_seen.values())
                torch.cuda.synchronize();allocated=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
                checkpoint=PRIVATE/f'final_{seed}_{condition}.pt';torch.save({name:value.detach().cpu() for name,value in model.state_dict().items()},checkpoint)
                monitor.close();write_json(RUN/f'monitor_{seed}_{condition}.json',monitor.samples)
                active.update(status='COMPLETED',end_utc=utc(),wall_seconds=time.perf_counter()-trajectory_start,train_seconds=train_seconds,eval_seconds=eval_seconds,
                    train_online_nll_sum=total_loss,train_online_nll=total_loss/total_tokens,valid_targets_per_train_second=total_tokens/train_seconds,
                    torch_peak_allocated_bytes=allocated,torch_peak_reserved_bytes=reserved,device_global_peak_used_mib=monitor.device_peak,cpu_sampled_peak_rss_bytes=monitor.rss_peak,
                    after_hashes=after,all_common_parameters_updated=True,first_step_nonzero_gradient=True,
                    checkpoint_path=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=sha(checkpoint),checkpoint_bytes=checkpoint.stat().st_size)
                write_json(RUN/f'resources_{seed}_{condition}.json',active);write_json(REPORT/'resource_profile.json',result)
                event('trajectory_end',**{k:v for k,v in active.items() if k!='evaluations'})
                active=None;monitor=None
                del model,optimizer,parameters,out,bias,final_common,ids,mask,labels,value;gc.collect();torch.cuda.empty_cache()
        result['status']='COMPLETED'
    except BaseException as exc:
        errors.append({'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()});event('error',**errors[-1])
        result['status']='BUDGET_REACHED' if isinstance(exc,TimeoutError) else 'FAILED'
        if active is not None:
            active.update(status='FAILED',end_utc=utc(),wall_seconds=time.perf_counter()-trajectory_start,
                torch_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
                torch_peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None)
            write_json(RUN/f"resources_{active['seed']}_{active['condition']}.json",active)
        raise
    finally:
        if monitor:
            monitor.close()
            if active:write_json(RUN/f"monitor_{active['seed']}_{active['condition']}.json",monitor.samples)
        result.update(runner_end_utc=utc(),runner_elapsed_seconds=time.perf_counter()-start,oom_count=sum('out of memory' in e['message'].lower() for e in errors))
        write_json(REPORT/'resource_profile.json',result);event('runner_end',status=result['status'],runner_elapsed_seconds=result['runner_elapsed_seconds']);log.close()

if __name__=='__main__':main()
