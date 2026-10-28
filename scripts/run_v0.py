"""One sequential local CUDA job. Resource, prefix, and initialization audit included."""
import contextlib, gc, json, math, os, platform, subprocess, sys, threading, time, traceback
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_OFFLINE']='1'
import psutil
import torch
import yaml
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vocabpriorbench.model import (Student, common_state_dict,load_common_state_dict,
    next_token_labels,state_sha256,tensor_sha256,unique_parameter_count)
from prepare import sha,write_json

def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())

def smi():
    try:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total,memory.used,memory.free,driver_version,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True,timeout=5)
        values=raw.strip().splitlines()[0].split(', ')
        return dict(zip(['name','uuid','total_mib','used_mib','free_mib','driver','temperature_c','power_w'],values))
    except Exception as e:return {'error':str(e)}

class Monitor:
    def __init__(self):self.stop=threading.Event();self.cpu_peak=0;self.device_peak=0;self.samples=[]
    def sample(self):
        proc=psutil.Process();rss=proc.memory_info().rss;self.cpu_peak=max(self.cpu_peak,rss)
        device=smi()
        try:self.device_peak=max(self.device_peak,float(device['used_mib']))
        except (KeyError,ValueError):pass
        self.samples.append({'utc':utc(),'rss_bytes':rss,'device':device})
    def work(self):
        while not self.stop.is_set():self.sample();self.stop.wait(.5)
    def start(self):self.thread=threading.Thread(target=self.work,daemon=True);self.thread.start()
    def close(self):self.stop.set();self.thread.join(timeout=7);self.sample()

def loss_sum(logits,labels):
    # Full logits, fused PyTorch CE. No chunked or approximate softmax.
    return F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),ignore_index=-100,reduction='sum')

@torch.no_grad()
def evaluate(model,part,blocks,microbatch):
    model.eval();total=0.;tokens=0;torch.cuda.synchronize();start=time.perf_counter()
    for i in range(0,min(blocks,len(part['ids'])),microbatch):
        ids=part['ids'][i:i+microbatch].cuda();mask=part['mask'][i:i+microbatch].cuda()
        labels=next_token_labels(ids,mask)
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(ids,mask);value=loss_sum(logits,labels)
        assert torch.isfinite(value)
        total+=value.item();tokens+=(labels!=-100).sum().item()
    torch.cuda.synchronize()
    model.train()
    return {'nll':total/tokens,'nll_sum':total,'valid_tokens':tokens,'seconds':time.perf_counter()-start,'blocks':min(blocks,len(part['ids']))}

def main():
    cfg=yaml.safe_load((ROOT/'configs/v0.yaml').read_text());train=cfg['training']
    run_dir=ROOT/'runs/v0_20261021'
    run_dir.mkdir(parents=True,exist_ok=True)
    if (run_dir/'events.jsonl').exists():raise RuntimeError('Existing job logs: refuse overwrite or unbudgeted repeat')
    oldpath=ROOT/'reports/gpu_budget.json'
    prior=json.loads(oldpath.read_text()) if oldpath.exists() else {'jobs':[],'cumulative_seconds':0}
    if train['estimated_job_seconds']>train['gpu_wall_limit_seconds']-prior['cumulative_seconds']:raise RuntimeError('Insufficient remaining hard budget')
    # All CPU-only correctness checks must already have succeeded.
    check=ROOT/'runs/numerical_checks.log'
    if not check.exists() or '\nOK' not in check.read_text(encoding='utf-8',errors='replace'):raise RuntimeError('Missing passing CPU numerical checks')
    if not (ROOT/'reports/EXPERIMENT_CONTRACT.md').exists():raise RuntimeError('Missing contract')
    subprocess.check_call([sys.executable,'-m','py_compile',str(ROOT/'scripts/run_v0.py')])
    t0=time.perf_counter();start_utc=utc();prior_seconds=prior['cumulative_seconds']
    job={'start_utc':start_utc,'end_utc':None,'wall_seconds':0,'status':'RUNNING','run_dir':str(run_dir),'command':' .venv\\Scripts\\python.exe -u scripts\\run_v0.py'}
    prior['jobs'].append(job)
    log=open(run_dir/'events.jsonl','w',encoding='utf-8',buffering=1)
    def event(kind,**kw):
        row={'utc':utc(),'event':kind,**kw};log.write(json.dumps(row,ensure_ascii=False)+'\n');print(json.dumps(row,ensure_ascii=False),flush=True)
    def budget():
        elapsed=time.perf_counter()-t0;job['wall_seconds']=elapsed;prior['cumulative_seconds']=prior_seconds+elapsed
        write_json(oldpath,prior)
        if prior['cumulative_seconds']>train['gpu_wall_limit_seconds']-100:raise RuntimeError('Hard-budget cleanup reserve reached')
    # Secondary watchdog; supervise_v0.py enforces the external process timeout.
    def watchdog():
        remaining=train['gpu_wall_limit_seconds']-prior_seconds-100
        if not watchdog_stop.wait(remaining):
            job.update(status='WATCHDOG_TERMINATED',end_utc=utc(),wall_seconds=time.perf_counter()-t0)
            prior['cumulative_seconds']=prior_seconds+job['wall_seconds'];write_json(oldpath,prior);os._exit(124)
    watchdog_stop=threading.Event();threading.Thread(target=watchdog,daemon=True).start()
    profiles=[];errors=[];monitor=None;active_profile=None;active_arm=None;active_stage='initialization'
    try:
        assert torch.cuda.is_available(),'CUDA unavailable'
        assert '4060' in torch.cuda.get_device_name(0),'Non-target device'
        assert torch.cuda.device_count()==1,'Expected exactly one local CUDA GPU'
        torch.set_num_threads(4);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
        torch.backends.cuda.matmul.allow_tf32=False
        env={'platform':platform.platform(),'python':sys.version,'executable':sys.executable,
             'torch':torch.__version__,'cuda_build':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
             'device':torch.cuda.get_device_name(0),'capability':torch.cuda.get_device_capability(),
             'bf16_supported':torch.cuda.is_bf16_supported(),'system_ram':psutil.virtual_memory()._asdict(),'device_initial':smi()}
        assert env['bf16_supported'];write_json(run_dir/'environment.json',env);event('environment',**env)
        if float(env['device_initial']['free_mib'])<1500:raise RuntimeError('Less than 1.5GiB free; no safe preflight headroom')
        codes=torch.load(ROOT/'artifacts/codes.pt',map_location='cpu',weights_only=True)
        common=torch.load(ROOT/'artifacts/common_init.pt',map_location='cpu',weights_only=True)
        packed=torch.load(ROOT/'data/packed.pt',map_location='cpu',weights_only=True)
        init_hash=state_sha256(common);used=train['optimizer_steps']*train['microbatch']*train['gradient_accumulation']
        assert len(packed['train']['ids'])>=used and packed['train']['mask'][:used].all()
        prefix_hash=state_sha256({k:v[:used] for k,v in packed['train'].items()})
        source=json.loads((ROOT/'reports/source_manifest.json').read_text(encoding='utf-8'))
        assert source['config_sha256']==sha(ROOT/'configs/v0.yaml'),'Config changed since preparation'
        for spec in [source['codes'],source['transforms']]:assert sha(ROOT/spec['path'])==spec['sha256']
        assert sha(ROOT/source['common_init']['path'])==source['common_init']['file_sha256']
        for path,digest in source['data_artifact_hashes'].items():assert sha(ROOT/path)==digest,path
        assert init_hash==source['common_init']['state_sha256']
        assert codes['input'].shape==codes['output'].shape==(50304,cfg['code_width'])
        assert packed['train']['ids'].shape[1]==train['sequence_length']
        source['code_files_before_gpu']={str(p.relative_to(ROOT)):sha(p) for folder in ['src','scripts','tests','configs'] for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)}
        write_json(ROOT/'reports/source_manifest.json',source)
        # The actual full-vocabulary A/B forward must agree before any optimization.
        with torch.no_grad():
            a=Student(codes['input'],codes['output'],True,**cfg['student']);load_common_state_dict(a,common)
            b=Student(codes['input'],codes['output'],False,**cfg['student']);load_common_state_dict(b,common)
            probe=packed['train']['ids'][:1,:8]
            delta=(a(probe)-b(probe)).abs().max().item();assert delta==0
            event('actual_a_b_cpu_forward_check',max_abs_difference=delta,common_hash=init_hash)
            del a,b
        # Small-batch and short-sequence preflight for all arms, no optimizer updates.
        for arm in train['arms']:
            active_arm=arm;active_stage='short_sequence_preflight'
            budget();ci,co=codes['input'],codes['output']
            if arm=='C':ci=torch.where(ci>=0,1.,-1.)/math.sqrt(cfg['code_width']);co=torch.where(co>=0,1.,-1.)/math.sqrt(cfg['code_width'])
            model=Student(ci,co,arm=='A',**cfg['student']);load_common_state_dict(model,common);model.cuda()
            ids=packed['train']['ids'][:1,:32].cuda();labels=next_token_labels(ids)
            with torch.autocast('cuda',dtype=torch.bfloat16):value=loss_sum(model(ids),labels)/31
            value.backward();torch.cuda.synchronize();assert torch.isfinite(value)
            event('preflight',arm=arm,batch=1,sequence=32,optimizer_steps=0,loss=value.item(),allocated=torch.cuda.memory_allocated())
            del model,ci,co,ids,labels,value;gc.collect();torch.cuda.empty_cache()
        for arm in train['arms']:
            active_arm=arm;active_stage='full_arm'
            budget();arm_start=time.perf_counter();monitor=Monitor();monitor.start();device_before=smi()
            ci,co=codes['input'],codes['output']
            if arm=='C':ci=torch.where(ci>=0,1.,-1.)/math.sqrt(cfg['code_width']);co=torch.where(co>=0,1.,-1.)/math.sqrt(cfg['code_width'])
            model=Student(ci,co,arm=='A',**cfg['student']);load_common_state_dict(model,common)
            assert state_sha256(common_state_dict(model))==init_hash
            hashes_before={'input':tensor_sha256(model.input_codes),'output':tensor_sha256(model.output_codes)}
            model.cuda();torch.cuda.reset_peak_memory_stats()
            parameters=[p for p in model.parameters() if p.requires_grad]
            optimizer=torch.optim.AdamW(parameters,lr=train['learning_rate'],weight_decay=train['weight_decay'],fused=True)
            profile={'arm':arm,'status':'RUNNING','start_utc':utc(),'common_init_sha256':init_hash,'training_prefix_sha256':prefix_hash,
                     'parameters_unique':unique_parameter_count(model),'parameters_trainable':unique_parameter_count(model,True),
                     'code_dtype':str(model.input_codes.dtype),'code_bit_packed':False,'code_gpu_bytes':2*model.input_codes.numel()*model.input_codes.element_size(),
                     'code_hashes_before':hashes_before,'device_before':device_before,'steps':0,'train_valid_tokens':0}
            profiles.append(profile);active_profile=profile
            event('arm_start',**profile)
            initial_train=evaluate(model,packed['train'],train['eval_blocks'],train['microbatch'])
            initial_dev=evaluate(model,packed['dev'],train['eval_blocks'],train['microbatch'])
            ids=packed['train']['ids'][:train['microbatch']].cuda();mask=packed['train']['mask'][:train['microbatch']].cuda();labels=next_token_labels(ids,mask)
            torch.cuda.synchronize();warm=time.perf_counter()
            for _ in range(train['warmup_forward_backward_only']):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):value=loss_sum(model(ids,mask),labels)/(labels!=-100).sum()
                value.backward()
            torch.cuda.synchronize();warm_seconds=time.perf_counter()-warm;optimizer.zero_grad(set_to_none=True)
            total_loss=0.;total_tokens=0;train_seconds=0.;stable_seconds=0.;stable_tokens=0;pointer=0;gradient_seen={}
            for step in range(1,train['optimizer_steps']+1):
                budget();torch.cuda.synchronize();ts=time.perf_counter();optimizer.zero_grad(set_to_none=True)
                step_sum=0.;step_tokens=0
                for _ in range(train['gradient_accumulation']):
                    ids=packed['train']['ids'][pointer:pointer+train['microbatch']].cuda();mask=packed['train']['mask'][pointer:pointer+train['microbatch']].cuda();pointer+=train['microbatch']
                    labels=next_token_labels(ids,mask);n=(labels!=-100).sum().item()
                    with torch.autocast('cuda',dtype=torch.bfloat16):value=loss_sum(model(ids,mask),labels)
                    assert torch.isfinite(value);(value/(n*train['gradient_accumulation'])).backward()
                    step_sum+=value.item();step_tokens+=n
                for name,p in model.named_parameters():
                    if p.requires_grad:
                        assert p.grad is not None and torch.isfinite(p.grad).all(),name
                        if step==1:gradient_seen[name]=float(p.grad.abs().max().item())
                    else:assert p.grad is None,name
                torch.nn.utils.clip_grad_norm_(parameters,train['max_grad_norm'],error_if_nonfinite=True);optimizer.step()
                assert all(torch.isfinite(p).all() for p in parameters)
                torch.cuda.synchronize();seconds=time.perf_counter()-ts;train_seconds+=seconds;total_loss+=step_sum;total_tokens+=step_tokens
                if step>=train['stable_timing_start_step']:stable_seconds+=seconds;stable_tokens+=step_tokens
                event('step',arm=arm,step=step,nll=step_sum/step_tokens,nll_sum=step_sum,valid_tokens=step_tokens,cumulative_tokens=total_tokens,seconds=seconds)
                profile.update(steps=step,train_valid_tokens=total_tokens)
            final_train=evaluate(model,packed['train'],train['eval_blocks'],train['microbatch']);final_dev=evaluate(model,packed['dev'],train['eval_blocks'],train['microbatch'])
            after_common=common_state_dict(model)
            updates={name:not torch.equal(common[name],after_common[name].cpu()) for name in common}
            assert all(updates.values()),updates
            assert all(x>0 for x in gradient_seen.values()),gradient_seen
            hashes_after={'input':tensor_sha256(model.input_codes),'output':tensor_sha256(model.output_codes)}
            assert all(hashes_before[k]!=hashes_after[k] for k in hashes_before) if arm=='A' else (hashes_before==hashes_after)
            torch.cuda.synchronize();allocated=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
            checkpoint=run_dir/f'final_{arm}.pt';torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},checkpoint)
            monitor.close()
            profile.update(status='COMPLETED',end_utc=utc(),wall_seconds=time.perf_counter()-arm_start,train_seconds=train_seconds,
                warmup_seconds=warm_seconds,eval_seconds=initial_train['seconds']+initial_dev['seconds']+final_train['seconds']+final_dev['seconds'],
                train_online_nll=total_loss/total_tokens,train_online_nll_sum=total_loss,initial_train=initial_train,initial_dev=initial_dev,final_train=final_train,final_dev=final_dev,
                stable_valid_tokens_per_second=stable_tokens/stable_seconds,stable_tokens=stable_tokens,stable_seconds=stable_seconds,
                train_valid_tokens_per_second=total_tokens/train_seconds,torch_peak_allocated_bytes=allocated,torch_peak_reserved_bytes=reserved,
                device_peak_used_mib_global=monitor.device_peak,cpu_peak_rss_bytes_sampled=monitor.cpu_peak,
                checkpoint_bytes=checkpoint.stat().st_size,checkpoint_sha256=sha(checkpoint),code_hashes_after=hashes_after,
                all_common_parameters_updated=all(updates.values()),all_trainable_parameters_nonzero_gradient=True,
                gradient_max_abs_step1=gradient_seen,common_final_sha256=state_sha256(after_common))
            write_json(run_dir/f'resources_{arm}.json',profile);write_json(run_dir/f'monitor_{arm}.json',monitor.samples)
            event('arm_end',**{k:v for k,v in profile.items() if k!='gradient_max_abs_step1'})
            monitor=None;active_profile=None
            del model,optimizer,parameters,ci,co,after_common,ids,mask,labels,value;gc.collect();torch.cuda.empty_cache()
        job['status']='COMPLETED'
    except BaseException as error:
        errors.append({'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(),'active_arm':active_arm,'active_stage':active_stage})
        if active_profile is not None:
            active_profile.update(status='FAILED',end_utc=utc(),wall_seconds=time.perf_counter()-arm_start)
            if torch.cuda.is_initialized():active_profile.update(torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            if monitor:
                active_profile.update(cpu_peak_rss_bytes_sampled=monitor.cpu_peak,device_peak_used_mib_global=monitor.device_peak)
                write_json(run_dir/f'monitor_{active_arm}.json',monitor.samples)
            write_json(run_dir/f'resources_{active_arm}.json',active_profile)
        event('error',**errors[-1]);job['status']='FAILED';raise
    finally:
        if monitor:monitor.close()
        watchdog_stop.set();job['end_utc']=utc();job['wall_seconds']=time.perf_counter()-t0
        prior['cumulative_seconds']=prior_seconds+job['wall_seconds'];write_json(oldpath,prior)
        report={'status':job['status'],'gpu_budget':prior,'config':cfg,'arms':profiles,'errors':errors,
            'oom_count':sum('out of memory' in e['message'].lower() for e in errors),
            'device_memory_scope':'nvidia-smi global device total incl. desktop/other applications; not process allocation',
            'cpu_memory_scope':'process RSS sampled every ~0.5s + nvidia-smi latency; approximate peak',
            'gpu_job_wall_scope':'entire runner incl. init/preflight/warmup/eval/checkpoint/CPU gaps; conservative',
            'common_completed_steps':min((a['steps'] for a in profiles),default=0) if len(profiles)==3 else 0}
        write_json(ROOT/'reports/resource_profile.json',report);event('job_end',**job,cumulative_gpu_wall_seconds=prior['cumulative_seconds']);log.close()

if __name__=='__main__':main()
