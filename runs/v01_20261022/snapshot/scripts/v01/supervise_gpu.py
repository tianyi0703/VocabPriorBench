"""Budget the full GPU child lifetime, including imports, before any CUDA work."""
import json,os,subprocess,sys,time
from pathlib import Path
root=Path(os.environ['VPB_WORKSPACE']);run=root/'runs/v01_20261022';reports=root/'reports/v01'
if (run/'events.jsonl').exists() or (run/'gpu_console.log').exists():raise SystemExit('Existing GPU evidence; refuse overwrite or implicit retry')
ledger_path=reports/'gpu_budget.json';ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'limit_seconds':3600,'v0_seconds':19.406,'new_cumulative_seconds':0,'jobs':[]}
remaining=3600-ledger['new_cumulative_seconds']
if remaining<600:
    ledger['status']='BUDGET_INCOMPLETE';ledger_path.write_text(json.dumps(ledger,indent=2));raise SystemExit('Cannot start estimated 600s job inside remaining budget')
job={'supervisor_start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'supervisor_end_utc':None,'elapsed_seconds':0,'exit_code':None,'status':'RUNNING',
     'command':[sys.executable,'-u',str(Path(__file__).with_name('run_gpu.py'))]}
ledger['jobs'].append(job);ledger_path.write_text(json.dumps(ledger,indent=2),encoding='utf-8')
env=os.environ.copy();env['V01_GPU_DEADLINE_EPOCH']=str(time.time()+remaining-60)
start=time.perf_counter();child=None;code=1
try:
    with open(run/'gpu_console.log','xb') as log:
        child=subprocess.Popen(job['command'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        job['pid']=child.pid;ledger_path.write_text(json.dumps(ledger,indent=2),encoding='utf-8')
        print(f'GPU child pid={child.pid}, external deadline={remaining-60:.1f}s; 12 paired trajectories',flush=True)
        try:code=child.wait(timeout=remaining-60)
        except subprocess.TimeoutExpired:
            code=124
            try:subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=log,stderr=subprocess.STDOUT,timeout=20)
            finally:
                if child.poll() is None:child.kill()
                child.wait(timeout=20)
except BaseException as exc:job['supervisor_error']={'type':type(exc).__name__,'message':str(exc)}
finally:
    if child is not None and child.poll() is None:
        child.kill();child.wait(timeout=5)
    elapsed=time.perf_counter()-start;resource_path=reports/'resource_profile.json'
    result=json.loads(resource_path.read_text()) if resource_path.exists() else None
    incomplete=code==124 or (result is not None and result.get('status')=='BUDGET_INCOMPLETE')
    job.update(supervisor_end_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),elapsed_seconds=elapsed,exit_code=code,status='COMPLETED' if code==0 else ('BUDGET_INCOMPLETE' if incomplete else 'FAILED'))
    ledger['new_cumulative_seconds']+=elapsed;ledger['project_cumulative_seconds']=19.406+ledger['new_cumulative_seconds'];ledger['status']=job['status']
    ledger_path.write_text(json.dumps(ledger,indent=2),encoding='utf-8');(run/'supervisor_result.json').write_text(json.dumps(job,indent=2),encoding='utf-8')
    if result is not None:
        result.update(gpu_budget=ledger,new_gpu_process_wall_seconds=ledger['new_cumulative_seconds'],project_gpu_process_wall_seconds=ledger['project_cumulative_seconds'])
        if incomplete:result['status']='BUDGET_INCOMPLETE'
        resource_path.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(job,indent=2),flush=True);raise SystemExit(code)
