"""One attempt, full-child-lifetime accounting, 1800s hard aggregate limit."""
import json,os,subprocess,sys,time,hashlib
from pathlib import Path
import yaml
root=Path(os.environ['VPB_WORKSPACE']).resolve();base=Path(__file__).resolve().parents[2]
cfg=yaml.safe_load((base/'configs/v02.yaml').read_text());run=root/'runs'/cfg['run_name'];report=root/cfg['report_directory']
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
if (run/'events.jsonl').exists() or (run/'gpu_console.log').exists():raise SystemExit('Existing attempt: no automatic retry or overwrite')
if (report/'input_failure.json').exists():raise SystemExit('CPU verification failed; GPU remains zero')
assert read(run/'prepare_result.json')['status']=='PASS'
assert '\nOK' in (run/'numerics.log').read_text(encoding='utf-8',errors='replace')
manifest=read(report/'source_manifest.json')
for name,spec in {**manifest['input_artifacts'],**manifest['derived_artifacts']}.items():
    path=root/name
    if not path.exists() or path.stat().st_size!=spec['bytes'] or sha(path)!=spec['sha256']:
        failure={'status':'BLOCKED_MISSING_OR_MISMATCHED_V01_INPUT','path':name,'failure':'PRE_GPU_SIZE_OR_SHA','gpu_seconds':0}
        (report/'input_failure.json').write_text(json.dumps(failure,indent=2),encoding='utf-8');raise SystemExit(str(failure))
snapshot=read(base/'SNAPSHOT_MANIFEST.json')
for name,spec in snapshot['files'].items():assert sha(base/name)==spec['sha256'],name
ledger_path=report/'gpu_budget.json'
ledger=read(ledger_path) if ledger_path.exists() else {'limit_seconds':1800,'previous_v0_seconds':19.406,'previous_v01_seconds':312.71703610004624,'previous_project_seconds':332.12303610004624,'new_cumulative_seconds':0,'jobs':[]}
if ledger['jobs']:raise SystemExit('A previous GPU attempt exists; V0.2 forbids an automatic retry')
remaining=1800-ledger['new_cumulative_seconds']
if remaining<cfg['budget']['estimated_gpu_job_seconds']:raise SystemExit('Insufficient budget for planned full run; no CUDA child started')
command=[sys.executable,'-u',str(Path(__file__).with_name('run_gpu.py'))]
env=os.environ.copy();env['V02_GPU_DEADLINE_EPOCH']=str(time.time()+remaining-60)
start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());start=time.perf_counter()
job={'command':command,'supervisor_start_utc':start_utc,'start_monotonic_seconds':start,'status':'RUNNING'}
ledger['jobs'].append(job);ledger_path.write_text(json.dumps(ledger,indent=2),encoding='utf-8');child=None;code=1
try:
    with open(run/'gpu_console.log','xb') as log:
        child=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT);job['pid']=child.pid
        ledger_path.write_text(json.dumps(ledger,indent=2),encoding='utf-8')
        print(f'V0.2 single CUDA child pid={child.pid}; timeout={remaining-60:.1f}s; eight one-pass trajectories',flush=True)
        try:code=child.wait(timeout=remaining-60)
        except subprocess.TimeoutExpired:
            code=124
            try:subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=log,stderr=subprocess.STDOUT,timeout=20)
            finally:
                if child.poll() is None:child.kill()
                child.wait(timeout=20)
except BaseException as exc:job['supervisor_error']={'type':type(exc).__name__,'message':str(exc)}
finally:
    if child is not None and child.poll() is None:child.kill();child.wait(timeout=5)
    end_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());elapsed=time.perf_counter()-start
    result_path=report/'resource_profile.json';result=read(result_path) if result_path.exists() else None
    reached=code==124 or (result and result.get('status')=='BUDGET_REACHED')
    job.update(supervisor_end_utc=end_utc,elapsed_seconds=elapsed,exit_code=code,status='COMPLETED' if code==0 else ('BUDGET_REACHED' if reached else 'FAILED'))
    ledger['status']=job['status'];ledger['new_cumulative_seconds']+=elapsed;ledger['project_cumulative_seconds']=332.12303610004624+ledger['new_cumulative_seconds']
    ledger_path.write_text(json.dumps(ledger,indent=2),encoding='utf-8');(run/'supervisor_result.json').write_text(json.dumps(job,indent=2),encoding='utf-8')
    if result is not None:
        result.update(gpu_budget=ledger,new_gpu_process_wall_seconds=ledger['new_cumulative_seconds'],project_gpu_process_wall_seconds=ledger['project_cumulative_seconds'])
        if reached:result['status']='BUDGET_REACHED'
        result_path.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(job,indent=2),flush=True);raise SystemExit(code)
