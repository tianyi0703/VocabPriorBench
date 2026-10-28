"""External process watchdog for the one authorized V0 GPU job."""
import json,subprocess,sys,time
from pathlib import Path
import yaml
root=Path(__file__).resolve().parents[1]
cfg=yaml.safe_load((root/'configs/v0.yaml').read_text())
if (root/'runs/v0_20261021/events.jsonl').exists() or (root/'runs/gpu_console.log').exists():
    raise SystemExit('Existing GPU evidence: refuse overwrite or automatic rerun')
budget_path=root/'reports/gpu_budget.json'
budget=json.loads(budget_path.read_text()) if budget_path.exists() else {'jobs':[],'cumulative_seconds':0}
remaining=cfg['training']['gpu_wall_limit_seconds']-budget['cumulative_seconds']
if remaining<cfg['training']['estimated_job_seconds']:raise SystemExit('Insufficient budget; no child started')
start=time.monotonic();start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
with open(root/'runs/gpu_console.log','wb') as transcript:
    child=subprocess.Popen([sys.executable,'-u',str(root/'scripts/run_v0.py')],cwd=root,stdout=transcript,stderr=subprocess.STDOUT)
    print(f'Started local runner pid={child.pid}; external timeout={remaining-100:.1f}s; transcript=runs/gpu_console.log',flush=True)
    try:code=child.wait(timeout=remaining-100)
    except subprocess.TimeoutExpired:
        subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=transcript,stderr=subprocess.STDOUT,timeout=20)
        child.wait(timeout=20);code=124
        ledger=json.loads(budget_path.read_text())
        job=ledger['jobs'][-1];job.update(status='EXTERNAL_WATCHDOG_TERMINATED',end_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),wall_seconds=time.monotonic()-start)
        ledger['cumulative_seconds']=budget['cumulative_seconds']+job['wall_seconds']
        budget_path.write_text(json.dumps(ledger,indent=2),encoding='utf-8')
elapsed=time.monotonic()-start
result={'start_utc':start_utc,'end_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'wall_seconds':elapsed,'exit_code':code,'scope':'whole supervisor interval including child imports'}
(root/'runs/supervisor_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
ledger=json.loads(budget_path.read_text()) if budget_path.exists() else None
if ledger and len(ledger['jobs'])>len(budget['jobs']):
    job=ledger['jobs'][-1]
    job['cuda_runner_wall_seconds']=job['wall_seconds'];job['supervisor_start_utc']=start_utc;job['supervisor_end_utc']=result['end_utc'];job['wall_seconds']=elapsed
    job['wall_scope']='whole supervisor interval including child imports'
    ledger['cumulative_seconds']=budget['cumulative_seconds']+elapsed
    budget_path.write_text(json.dumps(ledger,indent=2),encoding='utf-8')
    resource_path=root/'reports/resource_profile.json'
    if resource_path.exists():
        resource=json.loads(resource_path.read_text());resource['gpu_budget']=ledger
        resource['gpu_job_wall_scope']=job['wall_scope'];resource_path.write_text(json.dumps(resource,indent=2),encoding='utf-8')
print(f'Runner exit={code}; supervisor wall={elapsed:.3f}s',flush=True)
raise SystemExit(code)
