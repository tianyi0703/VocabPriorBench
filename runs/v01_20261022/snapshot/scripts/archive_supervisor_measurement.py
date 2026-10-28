"""Archive the first supervisor's actual captured stdout (no GPU execution).

The initial supervisor printed 19.406s to the tool console without saving JSON.
That exact measured value is retained here with its original 1ms display precision.
"""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
result={'wall_seconds':19.406,'display_precision_seconds':0.001,'exit_code':0,
        'measurement_source':'Actual supervise_v0.py stdout captured in tool result: Runner exit=0; supervisor wall=19.406s',
        'scope':'whole supervisor interval including child imports; monotonic measurement',
        'external_start_utc':None,'external_start_utc_note':'not recorded by initial supervisor; not fabricated'}
path=root/'runs/supervisor_result.json'
if path.exists():raise SystemExit('Refuse overwrite')
path.write_text(json.dumps(result,indent=2),encoding='utf-8')
ledger_path=root/'reports/gpu_budget.json';ledger=json.loads(ledger_path.read_text())
assert len(ledger['jobs'])==1 and abs(ledger['cumulative_seconds']-11.211335600004531)<1e-6
job=ledger['jobs'][0];job['cuda_runner_wall_seconds']=job['wall_seconds'];job['wall_seconds']=result['wall_seconds']
job['wall_scope']=result['scope'];job['external_measurement_source']='runs/supervisor_result.json';ledger['cumulative_seconds']=result['wall_seconds']
ledger_path.write_text(json.dumps(ledger,indent=2),encoding='utf-8')
path=root/'reports/resource_profile.json';r=json.loads(path.read_text());r['gpu_budget']=ledger;r['gpu_job_wall_scope']=result['scope']
path.write_text(json.dumps(r,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
