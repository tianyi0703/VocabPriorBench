"""External CPU process cap, including numerical tests and output fitting."""
import json,os,subprocess,sys,time
from pathlib import Path
root=Path(os.environ['VPB_WORKSPACE']);run=root/'runs/v01_20261022';reports=root/'reports/v01'
stage=sys.argv[1];allowed={'prepare':'prepare.py','diagnose':'diagnose.py','numerics':None}
if stage not in allowed:raise SystemExit('Unknown stage')
ledger_path=reports/'cpu_budget.json';ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'limit_seconds':1800,'cumulative_seconds':60,'measured_stage_seconds':0,'preseal_check_reserve_seconds':60,'preseal_test_internal_seconds':2.474,'preseal_process_wall_seconds':None,'note':'60s conservative reserve for preseal CPU checks; not an observed process duration','stages':[]}
remaining=1800-ledger['cumulative_seconds']
if remaining<=10:raise SystemExit('CPU numerical budget exhausted')
if any(s['stage']==stage for s in ledger['stages']):raise SystemExit('Refuse automatic repeat; preserve prior stage')
base=Path(__file__).resolve().parents[2];command=[sys.executable,'-u',str(Path(__file__).parent/allowed[stage])] if allowed[stage] else [sys.executable,'-m','unittest','discover','-s',str(base/'tests'),'-v']
env=os.environ.copy();env['V01_CPU_DEADLINE_EPOCH']=str(time.time()+remaining-5);env['CUDA_VISIBLE_DEVICES']=''
start=time.perf_counter();start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
child=None;code=1;error=None
try:
    with open(run/f'{stage}.log','xb') as log:
        child=subprocess.Popen(command,cwd=base,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:code=child.wait(timeout=remaining-2)
        except subprocess.TimeoutExpired:child.kill();child.wait(timeout=2);code=124
except BaseException as exc:error={'type':type(exc).__name__,'message':str(exc)}
finally:
    if child is not None and child.poll() is None:child.kill();child.wait(timeout=2)
    seconds=time.perf_counter()-start
    ledger['stages'].append({'stage':stage,'command':command,'start_utc':start_utc,'end_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'wall_seconds':seconds,'exit_code':code,'supervisor_error':error})
    ledger['cumulative_seconds']+=seconds;ledger['measured_stage_seconds']+=seconds;ledger_path.write_text(json.dumps(ledger,indent=2),encoding='utf-8')
print(json.dumps(ledger['stages'][-1],indent=2));raise SystemExit(code)
