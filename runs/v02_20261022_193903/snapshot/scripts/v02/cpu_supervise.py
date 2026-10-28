"""900-second aggregate complete-process cap for V0.2 CPU preparation/checks."""
import json,os,subprocess,sys,time
from pathlib import Path
import yaml
root=Path(os.environ['VPB_WORKSPACE']).resolve();base=Path(__file__).resolve().parents[2];cfg=yaml.safe_load((base/'configs/v02.yaml').read_text())
run=root/'runs'/cfg['run_name'];reports=root/cfg['report_directory'];stage=sys.argv[1]
if stage not in ['prepare','numerics']:raise SystemExit('Unknown stage')
path=reports/'cpu_budget.json';ledger=json.loads(path.read_text()) if path.exists() else {'limit_seconds':900,'cumulative_seconds':0,'stages':[]}
remaining=900-ledger['cumulative_seconds']
if remaining<=10:raise SystemExit('CPU budget reached')
if any(s['stage']==stage for s in ledger['stages']):raise SystemExit('Existing stage evidence; use a new directory for a repaired CPU version')
command=[sys.executable,'-u',str(Path(__file__).with_name('prepare.py'))] if stage=='prepare' else [sys.executable,'-m','unittest','discover','-s',str(base/'tests'),'-p','test_v02_tables.py','-v']
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';child=None;code=1;error=None
start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());start=time.perf_counter()
try:
    with open(run/f'{stage}.log','xb') as log:
        child=subprocess.Popen(command,cwd=base,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:code=child.wait(timeout=remaining-5)
        except subprocess.TimeoutExpired:child.kill();child.wait(timeout=3);code=124
except BaseException as exc:error={'type':type(exc).__name__,'message':str(exc)}
finally:
    if child is not None and child.poll() is None:child.kill();child.wait(timeout=3)
    seconds=time.perf_counter()-start
    ledger['stages'].append({'stage':stage,'command':command,'start_utc':start_utc,'end_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'elapsed_seconds':seconds,'exit_code':code,'error':error})
    ledger['cumulative_seconds']+=seconds;path.write_text(json.dumps(ledger,indent=2),encoding='utf-8')
print(json.dumps(ledger['stages'][-1],indent=2));raise SystemExit(code)
