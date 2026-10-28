"""Copy the sealed V0.1 numerical loop, making only the V0.2 interventions."""
import difflib
from pathlib import Path
root=Path(__file__).resolve().parents[2]
source=root/'runs/v01_20261022/snapshot/scripts/v01/run_gpu.py'
old=source.read_text(encoding='utf-8');new=old
replacements=[
('Twelve sealed paired one-pass trajectories; the external supervisor owns budget.','Eight frozen-output norm/direction trajectories; unchanged V0.1 numerical loop.'),
("deadline=float(os.environ['V01_GPU_DEADLINE_EPOCH'])","deadline=float(os.environ['V02_GPU_DEADLINE_EPOCH'])"),
("'new_gpu_process_wall_seconds':None,'v0_gpu_process_wall_seconds':19.406", "'new_gpu_process_wall_seconds':None,'previous_project_gpu_process_wall_seconds':332.12303610004624"),
("prior=torch.load(PRIVATE/'prepared.pt',map_location='cpu',weights_only=True);logpi=prior['pi'].log().float()", "prior=torch.load(ROOT/CFG['reuse']['prior'],map_location='cpu',weights_only=True);logpi=prior['pi'].log().float()\n        tables=torch.load(PRIVATE/'tables.pt',map_location='cpu',weights_only=True)\n        proof=read_json(RUN/'input_proof.json')\n        assert set(tables)=={'C','U','RS','S'}"),
("selection=read_json(REPORT/'data_selection.json')","selection=read_json(ROOT/CFG['reuse']['selection'])"),
("sha(CODE_ROOT/'configs/v01.yaml')","sha(CODE_ROOT/'configs/v02.yaml')"),
("common=torch.load(PRIVATE/f'common_init_{seed}.pt',map_location='cpu',weights_only=True)","common=torch.load(ROOT/CFG['reuse']['common_init_template'].format(seed=seed),map_location='cpu',weights_only=True)"),
("order=read_json(RUN/f'block_order_{seed}.json');assert len(order)==len(set(order))==3068", "order_path=ROOT/CFG['reuse']['block_order_template'].format(seed=seed)\n            order=read_json(order_path);assert len(order)==len(set(order))==3068\n            assert set(order)==set(selection['selected_block_indices'])"),
("guard();trajectory_start=time.perf_counter();monitor=Monitor();monitor.start()", "guard();trajectory_start_utc=utc();trajectory_start=time.perf_counter()\n                monitor=Monitor();monitor.start()"),
("out=codes['output'] if condition[0]!='S' else torch.where(codes['output']>=0,1.,-1.)/math.sqrt(128)\n                bias=logpi if condition.endswith('p') else torch.zeros_like(logpi)\n                model=OutputStudent(codes['input'],out,condition[0]=='L',bias,**CFG['student']);load_common_state_dict(model,common)", "out=tables[condition]\n                bias=logpi\n                model=OutputStudent(codes['input'],out,False,bias,**CFG['student']);load_common_state_dict(model,common)"),
("model.cuda();torch.cuda.reset_peak_memory_stats()\n                parameters", "assert before==proof['seeds'][str(seed)]['conditions'][condition]['initial_hashes']\n                model.cuda();torch.cuda.reset_peak_memory_stats()\n                parameters"),
("optimizer=torch.optim.AdamW(parameters,lr=3e-4,weight_decay=.01,fused=True)", "optimizer=torch.optim.AdamW(parameters,lr=3e-4,weight_decay=.01,betas=(.9,.999),eps=1e-8,fused=True)"),
("'status':'RUNNING','start_utc':utc(),'steps':0", "'status':'RUNNING','start_utc':trajectory_start_utc,'start_monotonic_seconds':trajectory_start,'steps':0"),
("sha(RUN/f'block_order_{seed}.json')","sha(order_path)"),
("assert (before['output_codes']!=after['output_codes'])==(condition[0]=='L')", "assert before['output_codes']==after['output_codes']"),
("'BUDGET_INCOMPLETE' if isinstance(exc,TimeoutError) else 'FAILED'", "'BUDGET_REACHED' if isinstance(exc,TimeoutError) else 'FAILED'"),
]
for before,after in replacements:
    if new.count(before)!=1:raise RuntimeError(f'Expected one exact source match: {before[:90]} / count={new.count(before)}')
    new=new.replace(before,after)
start=new.index('        # Small synthetic shape/memory check;')
end=new.index("        for seed in CFG['seeds']:",start)
new=new[:start]+'''        # Exactly one synthetic shape check, no optimizer and no corpus replay.
        guard();probe=OutputStudent(codes['input'],tables['C'],False,logpi,**CFG['student']).cuda()
        ids=torch.arange(32,device='cuda').remainder(50304)[None];labels=next_token_labels(ids)
        with torch.autocast('cuda',dtype=torch.bfloat16):value=ce(probe(ids),labels)/31
        value.backward();torch.cuda.synchronize();assert torch.isfinite(value)
        event('gpu_preflight',output_type='C',microbatch=1,sequence_length=32,optimizer_steps=0,synthetic_targets=31,allocated_bytes=torch.cuda.memory_allocated())
        del probe,ids,labels,value;gc.collect();torch.cuda.empty_cache()
'''+new[end:]
assert "condition[0]" not in new and "condition.endswith" not in new
assert "PRIVATE/f'common_init" not in new and "PRIVATE/'prepared.pt'" not in new
destination=root/'scripts/v02/run_gpu.py'
if destination.exists():raise SystemExit('Refuse overwrite generated runner')
destination.write_text(new,encoding='utf-8')
report=root/'reports/v02/runner_v01_to_v02.diff';report.parent.mkdir(parents=True,exist_ok=True)
report.write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='V01_sealed_run_gpu.py',tofile='V02_run_gpu.py')),encoding='utf-8')
print(f'Created {destination}; numerical loop retained; diff={report}')
