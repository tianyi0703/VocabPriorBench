"""Audit raw measurements, apply the locked screen, write report and review ZIP."""
import csv,hashlib,json,math,sys,zipfile
from pathlib import Path
from common import CFG,ROOT,CODE_ROOT,RUN,REPORT,PRIVATE,read_json,write_json,sha,utc,verify_snapshot,verify_inputs

def write_csv(path,rows):
    with open(path,'w',encoding='utf-8-sig',newline='') as output:
        writer=csv.DictWriter(output,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def classify(zero,prior):
    tau=CFG['screen']['tau_nats_per_token'];fraction=CFG['screen']['sensitive_remaining_fraction']
    if zero[0]*zero[1]>0 and all(abs(x)>tau for x in zero) and all(abs(p)<=tau and abs(p)<=fraction*abs(z) for z,p in zip(zero,prior)):
        return 'PRIOR_SENSITIVE_IN_THIS_SCREEN'
    if prior[0]*prior[1]>0 and all(abs(p)>tau for p in prior):return 'RESIDUAL_TO_EXPLAIN'
    if all(abs(x)<=tau for x in zero+prior):return 'NO_MATERIAL_GAP_IN_THIS_SCREEN'
    return 'INCONCLUSIVE'

def main():
    snapshot=verify_snapshot();source=verify_inputs();resource=read_json(REPORT/'resource_profile.json');diag=read_json(REPORT/'diagnostics.json')
    selection=read_json(REPORT/'data_selection.json');budget=read_json(REPORT/'gpu_budget.json');cpu=read_json(REPORT/'cpu_budget.json')
    rows=read_json(RUN/'metrics_raw.json');events=[json.loads(s) for s in (RUN/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    complete=resource['status']=='COMPLETED' and len(resource['trajectories'])==12 and len(rows)==36
    checks={'snapshot_and_input_hashes':True,'holdout_opened':False,'full_dev_valid_targets':all(r['dev_valid_targets']==62601 for r in rows),'new_gpu_budget':budget['new_cumulative_seconds']<=3600,
            'cpu_conservative_charged_budget':cpu['cumulative_seconds']<=1800,'old_source_history':'LIMITED_V0_HISTORY'}
    if complete:
        for trajectory in resource['trajectories']:
            seed,condition=trajectory['seed'],trajectory['condition']
            steps=[r for r in events if r['event']=='step' and r['seed']==seed and r['condition']==condition]
            assert [s['step'] for s in steps]==list(range(1,768))
            assert sum(s['valid_targets'] for s in steps)==389636
            assert abs(sum(s['nll_sum'] for s in steps)-trajectory['train_online_nll_sum'])<1e-8
            assert abs(sum(s['seconds'] for s in steps)-trajectory['train_seconds'])<1e-8
            assert trajectory['before_hashes']['input_codes']==trajectory['after_hashes']['input_codes']
            assert trajectory['before_hashes']['output_bias']==trajectory['after_hashes']['output_bias']
            assert (trajectory['before_hashes']['output_codes']!=trajectory['after_hashes']['output_codes'])==condition.startswith('L')
            assert trajectory['all_common_parameters_updated'] and trajectory['first_step_nonzero_gradient']
            assert sha(ROOT/trajectory['checkpoint_path'])==trajectory['checkpoint_sha256']
            evaluations=[r for r in rows if r['seed']==seed and r['condition']==condition]
            assert [r['step'] for r in evaluations]==[0,100,767]
            assert all(abs(r['nll']-r['nll_sum']/r['dev_valid_targets'])<1e-12 for r in evaluations)
        for seed in CFG['seeds']:
            trajectories=[t for t in resource['trajectories'] if t['seed']==seed]
            assert len({t['common_initial_state_sha256'] for t in trajectories})==1
            assert len({t['block_order_sha256'] for t in trajectories})==1
            assert len({t['before_hashes']['input_codes'] for t in trajectories})==1
            lookup={r['condition']:r for r in rows if r['seed']==seed and r['step']==0}
            assert lookup['L0']['nll']==lookup['F0']['nll'] and lookup['Lp']['nll']==lookup['Fp']['nll']
        checks.update(all_12_trajectories_one_pass=True,all_common_initializations_and_orders_paired=True,numerical_invariants_and_saved_checkpoints=True)
    assert checks['new_gpu_budget'] and checks['cpu_conservative_charged_budget']
    write_json(RUN/'final_audit.json',checks);write_csv(REPORT/'metrics.csv',rows)
    contrasts=[]
    if complete:
        for seed in CFG['seeds']:
            for step in CFG['training']['evaluation_steps']:
                n={r['condition']:r['nll'] for r in rows if r['seed']==seed and r['step']==step}
                f0=n['F0']-n['L0'];fp=n['Fp']-n['Lp'];s0=n['S0']-n['F0'];sp=n['Sp']-n['Fp']
                contrasts.append({'seed':seed,'step':step,'delta_freeze_0':f0,'delta_freeze_p':fp,'delta_sign_0':s0,'delta_sign_p':sp,'interaction_freeze':fp-f0,'interaction_sign':sp-s0})
        write_csv(REPORT/'contrasts.csv',contrasts)
    final=[r for r in contrasts if r['step']==767]
    freeze=classify([r['delta_freeze_0'] for r in final],[r['delta_freeze_p'] for r in final]) if complete else 'NOT_RUN'
    sign=classify([r['delta_sign_0'] for r in final],[r['delta_sign_p'] for r in final]) if complete else 'NOT_RUN'
    literature=read_json(REPORT/'literature_sources.json')
    # The literature table states no verified complete overlap; no novelty is inferred.
    overlap='FULL_OVERLAP = NOT_ESTABLISHED' in (REPORT/'LITERATURE_DIFFERENCE.md').read_text(encoding='utf-8')
    residual='RESIDUAL_TO_EXPLAIN' in [freeze,sign]
    next_state='DRAFT_RESIDUAL_HYPOTHESIS' if complete and residual and overlap else ('PAUSE_CURRENT_CLAIM' if complete and freeze!='INCONCLUSIVE' and sign!='INCONCLUSIVE' else 'INCONCLUSIVE')
    context={condition:all(r['context_gain']>0 for r in rows if r['condition']==condition and r['step']==767) for condition in CFG['conditions']} if complete else {}
    decision={'tau':.05,'endpoint':767,'OUTPUT_FREEZE':freeze,'OUTPUT_SIGN':sign,'NEXT':next_state,'context_gain_positive_both_seeds_by_condition':context,'final_contrasts':final,
              'interpretation':'exploratory engineering/mechanism screen; no equivalence, universal, unique-cause, or novelty claim'}
    write_json(REPORT/'screening.json',decision)
    source.update(completed_utc=utc(),execution_environment=read_json(RUN/'environment.json'),gpu_budget=budget,cpu_budget=cpu,
                  literature_sources={'path':'reports/v01/literature_sources.json','sha256':sha(REPORT/'literature_sources.json')},
                  preserved_v0={'source_manifest_sha256':sha(ROOT/'reports/source_manifest.json'),'resource_profile_sha256':sha(ROOT/'reports/resource_profile.json'),
                   'normalized_timing':read_json(REPORT/'V0_LOG_AUDIT.json')['normalized_v0_timing']},
                  output_metrics_sha256=sha(REPORT/'metrics.csv'),output_diagnostics_sha256=sha(REPORT/'diagnostics.json'))
    write_json(REPORT/'source_manifest.json',source)
    baseline=diag['train_pi_dev_nll']
    metrics_table='\n'.join(f"| {r['seed']} | {r['condition']} | {r['step']} | {r['train_valid_targets']:,} | {r['nll_sum']:.6f} / {r['dev_valid_targets']:,} | {r['nll']:.6f} | {r['context_gain']:+.6f} |" for r in rows)
    contrast_table='\n'.join(f"| {r['seed']} | {r['step']} | {r['delta_freeze_0']:+.6f} | {r['delta_freeze_p']:+.6f} | {r['interaction_freeze']:+.6f} | {r['delta_sign_0']:+.6f} | {r['delta_sign_p']:+.6f} | {r['interaction_sign']:+.6f} |" for r in contrasts)
    diagnostic_table='\n'.join(f"| {name} | {fit.get('iterations')} / {fit.get('closure_calls_logged')} | {fit.get('objective',float('nan')):.8f} | {fit.get('L_minus_Hpi',float('nan')):.8f} | {fit.get('gradient_inf',float('nan')):.3g} / {fit.get('gradient_l2',float('nan')):.3g} | {fit.get('dev_nll',float('nan')):.6f} | {fit.get('elapsed_seconds',0):.3f} | {fit.get('stop_reason')} | {fit.get('residual_interpretation')} |" for name,fit in diag['fits'].items())
    resources_table='\n'.join(f"| {t['seed']} | {t['condition']} | {t['steps']} | {t.get('wall_seconds',0):.3f} | {t.get('train_seconds',0):.3f} / {t.get('eval_seconds',0):.3f} | {t.get('torch_peak_allocated_bytes',0)/2**20:.1f} / {t.get('torch_peak_reserved_bytes',0)/2**20:.1f} | {t.get('device_global_peak_used_mib',0):.0f} | {t.get('cpu_sampled_peak_rss_bytes',0)/2**20:.1f} |" for t in resource['trajectories'])
    next_page=''
    if residual and overlap:
        next_page='''## 仅供下一阶段设计的残差假设（一页候选，未执行）

具体目标为本轮共同先验下仍超过tau的**有符号**冻结/sign残差，不将其先命名为语义损失。一个训练前可计算的候选预测量为：在训练频率分箱内，比较连续与sign输出码对指定token行的训练pi加权投影margin变化；预先固定抽样与分箱，仅在来源条件选择一次。该量是候选关联量，不自动有因果含义。

最简单竞争解释是行范数变化、pi质量与稀有token误差、输出谱/条件数、优化速度、可训练参数自由度及固定bias造成的仿射空间平移。优先检验这些解释是否已足够，不能只对候选量画相关图。任何原维表若学生hidden仍128，均不称解除128维线性rank界。

后续验证应在**未参与当前筛选**的既定条件锁定方向/量级预测，并预先明确一项可否证对照，而非继续用本dev挑预测量；具体来源及预算需另行授权。本轮不打开holdout、不下载语料、不新增条件或训练。若最简单频率/范数/谱解释已消除预测增益，或候选量在保留条件方向失败，则否定该预测假设并暂停贡献主张；不是否定领域。若本轮模型未在两个seed都超出unigram，当前唯一优先缺口是尚无成熟上下文信号，不能依残差扩大训练。
'''
    status_resource='WITHIN_1H' if complete else 'BUDGET_INCOMPLETE'
    report=f'''# VocabPriorBench V0.1 输出端与共同先验诊断

EVIDENCE = LIMITED_V0_HISTORY

NUMERICS = {'PASS' if complete else 'NOT_RUN'}

DIAGNOSTIC = {diag['status']}

OUTPUT_FREEZE = {freeze}

OUTPUT_SIGN = {sign}

RESOURCE = {status_resource}

NEXT = {next_state}

日期：2026-10-22（Asia/Shanghai）。本轮已按 [完整指令](../../CODEX_V01_VocabPriorBench.txt) 执行限定筛选，固定输入、6输出×bias条件、仅1729/2718两seed。{('12条轨迹全部完成，每条767步，一次遍历3068块/389636有效目标。' if complete else '预算或执行不完整，保留现有记录，不择取有利子集。')} 新增GPU完整子进程累计 **{budget['new_cumulative_seconds']:.3f}秒**，V0独立19.406秒，项目累计 **{budget['project_cumulative_seconds']:.3f}秒**。完成即停止，未启动V1。

## 筛选结论与能支持的主张

冻结输出分类 **{freeze}**；sign输出分类 **{sign}**，保留两效应独立结论。tau=.05 nats/token及25%规则在结果出现前锁定，没有搜索阈值、seed、码宽、数据或终点。该阈值是工程预算容忍值，约exp(.05)=1.0513 PPL比，不是显著性/等价/发表标准。`screening.json`保留判定输入与分类；正值为该操作在本终点增加dev NLL，负值也原样保存。

固定训练unigram的完整dev NLL为 **{baseline:.6f}**。最终每条件在两seed均获得正context_gain的描述标记为 `{context}`；它仅说明超过固定unigram的预测收益，不等于语义能力或长期稳定。若有False，该条件尚不足讨论成熟上下文机制。两个seed不能支持统计等价、普适性或新颖性；本轮不把token/块/文档当独立训练重复。不将新条件与V0三臂组成统计对照。

## V0证据链与封存

见 [源码审计](V0_SOURCE_AUDIT.md)、[原始日志复核](V0_LOG_AUDIT.md)及各自JSON。base模型、run_v0、基础检查与执行hash一致，因果mask、padding/shift、有效目标、冻结、公共初态及同步计时路径可审。prepare/build_report/supervise的执行前/交付hash不同，项目中未找到真实旧正文，**不能仅凭hash恢复diff或断言只有元数据改动**；因此保留LIMITED_V0_HISTORY。现有码表、数据和初态可对照旧hash；这一历史文本缺失不阻止已封存V0.1。

V0规范化：runner_start_utc=2026-10-21T12:15:17Z，runner_end_utc=12:15:28Z，runner_elapsed_seconds=11.211335600004531；supervisor_elapsed_seconds=19.406，supervisor起止UTC均null/unknown。19.406来自原工具stdout后归档，没有把它重新标成当时自动保存的原始测量，也未反推伪精确时间。V0所有原文件保留，未重跑GPU。

本轮实际执行路径是 `runs/v01_20261022/snapshot`，封存树hash `{snapshot['tree_sha256']}`，指令、配置、契约、源码与测试均在其中；所有数值/GPU子进程从该树执行，源码每阶段重验。来源manifest保存输入产物及公共初态/具体块顺序hash，checkpoint留本地private并核对SHA，不进入审查ZIP。holdout本轮未打开或重哈希，旧metadata中的hash只作为已有记录。

## 定向文献与数学边界

{(REPORT/'LITERATURE_DIFFERENCE.md').read_text(encoding='utf-8')}

## 已锁定的数据和实验

原train有3070个完整块及一个122有效输入token的尾块；选前3068完整块，舍弃两个不能组成batch4的完整尾块与不完整尾块，实际列表在data_selection.json。pi按选定集合实际损失目标n_v计算，alpha=.1，N=389636、V=50304，计入有效EOS且排除pad/块末忽略位置。第0步已有一次完整训练统计遍历成本，记录 {selection['prior_count_seconds']:.6f}秒/389636目标；不声称只按在线更新token公平的数据效率优势。完整dev493块、62601目标均评估；train不作重复模型评估。原packed没有可靠逐token文档映射，**不支持文档级重采样**。

L0/F0/S0是连续可训/连续冻结/sign冻结输出和零固定bias；Lp/Fp/Sp是对应输出和同一固定log(pi)。六条件连续输入都冻结，S只sign输出；bias均FP32 buffer、相同计算路径。公共主干/投影444928参数可训；L另训6438912输出表参数，输入表始终不训。相同seed从同一初态文件加载，独立storage，不顺序续训；不同seed变更初始化和排列、集合相同。全部沿用lr3e-4、AdamW/.01、clip1、seq128、microbatch2×accum2、bf16 autocast/FP32参数、dropout0。固定顺序1729正序、2718反序，评估step0/100/767，最终点为主要终点。契约 [EXPERIMENT_CONTRACT.md](EXPERIMENT_CONTRACT.md) 与 [实际配置](../../configs/v01.yaml) 已封存。

原7项基础数值测试及新增3项均通过，封存版本又经CPU检查；实际全词表同bias L/F初态前向差为0。GPU逐步loss/grad/param有限，公共参数更新，输入和固定bias不变，F/S输出不变而L输出改变；最终文件hash与资源/日志对应，final_audit.json记录逐轨迹一次遍历和计数。GPU预检使用合成32token序列，无optimizer更新；无额外训练热身遍历。

## CPU无上下文拟合（实测）

H(pi)={diag['entropy_pi']:.8f}，直接pi在dev的NLL={baseline:.8f}。float64、零a、固定LBFGS/strong_wolfe，迭代/容差均来自事前配置；无dev调参。a=0加log(pi)后softmax与pi最大绝对误差 {diag['softmax_log_pi_max_abs_error']:.3g}，容差1e-12；手写梯度与autograd已核对。

| 初始输出 | iter/closure | L(a) | L−H(pi) | grad inf/l2 | 完整dev NLL | 拟合秒 | 实际停止 | 残差解释 |
|---|---:|---:|---:|---:|---:|---:|---|---|
{diagnostic_table}

收敛验收用独立重算grad_inf≤1e-6；优化器具体停止分支若不暴露，原样记录而不猜。未充分收敛仅称当前优化残差；即使满足一阶验收，也没有精确最优误差证书。常量KL拟合只说明无上下文表达，不是上下文LM NLL下界，不单独解释训练差距。CPU数值阶段完整进程实测合计 {cpu['measured_stage_seconds']:.3f}秒，加封存前检查保守预留60秒后的预算占用 {cpu['cumulative_seconds']:.3f}/1800秒；预留不是伪造实测时长。原封存前unittest内部2.474秒另存。

## 六条件、两个seed、三个预定评价点

每项分母均完整dev62601；context_gain=训练pi dev NLL−模型dev NLL。train累计目标只含实际optimizer训练，online train分子/分母另见metrics.csv和逐step事件。

| seed | 条件 | step | 累计训练目标 | dev NLL分子/分母 | dev NLL | context_gain |
|---:|---|---:|---:|---|---:|---:|
{metrics_table}

| seed | step | freeze0 | freezep | interaction_freeze | sign0 | signp | interaction_sign |
|---:|---:|---:|---:|---:|---:|---:|---:|
{contrast_table}

差距对共同固定先验干预敏感不唯一证明“词频解释完毕”：它同时改变频率信息和logits仿射空间。残差也不自动证明语义损失、新方法、最终优劣或可发表性。

## 真实资源与预算

| seed | 条件 | steps | 整轨迹秒 | train/eval秒 | allocated/reserved MiB | 全设备峰值MiB | CPU采样RSS MiB |
|---:|---|---:|---:|---:|---:|---:|---:|
{resources_table}

新增完整GPU子进程含导入、初始化、合成预检、所有训练/评估/保存/失败，独立supervisor直接保存起止UTC/单调计时/exit code。新增 {budget['new_cumulative_seconds']:.3f}/3600秒，项目累计 {budget['project_cumulative_seconds']:.3f}秒，OOM={resource['oom_count']}、错误={len(resource['errors'])}。短序列速度只记录实际开销，不外推持续热稳态/12小时/更大模型。PyTorch allocator峰值只属本进程，nvidia-smi全设备含桌面/其他程序；CPU RSS为采样近似。sign仍FP32、两码表49.125MiB，bias约0.192MiB，普通浮点matmul，没有1-bit GPU存储或新kernel。

## 实际命令与审查包

```powershell
Set-Location F:\\VocabPriorBench
& .\\.venv\\Scripts\\python.exe scripts\\v01\\seal.py
$env:VPB_WORKSPACE='F:\\VocabPriorBench'
& .\\.venv\\Scripts\\python.exe runs\\v01_20261022\\snapshot\\scripts\\v01\\cpu_supervise.py numerics
& .\\.venv\\Scripts\\python.exe runs\\v01_20261022\\snapshot\\scripts\\v01\\cpu_supervise.py prepare
& .\\.venv\\Scripts\\python.exe runs\\v01_20261022\\snapshot\\scripts\\v01\\cpu_supervise.py diagnose
& .\\.venv\\Scripts\\python.exe runs\\v01_20261022\\snapshot\\scripts\\v01\\supervise_gpu.py
& .\\.venv\\Scripts\\python.exe runs\\v01_20261022\\snapshot\\scripts\\v01\\finalize.py
```

这些入口保护现有日志，拒绝隐式覆盖/重跑。原始CPU closure、每step训练事件、36个完整dev测量、monitor/resource、supervisor预算与命令记录保留在runs/v01_20261022；无模型或语料下载。V01_REVIEW_BUNDLE.zip包含本报告/配置/契约、执行源码快照、输入哈希、V0关键数值与事件原日志、本轮原日志/metrics/diagnostics和逐文件校验清单；**排除.venv、任何.pt/.safetensors权重、原始/分词语料和private目录**。精确逐文件打包清单见 [BUNDLE_MANIFEST.json](BUNDLE_MANIFEST.json)，包本身SHA另存BUNDLE_SHA256.txt；包中清单不自引用。

{next_page}

## 停止与限制

本轮是工程筛选和有限机制敏感性证据；新颖性仍未确立。一个donor/码宽/架构/小语料，一次epoch、两个seed，不能宣称成熟训练最优、等价、普适或推翻原文。V0旧源码文本及supervisor原始输出封存限制如实保留；WikiText来源卡许可元数据/正文版本冲突与donor重叠UNKNOWN沿用原记录。本轮没有打开holdout、追加种子/数据或启动七臂V1。唯一后续动作为{('起草可否证的残差假设并先排除最简单竞争解释，不自动训练' if next_state=='DRAFT_RESIDUAL_HYPOTHESIS' else '暂停当前差距作为独立贡献，或针对INCONCLUSIVE列明最小缺口，不追加实验')}。
'''
    (REPORT/'V01_REPORT.md').write_text(report,encoding='utf-8')
    # Allowlist only: never enumerate or read data/, artifacts/, .venv/, or private/.
    files=[ROOT/'configs/v01.yaml',ROOT/'CODEX_V01_VocabPriorBench.txt',ROOT/'requirements.txt']
    files.extend(p for p in REPORT.iterdir() if p.is_file() and p.name not in ['BUNDLE_MANIFEST.json','BUNDLE_SHA256.txt'])
    files.extend(p for p in RUN.rglob('*') if p.is_file() and 'private' not in p.relative_to(RUN).parts and '__pycache__' not in p.parts and p.suffix not in ['.pt','.pyc','.safetensors'])
    v0names=['reports/V0_REPORT.md','reports/source_manifest.json','reports/resource_profile.json','reports/gpu_budget.json','configs/v0.yaml','runs/COMMANDS.md','runs/numerical_checks.log','runs/gpu_console.log','runs/artifact_audit.json','runs/supervisor_result.json','runs/prepare.log','runs/pip_freeze.txt']
    files.extend(ROOT/name for name in v0names)
    files.extend(p for p in (ROOT/'runs/v0_20261021').iterdir() if p.suffix in ['.json','.jsonl'])
    if (ROOT/'runs/v01_numerics.log').exists():files.append(ROOT/'runs/v01_numerics.log')
    files=sorted(set(files));manifest={'created_utc':utc(),'exclusions':['.venv','private','all weights','raw/tokenized corpora'],
                                     'files':{p.relative_to(ROOT).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in files}}
    write_json(REPORT/'BUNDLE_MANIFEST.json',manifest)
    archive=ROOT/'V01_REVIEW_BUNDLE.zip'
    if archive.exists():raise RuntimeError('Refuse overwrite review bundle')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
        for path in files+[REPORT/'BUNDLE_MANIFEST.json']:bundle.write(path,path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        for name,spec in manifest['files'].items():assert hashlib.sha256(bundle.read(name)).hexdigest()==spec['sha256']
        assert not any(name.startswith(('data/','artifacts/','.venv/')) or '/private/' in name or name.endswith(('.pt','.safetensors')) for name in bundle.namelist())
    (REPORT/'BUNDLE_SHA256.txt').write_text(f'{sha(archive)}  V01_REVIEW_BUNDLE.zip\n',encoding='utf-8')
    print(json.dumps({'screening':decision,'gpu_seconds':budget['new_cumulative_seconds'],'bundle':str(archive),'bundle_files':len(files)+1,'bundle_bytes':archive.stat().st_size},indent=2))

if __name__=='__main__':main()
