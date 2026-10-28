"""Post-run, CPU-only arithmetic, provenance verification and review packaging.

This file is deliberately separate from the immutable training snapshot. It
does not import torch, read a tensor object, initialize CUDA or fit a model.
Its classification rules are the rules sealed before the current GPU run.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from analysis import CONTRAST_NAMES, classify, contrasts

ROOT = Path(os.environ['VPB_WORKSPACE']).resolve()
RUN = ROOT / 'runs/v02_20261022_193903'
REPORT = ROOT / 'reports/v02'
SNAP = RUN / 'snapshot'
SEEDS = (1729, 2718)
CONDITIONS = ('C', 'U', 'RS', 'S')
STEPS = (0, 100, 767)
NEXT = 'PAUSE_CURRENT_CLAIM_FOR_RESEARCH_DECISION'


def utc():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def spec(path):
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def csv_write(name, rows, fields=None):
    with (REPORT / name).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(map(str, headers)) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def verify_file(path, expected):
    actual = spec(path)
    assert actual == {k: expected[k] for k in ('bytes', 'sha256')}, str(path)
    return actual


def build():
    begin = time.perf_counter()
    manifest = read(REPORT / 'source_manifest.json')
    sealed = read(SNAP / 'SNAPSHOT_MANIFEST.json')
    checks = {}
    for relative, expected in sealed['files'].items():
        verify_file(SNAP / relative, expected)
    tree = hashlib.sha256(json.dumps(sealed['files'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert tree == sealed['tree_sha256'] == manifest['snapshot_tree_sha256']
    checks['training_snapshot_unchanged'] = True
    verified_inputs = {}
    for relative, expected in manifest['input_artifacts'].items():
        verified_inputs[relative] = verify_file(ROOT / relative, expected)
    for relative, expected in manifest['derived_artifacts'].items():
        verify_file(ROOT / relative, expected)
    checks['all_eight_V01_input_files_still_match'] = len(verified_inputs) == 8
    checks['derived_preparation_files_unchanged'] = True
    for relative, record in manifest['model_dependencies_copied_verbatim'].items():
        assert sha(SNAP / relative) == sha(ROOT / record['source']) == record['sha256']
    checks['actual_V01_model_dependency_bytes_identical'] = True
    historical_sources = read(ROOT / 'reports/v01/source_manifest.json')
    assert sha(ROOT / 'reports/source_manifest.json') == historical_sources['preserved_v0']['source_manifest_sha256']
    checks['original_donor_tokenizer_data_metadata_unchanged'] = True

    metrics = read(RUN / 'metrics_raw.json')
    old = read(ROOT / 'runs/v01_20261022/metrics_raw.json')
    profile = read(REPORT / 'resource_profile.json')
    budget = read(REPORT / 'gpu_budget.json')
    cpu = read(REPORT / 'cpu_budget.json')
    table = read(REPORT / 'TABLE_CONSTRUCTION.json')
    proof = read(RUN / 'input_proof.json')
    env = read(RUN / 'environment.json')
    old_env = read(ROOT / 'runs/v01_20261022/environment.json')
    supervisor = read(RUN / 'supervisor_result.json')
    events = [json.loads(line) for line in (RUN / 'events.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    assert not [e for e in events if e['event'] in ('error', 'trajectory_error', 'job_error')]
    assert budget['status'] == profile['status'] == 'COMPLETED' and supervisor['exit_code'] == 0
    assert len(budget['jobs']) == 1 and budget['new_cumulative_seconds'] < 1800
    assert cpu['cumulative_seconds'] < 900 and all(s['exit_code'] == 0 for s in cpu['stages'])
    assert table['checks']['passed'] and proof['status'] == 'PASS'
    assert 'Ran 4 tests' in (RUN / 'numerics.log').read_text() and '\nOK' in (RUN / 'numerics.log').read_text()
    assert env['torch'] == old_env['torch'] and env['cuda_build'] == old_env['cuda_build']
    checks['same_torch_and_cuda_build_as_V01'] = True
    indexed = {(m['seed'], m['condition'], m['step']): m for m in metrics}
    expected_keys = {(s, c, t) for s in SEEDS for c in CONDITIONS for t in STEPS}
    assert len(metrics) == len(indexed) == 24 and set(indexed) == expected_keys
    trajectories = profile['trajectories']
    assert [(r['seed'], r['condition']) for r in trajectories] == [(1729, c) for c in CONDITIONS] + [(2718, c) for c in reversed(CONDITIONS)]
    checkpoints = {}
    total_steps = 0
    total_targets = 0
    for r in trajectories:
        s, c = r['seed'], r['condition']
        init = proof['seeds'][str(s)]
        assert r['status'] == 'COMPLETED' and r['steps'] == 767 and r['train_valid_targets'] == 389636
        assert r['before_hashes'] == r['after_hashes'] == init['conditions'][c]['initial_hashes']
        assert r['common_initial_state_sha256'] == init['common_state_sha256']
        assert r['block_order_sha256'] == init['order_sha256']
        assert r['parameters_trainable'] == 444928 and r['all_common_parameters_updated'] and r['first_step_nonzero_gradient']
        assert r['code_dtype'] == 'torch.float32' and r['bit_packed'] is False
        step_events = [e for e in events if e['event'] == 'step' and e['seed'] == s and e['condition'] == c]
        assert [e['step'] for e in step_events] == list(range(1, 768))
        assert all(e['valid_targets'] == 508 and e['cumulative_targets'] == e['step'] * 508 for e in step_events)
        assert math.isclose(sum(e['nll_sum'] for e in step_events), r['train_online_nll_sum'], rel_tol=0, abs_tol=1e-8)
        assert math.isclose(sum(e['seconds'] for e in step_events), r['train_seconds'], rel_tol=0, abs_tol=1e-8)
        for t in STEPS:
            m = indexed[s, c, t]
            assert m['dev_valid_targets'] == 62601 and m['train_valid_targets'] == t * 508
            assert m['nll'] == m['nll_sum'] / 62601
            assert m['context_gain'] == m['dev_unigram_nll'] - m['nll']
            match = [e for e in events if e['event'] == 'evaluation' and (e['seed'], e['condition'], e['step']) == (s, c, t)]
            assert len(match) == 1 and all(match[0][key] == value for key, value in m.items())
        # Hash the new local save; never deserialize a checkpoint, old or new.
        checkpoint = ROOT / r['checkpoint_path']
        assert checkpoint.parent == RUN / 'private'
        checkpoints[checkpoint.relative_to(ROOT).as_posix()] = verify_file(checkpoint, {'bytes': r['checkpoint_bytes'], 'sha256': r['checkpoint_sha256']})
        total_steps += len(step_events)
        total_targets += sum(e['valid_targets'] for e in step_events)
    assert total_steps == 6136 and total_targets == 3117088
    checks.update(all_trajectories_complete=True, raw_steps_and_target_denominators_match=True,
                  frozen_hashes_unchanged_and_common_parameters_updated=True,
                  original_initial_states_and_block_orders_reused=True,
                  metrics_raw_events_and_resources_agree=True, new_checkpoint_bytes_sha_verified_without_loading=True)

    bridge = []
    old_index = {(m['seed'], m['condition'], m['step']): m for m in old}
    for s in SEEDS:
        for c, previous in (('C', 'Fp'), ('S', 'Sp')):
            for t in STEPS:
                now = indexed[s, c, t]['nll']
                before = old_index[s, previous, t]['nll']
                bridge.append({'seed': s, 'condition': c, 'v01_condition': previous, 'step': t,
                               'v02_nll': now, 'v01_raw_nll': before, 'signed_difference': now - before,
                               'absolute_difference': abs(now - before), 'tolerance': 0.001,
                               'passed': abs(now - before) <= 0.001})
    bridge_max = max(b['absolute_difference'] for b in bridge)
    contrast_rows = []
    for s in SEEDS:
        for t in STEPS:
            losses = {c: indexed[s, c, t]['nll'] for c in CONDITIONS}
            values = contrasts(losses)
            contrast_rows.append({'seed': s, 'step': t, **{c + '_nll': losses[c] for c in CONDITIONS},
                                  **{k: values[k] for k in CONTRAST_NAMES},
                                  **values['identity_errors'], 'max_abs_identity_error': values['max_abs_identity_error']})
    evidence = 'VALID' if all(b['passed'] for b in bridge) else 'BRIDGE_UNRESOLVED'
    final_rows = [r for r in contrast_rows if r['step'] == 767]
    decision = classify(final_rows, tau=0.05) if evidence == 'VALID' else {'label': 'NOT_ASSESSED'}
    statuses = {'EVIDENCE': evidence, 'NUMERICS': 'PASS', 'NORM_DIRECTION': decision['label'],
                'RESOURCE': 'WITHIN_30MIN', 'PAPER_READINESS': 'NOT_ESTABLISHED', 'NEXT': NEXT}
    screening = {'statuses': statuses, 'primary_endpoint': 767, 'exploratory_after_V01_dev': True,
                 'gates': checks, 'bridge_max_absolute_difference': bridge_max,
                 'bridge_tolerance_nats_per_token': 0.001,
                 'rules_source': 'runs/v02_20261022_193903/snapshot/reports/v02/EXPERIMENT_CONTRACT.md',
                 'rules': {'A': 'both seeds: G>.05, abs(D_r)<=.05, abs(D_1)<=.05, N_cont>.05, N_sign>.05',
                           'B': 'both seeds: D_r>.05 and D_1>.05',
                           'C': 'both seeds: abs(D_r)<=.05 and D_1>.05; A not satisfied', 'D': 'all other complete valid patterns'},
                 'primary_input': final_rows, 'classification': decision,
                 'no_additional_G_gate_for_B_or_C': True,
                 'not_significance_equivalence_causal_mediation_or_novelty': True}
    csv_write('metrics.csv', sorted(metrics, key=lambda m: (m['seed'], CONDITIONS.index(m['condition']), m['step'])))
    csv_write('contrasts.csv', contrast_rows)
    csv_write('bridge.csv', bridge)
    write(REPORT / 'screening.json', screening)
    write(REPORT / 'checkpoint_hashes.json', {'policy': 'new V02 files hashed as bytes only, no checkpoint loaded; excluded from ZIP', 'files': checkpoints})

    final_table = markdown_table(['seed', 'C', 'U', 'RS', 'S'], [[s] + [f'{indexed[s,c,767]["nll"]:.12f}' for c in CONDITIONS] for s in SEEDS])
    contrast_table = markdown_table(['seed', *CONTRAST_NAMES], [[r['seed']] + [f'{r[k]:+.12f}' for k in CONTRAST_NAMES] for r in final_rows])
    gain_table = markdown_table(['seed', 'C gain', 'U gain', 'RS gain', 'S gain'], [[s] + [f'{indexed[s,c,767]["context_gain"]:.12f}' for c in CONDITIONS] for s in SEEDS])
    norm_table = markdown_table(['表', 'global RMS', '训练 pi 加权 RMS', '单表 FP32 bytes'], [[c, f'{table["tables"][c]["global_RMS"]:.12f}', f'{table["tables"][c]["training_pi_weighted_RMS"]:.12f}', table['tables'][c]['bytes']] for c in CONDITIONS])
    error_table = markdown_table(['恒等检查', '最大绝对误差', '最大非零参考相对误差', '通过'], [[k, f'{v["max_abs_error"]:.12g}', f'{v["max_relative_error_on_nonzero_reference"]:.12g}', str(v['passed'])] for k, v in table['checks']['actual_errors'].items()])
    timing_table = markdown_table(['seed/表', '同步训练 s', '完整 dev 评价 s', '轨迹墙钟 s', 'PyTorch allocated/reserved bytes', '全设备峰值 MiB', '进程采样 RSS bytes'], [[f'{r["seed"]}/{r["condition"]}', f'{r["train_seconds"]:.6f}', f'{r["eval_seconds"]:.6f}', f'{r["wall_seconds"]:.6f}', f'{r["torch_peak_allocated_bytes"]}/{r["torch_peak_reserved_bytes"]}', r['device_global_peak_used_mib'], r['cpu_sampled_peak_rss_bytes']] for r in trajectories])
    trajectory_table = markdown_table(['seed', 'step', 'C', 'U', 'RS', 'S'], [[s,t] + [f'{indexed[s,c,t]["nll"]:.12f}' for c in CONDITIONS] for s in SEEDS for t in STEPS])
    all_contrasts = markdown_table(['seed', 'step', *CONTRAST_NAMES], [[r['seed'],r['step']] + [f'{r[k]:+.12f}' for k in CONTRAST_NAMES] for r in contrast_rows])
    q = table['r_quantiles']
    numerical_max = max(r['max_abs_identity_error'] for r in contrast_rows)
    text = '```text\n' + '\n'.join(f'{k} = {v}' for k,v in statuses.items()) + '\n```\n' + f'''

# VocabPriorBench V0.2：输出表行范数 × 方向

## 主要终点与唯一判定

全部 NLL 均为同一完整 dev 的 nats/token，分母 62,601；主要终点固定 step767。原始分子、分母及全精度浮点值见 metrics.csv 与 metrics_raw.json。四表共享原连续冻结输入和同一冻结 FP32 log(pi)。

{final_table}

有符号差值：G=S−C；D_r=RS−C；D_1=S−U；N_cont=U−C；N_sign=S−RS；I=D_1−D_r=N_sign−N_cont。正方向差意味着 sign 方向 dev NLL 更高。

{contrast_table}

**判定为 {decision['label']}。** 两个 seed 的 D_r、D_1 均严格超过预先固定 tau=.05；A 所需的两方向差处于容忍区间及两条正 norm effect 均未满足；C 所需 abs(D_r)≤tau 也未满足。逐项布尔值和失败原因见 screening.json，按 A→B→C→D 判定，B/C 未增加 G 门槛。两个 seed 分别报告，没有均值筛选或显著性检验。

在这组有限步对照中，恢复原 L2 行范数没有消除 sign 差距：RS 比 C 高约 .179/.185，S 比 U 高约 .170/.175。将连续方向单位化只改变约 −.000899/−.001081；将 sign 从原范数改为单位范数改变约 −.010756/−.010576，均为降低 loss。因而“差距只来自丢失原 L2 行范数”这一具体解释不足以解释本轮终点。RS 略差于 S，不支持“恢复原范数即可解决”的工程筛选判断。

G=N_sign+D_r=D_1+N_cont 及 I 的两种表达在全部六个评价格的最大绝对误差为 {numerical_max:.12g}。这只是差值代数恒等式，不是因果中介或语义解释比例。

## 支持范围、竞争解释与研究去留

当前结果保留“在原行范数和单位行范数两种匹配下，sign 方向仍有同向预测代价”这一探索性现象。它不识别语义损失或唯一机制，不排除有限步优化、输出函数族变化或其他尺度方案。r 是原 L2 行范数，非最小二乘最优重建尺度；本轮没有第五臂或事后校准。

r*S 的逐行缩放通常改变输出列空间，四格不能称为完全相同函数类。存在可训练输出投影 P 时，可逆右变换 C→C*A、P→A^(-1)*P 可保持函数却改变原表行余弦；原表 margin 是参数化相关代理量，不能自动赋予因果或语义意义。同公共初态、同步数是配对控制，不能证明模型已优化收敛。单 donor、单语料、单宽度、两个 seed 和一次短预算仅支持本设置的探索判断，token/块不是独立训练重复。

**当前冻结/sign 差距不足以确立 CAS 一区或 CCF A/B 独立论文主线。** 两篇指定近邻已包含范数恢复、符号加浮点幅度、norm/direction 控制和保留文本增益预测。dense 固定输出表、重新训练随机学生的设置差别与残差存在本身均不等于实质贡献。RESEARCH_DECISION.md 建议暂存当前解释路线，保留工程资产与本轮负控制结果，停止继续加基础诊断。本轮已结束，未进入 V0.3、V1、margin 回归或投稿写作。

## 相对训练 unigram 的开发集预测收益

复用的训练 unigram 基线 dev NLL={metrics[0]['dev_unigram_nll']:.15f}。gain=该基线 NLL−模型 dev NLL，仅表示模型相对无上下文训练 unigram 的预测收益，不等于语义信息量或因果贡献。

{gain_table}

step0 各表 gain 均为负，step100/767 均为正。四表公共初态相同不要求初始 logits 相同；没有投影清零、温度或 RMS 调整来匹配 step0。

## 来源核对、实际实现与封存

完整指令 CODEX_V02_VocabPriorBench.txt、配置及契约先于本轮结果封存。八个必要输入按 V0.1 source_manifest 的路径规范化后核对 bytes/SHA；CPU weights_only=True 加载输入并重算两份公共 state SHA，与原 data_selection 的记录一致。prior 复用原 prepared.pt，校验统计不替换输入；两 seed 原 block_order 是原选中 3,068 个完整块的排列。本轮没有重抽样、重生成初态、重新排列、重做 PCA、重打包或读取训练终点。input_proof.json 与 V01_INPUT_EVIDENCE_AUDIT.md 给出来源证明。

原初 C/S tensor hash 与 V0.1 Fp/Sp initial output hash 相同。Student/OutputStudent 的三个依赖逐字节复制自 V0.1 实际 snapshot；本轮使用显式 tables[condition]、统一 trainable_output=False 和固定 logpi，避免旧标签前缀/后缀分派。新旧 runner 精确 diff 已保存；causal mask、标签 shift、交叉熵分母、梯度累积核心数值逻辑沿用原执行版。

训练前快照树 SHA256：`{tree}`（16 文件）。实际执行来自 runs/v02_20261022_193903/snapshot，GPU job_start 记录同一树 hash；本次核算再次验证全部快照文件未变。analysis.py/finalize.py 是指令允许在运行后编写的报告代码，单独封存于 report_snapshot，未冒充训练前快照；其分类式严格来自旧契约。

V01_ERRATA.md 补充三点且保留全部历史原文：训练 pi 加权投影 margin 与候选稿未加权最大余弦 margin 不同；候选未入原训练前快照，现包不能独立核实其自述盲态，不能称独立预注册；旧逐轨迹 UTC 记录位置晚于 monotonic，外部总进程时间仍有效。本轮 UTC/monotonic 在同一位置记录，不为勘误重跑旧实验。

## 四表构造与 CPU 数值检查

C 是 artifacts/codes.pt 原始 FP32 output；r=CPU FP64 L2(C)；S=where(C>=0,1.,−1.)/sqrt(128) 的精确旧 FP32 表达式，sign(0)=+1。U=C.double()/r、RS=r*S.double() 在 CPU FP64 计算后各转 FP32 一次；零行 U=S、RS=0，不夹 epsilon。bias 始终在表分数之后加，未乘 r。

实际零行=0、0<r<1e−12 行=0，其 ID/目标数列表为空；toy 检查覆盖非单位 r、零行、极小正行、负零与零坐标。原 r 分位数：min={q['min']:.12f}、p1={q['p1']:.12f}、p25={q['p25']:.12f}、median={q['median']:.12f}、p75={q['p75']:.12f}、p99={q['p99']:.12f}、max={q['max']:.12f}。完整 50,304 类保留，含 EOS0 和 tokenizer 未使用行 50277–50303（27 行）。

{norm_table}

固定 rtol=2e−6、atol=2e−7 的联合容差全部通过；未因结果放宽。RS 范数最大绝对误差略高于 atol，但最大相对误差约7.62e−8，满足原联合容差。

{error_table}

CPU 准备与4项新增必要测试全部通过。独立只读 CPU 审计20项一致性检查通过；没有重跑旧整套模型检查、V×V、全词表近邻、逐 token 挖掘或拟合预测器。四条件 input/bias/common 初态相同、storage 独立、实际 output hash 对应正确表；训练前后 input/output/bias hash 不变，冻结项无梯度，444,928 公共参数有有限非零梯度且实际更新。原始 CPU 输出与运行时断言、before/after hash 均在包内。

## 限定运行、原始核算与 V0.1 桥接

运行顺序：1729 C/U/RS/S，2718 S/RS/U/C。每条从该 seed 保存的公共随机初态开始，用新 optimizer，严格767步、每步508有效目标、389636目标，一次遍历3068块。合计 {total_steps} optimizer steps、{total_targets} 更新目标、24次完整 dev 评价；逐步分子/分母与资源汇总全部核对一致。主要终点未从最好 checkpoint 挑选。新 checkpoint 仅保存并按字节核验 SHA，未加载；旧 final checkpoint 从未加载。

2层、4heads、hidden/code128、FF512、seq128、dropout0；mb2×accum2；AdamW lr3e−4 constant、wd.01、betas(.9,.999)、eps1e−8、fused=True、clip1；BF16 autocast、FP32参数/表/bias，TF32=False，无 scheduler/训练热身。仅一次合成32token有限性预检、零optimizer更新，其耗时含在同一 GPU 完整进程内。

C↔V0.1 Fp、S↔V0.1 Sp 在0/100/767的12个 raw NLL 桥接格全部通过，最大绝对差={bridge_max:.12g}，容忍=.001，仅诊断版本/执行漂移。V0.1与本轮 PyTorch/CUDA build 相同，模型依赖 bytes、初态、order、prior、输入及 C/S 输出锚点一致；无需解释超限漂移，无重跑。bridge.csv 保留逐格原值。

## 实际资源

仅一次 GPU 完整子进程，UTC {budget['jobs'][0]['supervisor_start_utc']} 至 {budget['jobs'][0]['supervisor_end_utc']}，exit0；外部 monotonic 实测 **{budget['new_cumulative_seconds']:.12f}秒 / 1800秒**，含导入、初始化、预检、训练、评价、保存与退出。没有失败作业或重试，未触发预算终止。runner 墙钟={profile['runner_elapsed_seconds']:.12f}秒；不能用下表训练秒数替代预算。

V0完整GPU=19.406秒；V0.1完整GPU=312.71703610004624秒；此前累计332.12303610004624秒；项目累计 **{budget['project_cumulative_seconds']:.12f}秒**。CPU准备/检查完整子进程累计 **{cpu['cumulative_seconds']:.12f}秒 / 900秒**（prepare17.9689173、numerics5.8037943）；另有报告纯算术合成边界检查内部0.0632873秒，非GPU作业、非全进程计时。报告核算/文件哈希耗时另见 POST_RUN_VERIFICATION.json，不计为训练。

{timing_table}

设备：{env['device']}，8188MiB；Windows11、Python3.12.15、PyTorch{env['torch']}、CUDA build{env['cuda_build']}、cuDNN{env['cudnn']}、driver616.92；环境未升级。nvidia-smi 是含桌面/其他应用的全设备采样，PyTorch allocated/reserved 为本进程 allocator 口径；CPU RSS 为约1秒加smi延迟的采样，不是精确峰值。同步步吞吐是本次短运行观察，不外推持续热稳态。单输出表25,755,648bytes；input+output共51,511,296bytes，bias201,216bytes；全为FP32，bit_packed=false，普通浮点matmul，未实现1bit GPU存储或新kernel。

## 两个描述性评价点与完整轨迹

{trajectory_table}

{all_contrasts}

step0/100仅作描述，不修改767主要终点与tau。不把两个seed短程结果称为统计显著、统计等价、稳定扩展规律或新方法。

## 文献、来源边界与交付

定向原始全文：[N-Gram Memories at Two Bits v3](https://arxiv.org/html/2610.21810v3)，版本2026-10-21，§4.3/E.2/F.2/F.3已比较全局/逐行LS/原范数恢复，区分norm/direction并检验保留文本增益预测；[OneBit NeurIPS2024 §3](https://proceedings.neurips.cc/paper_files/paper/2024/file/7a7a3f53faafc0161be0fcb57e5fa078-Paper-Conference.pdf)已有sign+浮点幅度表示。准确对象差异与预测器失败边界见一页 LITERATURE_DELTA.md 和 literature_sources.json。两文访问成功；无开放式综述或作者代码复现。

方法技能 experimental-design/scientific-critical-thinking 用于配对设计与证据边界，不是实验结果来源；技能来源沿用已核验记录：Kassis、Agarwal、He、Patel、Brueckner（2026），[Scientific Agent Skills](https://arxiv.org/abs/2609.00065)，v2，DOI10.48550/arXiv.2609.00065；没有已核验期刊出版商DOI，不将技能当作近邻研究创新证据。

无新模型/数据/宽度/seed、无下载/环境修改；未读取或重新哈希holdout。审查包含实际训练快照/配置/契约、CPU日志、events/gpu_console原始日志、raw metrics、监控、supervisor账本、顺序证明、checkpoint哈希、V0.1必要原文/源码/日志、逐文件bytes/SHA。排除环境目录、全部权重/码表tensor、private、原始/分词语料和凭据。包内manifest不自引用；ZIP SHA在外部。包不是含私有tensor的独立恢复训练状态包。

实际命令见 [COMMANDS.md](../../runs/v02_20261022_193903/COMMANDS.md)。主要交付为本文件及项目根 V02_REVIEW_BUNDLE.zip，外部校验 V02_REVIEW_BUNDLE.zip.sha256。完成 V0.2，停止于研究去留判断。
'''
    (REPORT / 'V02_REPORT.md').write_text(text, encoding='utf-8')
    commands = '''# V0.2 实际执行命令（已完成，不是续训指令）

工作目录 `F:\\VocabPriorBench`，PowerShell；已有 `.venv`，未安装/升级依赖。

```powershell
$env:VPB_WORKSPACE='F:\\VocabPriorBench'
& .\\.venv\\Scripts\\python.exe scripts\\v02\\create_runner.py
& .\\.venv\\Scripts\\python.exe -m compileall -q scripts\\v02 tests\\test_v02_tables.py
& .\\.venv\\Scripts\\python.exe scripts\\v02\\seal.py
& .\\.venv\\Scripts\\python.exe runs\\v02_20261022_193903\\snapshot\\scripts\\v02\\cpu_supervise.py prepare
& .\\.venv\\Scripts\\python.exe runs\\v02_20261022_193903\\snapshot\\scripts\\v02\\cpu_supervise.py numerics
& .\\.venv\\Scripts\\python.exe runs\\v02_20261022_193903\\snapshot\\scripts\\v02\\supervise_gpu.py
& .\\.venv\\Scripts\\python.exe scripts\\v02\\finalize.py build
& .\\.venv\\Scripts\\python.exe scripts\\v02\\finalize.py pack
```

GPU supervisor 账本另保存实际 child 完整 argv、PID、UTC、exit code 和完整墙钟。
CPU supervisor 账本另保存 prepare 与 unittest 的完整 argv/起止/耗时。
封存脚本和 GPU supervisor 会拒绝复用现有执行输出；本轮已结束，不重跑。
finalize.py 是训练后报告/校验/打包代码，未替换训练前契约或执行快照。
'''
    (RUN / 'COMMANDS.md').write_text(commands, encoding='utf-8')
    report_snapshot = RUN / 'report_snapshot'
    report_files = {}
    for name in ('analysis.py', 'finalize.py'):
        source = ROOT / 'scripts/v02' / name
        target = report_snapshot / 'scripts/v02' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        report_files[target.relative_to(report_snapshot).as_posix()] = spec(target)
    write(report_snapshot / 'REPORT_CODE_MANIFEST.json', {'created_after_GPU_run': True, 'utc': utc(), 'files': report_files, 'training_snapshot_not_modified': True})
    write(REPORT / 'POST_RUN_VERIFICATION.json', {'utc': utc(), 'statuses': statuses, 'checks': checks,
          'training_snapshot_tree_sha256': tree, 'input_files': verified_inputs,
          'raw_event_count': len(events), 'train_step_count': total_steps, 'train_target_count': total_targets,
          'evaluation_count': len(metrics), 'bridge_count': len(bridge), 'bridge_max_absolute_difference': bridge_max,
          'raw_evidence_files': {p.relative_to(ROOT).as_posix(): spec(p) for p in [RUN/'metrics_raw.json', RUN/'events.jsonl', RUN/'gpu_console.log', REPORT/'source_manifest.json']},
          'report_code_files': report_files, 'postprocessing_elapsed_seconds': time.perf_counter() - begin,
          'scope': 'CPU stdlib arithmetic and file-byte hashing only, no tensor deserialization/CUDA/predictor/holdout'})
    print(json.dumps({'stage': 'build', 'statuses': statuses, 'bridge_max': bridge_max, 'final_contrasts': final_rows}, ensure_ascii=False, indent=2))


def pack():
    bundle = ROOT / 'V02_REVIEW_BUNDLE.zip'
    assert not bundle.exists(), 'Do not overwrite an existing review bundle'
    required = ['V02_REPORT.md', 'RESEARCH_DECISION.md', 'RAW_RESULTS_AUDIT.md', 'CPU_NUMERICS_AUDIT.md',
                'metrics.csv', 'contrasts.csv', 'bridge.csv', 'screening.json', 'POST_RUN_VERIFICATION.json']
    assert all((REPORT / name).is_file() for name in required)
    paths = set()
    allowed_suffixes = {'.md', '.json', '.jsonl', '.csv', '.log', '.py', '.yaml', '.txt', '.diff'}
    for folder in (REPORT, RUN):
        for path in folder.rglob('*'):
            if not path.is_file() or path.name == 'BUNDLE_MANIFEST.json':
                continue
            if any(part.lower() in {'private', '__pycache__', '.venv'} for part in path.relative_to(ROOT).parts):
                continue
            if path.suffix.lower() in allowed_suffixes:
                paths.add(path)
    # Explicit, text-only V01 provenance and raw-result references. No directory
    # sweep of old artifacts/data or old private checkpoints.
    historical = ['configs/v01.yaml', 'reports/source_manifest.json', 'reports/v01/V01_REPORT.md', 'reports/v01/EXPERIMENT_CONTRACT.md',
                  'reports/v01/source_manifest.json', 'reports/v01/data_selection.json',
                  'reports/v01/RESIDUAL_HYPOTHESIS_CANDIDATE.md', 'runs/v01_20261022/environment.json',
                  'runs/v01_20261022/metrics_raw.json', 'runs/v01_20261022/events.jsonl',
                  'runs/v01_20261022/gpu_console.log', 'runs/v01_20261022/supervisor_result.json',
                  'runs/v01_20261022/block_order_1729.json',
                  'runs/v01_20261022/block_order_2718.json', 'runs/v01_20261022/snapshot/SNAPSHOT_MANIFEST.json']
    historical += ['runs/v01_20261022/snapshot/scripts/v01/' + name for name in
                   ('prepare.py', 'run_gpu.py', 'common.py', 'supervise_gpu.py')]
    historical += [f'runs/v01_20261022/resources_{s}_{c}.json' for s in SEEDS for c in ('Fp','Sp')]
    for name in historical + ['configs/v02.yaml', 'CODEX_V02_VocabPriorBench.txt', 'requirements.txt']:
        path = ROOT / name
        assert path.is_file(), name
        paths.add(path)
    files = {path.relative_to(ROOT).as_posix(): spec(path) for path in sorted(paths)}
    write(REPORT / 'BUNDLE_MANIFEST.json', {'created_utc': utc(), 'files': files, 'manifest_excludes_itself': True,
          'zip_sha256_external_only': True, 'excluded': ['.venv', 'private', 'all tensor/weight files', 'raw/tokenized corpora', 'credentials'],
          'private_tensor_inputs_remain_local_and_are_not_recoverable_from_this_bundle': True})
    paths.add(REPORT / 'BUNDLE_MANIFEST.json')
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(paths):
            archive.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(bundle) as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())
        assert names == set(files) | {'reports/v02/BUNDLE_MANIFEST.json'}
        for name, expected in files.items():
            data = archive.read(name)
            assert len(data) == expected['bytes'] and hashlib.sha256(data).hexdigest() == expected['sha256'], name
        assert all('private' not in Path(name).parts and Path(name).suffix.lower() not in {'.pt','.bin','.safetensors','.npy','.npz','.parquet'} for name in names)
    digest = sha(bundle)
    (ROOT / 'V02_REVIEW_BUNDLE.zip.sha256').write_text(digest + '  V02_REVIEW_BUNDLE.zip\n', encoding='ascii')
    print(json.dumps({'stage': 'pack', 'files': len(paths), 'bytes': bundle.stat().st_size,
                      'sha256': digest, 'zip_and_manifest_verification': 'PASS'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['build', 'pack'])
    arguments = parser.parse_args()
    build() if arguments.stage == 'build' else pack()
