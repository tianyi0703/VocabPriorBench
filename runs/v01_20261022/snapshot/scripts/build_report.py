"""Build the final report only from completed local measurements and saved evidence."""
import json,math,subprocess,sys,time
from pathlib import Path
from provenance import sha
root=Path(__file__).resolve().parents[1]
def read(path):return json.loads((root/path).read_text(encoding='utf-8'))
r=read('reports/resource_profile.json');m=read('reports/source_manifest.json');d=read('data/manifest.json');audit=read('runs/artifact_audit.json')
assert r['status']=='COMPLETED' and audit['run_accounting']=='PASS'
arms=r['arms'];env=read('runs/v0_20261021/environment.json');ins=m['donor_inspection'];cfg=r['config'];g=r['gpu_budget']['cumulative_seconds']
paper_ids=['2610.22428v1','2407.05841v1','2608.03494v1','2605.09751v2','2610.04002v1','2304.01373v2','2004.12073v3','1711.01068v2']
m['papers']=[{'arxiv':p,'full_text_url':f'https://arxiv.org/html/{p}','verification':'original full text read; not author-experiment reproduction'} for p in paper_ids]
m['seed_official_resource']={'url':'https://huggingface.co/QLNI/embeddings-are-overpriced','revision_observed':'af839128c84a5dee79017a50fe80ecbd0b21aacb','scope':'training code content + checksums read; large weights/data metadata inspected, not downloaded'}
m['literature_evidence']={str(p.relative_to(root)):sha(p) for p in (root/'reports').glob('*LITERATURE*.md')}
for name in ['literature_seed.md','literature_neighbors.md']:m['literature_evidence']['reports/'+name]=sha(root/'reports'/name)
m['execution_environment']=env;m['delivery_source_files']={str(p.relative_to(root)):sha(p) for folder in ['src','scripts','tests','configs'] for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)}
m['requirements_sha256']=sha(root/'requirements.txt')
m['data']['license']='Pinned card YAML: cc-by-sa-3.0 + gfdl; Licensing Information prose: CC-BY-SA-4.0'
m['data']['license_version_status']='SOURCE_INTERNAL_CONFLICT; exact version unresolved; source card kept verbatim'
(root/'reports/source_manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
slow=min(a['stable_valid_tokens_per_second'] for a in arms)
planning=slow/4
# Conditional 12h plan: 4h reserve and 8h training / 42 runs, 508 tokens per step.
steps=math.floor((8*3600*planning)/(42*508));tokens=steps*508
table='\n'.join(f"| {a['arm']} | {a['steps']} | {a['train_valid_tokens']:,} | {a['train_online_nll']:.6f} | {a['initial_train']['nll']:.6f} → {a['final_train']['nll']:.6f} | {a['initial_dev']['nll']:.6f} → {a['final_dev']['nll']:.6f} |" for a in arms)
resources='\n'.join(f"| {a['arm']} | {a['wall_seconds']:.2f} | {a['train_seconds']:.2f} / {a['eval_seconds']:.2f} / {a['warmup_seconds']:.2f} | {a['stable_valid_tokens_per_second']:.1f} | {a['torch_peak_allocated_bytes']/2**20:.1f} / {a['torch_peak_reserved_bytes']/2**20:.1f} | {a['device_peak_used_mib_global']:.0f} | {a['cpu_peak_rss_bytes_sampled']/2**20:.1f} | {a['checkpoint_bytes']/2**20:.2f} |" for a in arms)
params='\n'.join(f"| {a['arm']} | {a['parameters_unique']:,} | {a['parameters_trainable']:,} | {a['code_gpu_bytes']/2**20:.3f} |" for a in arms)
lit=(root/'reports/LITERATURE_VERDICT.md').read_text(encoding='utf-8')
lit=lit[lit.index('## 统一差异表'):]
if '\n方法工具说明：' in lit:lit=lit.split('\n方法工具说明：')[0]
text=f'''# VocabPriorBench V0 实际执行报告

ENVIRONMENT = VERIFIED_LOCAL_CUDA

LITERATURE = UNRESOLVED

IMPLEMENTATION = VERIFIED

RESOURCE = MEASURED_WITHIN_BUDGET

NEXT = RESOLVE_NOVELTY

日期：2026-10-21（Asia/Shanghai）。已完成8篇定向原始全文核查、事前实验契约、最小三臂、CPU数值检查与本机串行CUDA短跑。没有进入正式pilot、长预训练或论文撰写。文献已有实质重合，未证实完整问题已被回答，也未证明新颖；没有触发 STOP_OVERLAP。独立保留集未分词/评估。

## 1. 环境与范围

工作区 `F:\\VocabPriorBench` 起初为空且非Git仓库，`F:\\AGENTS.md` 和项目内 AGENTS.md 均未发现；未修改旧研究。Windows11 HP OMEN，系统物理内存约31.7GiB（标称32GB）。F盘开始约655GiB可用。默认 `E:\\annaconda\\python.exe` 为 Python3.13.5/torch2.10.0+cpu，本轮只读检查，没有借其结果冒充GPU。新增独立 `.venv` 为 Python3.12.15，最终 torch `{env['torch']}`，CUDA build `{env['cuda_build']}`、cuDNN `{env['cudnn']}`，驱动 `{env['device_initial']['driver']}`。

实际设备 `{env['device']}`，capability {env['capability']}，显存 {env['device_initial']['total_mib']}MiB；GPU作业启动时设备占用 {env['device_initial']['used_mib']}MiB、可用 {env['device_initial']['free_mib']}MiB。背景桌面/其他应用保留运行。硬件与系统原始记录见 `runs/system_inventory.json`、`initial_nvidia_smi.csv` 和 `v0_20261021/environment.json`。未租云、未用付费训练/推理API、未转移机器、未改全局驱动或旧环境。依赖安装不是GPU作业。

## 2. 文献差异与停止判断

以下8篇全文加官方资源是定向核查，不是穷尽系统综述。两篇Bochkov论文为连续研究，不是两次独立复现。已有工作覆盖固定二进制输入、冻结压缩预训练表、随机学生重训、词频、训练预算、多语言及初始化尺度，不能把这些部件当贡献。

{lit}

## 3. 已执行实验契约及来源

事前全文见 [EXPERIMENT_CONTRACT.md](EXPERIMENT_CONTRACT.md)。实际配置为 `configs/v0.yaml`，不把V1候选写入执行配置。唯一donor为 Apache-2.0 的 Pythia-70m，最终step143000，锁定 `{m['donor']['revision']}`。实际只下载一个 `model.safetensors`，{m['donor_files'][-1]['bytes']:,} bytes；模型全部允许文件共 {sum(x['bytes'] for x in m['donor_files']):,} bytes。未下载其他检查点。

实际输入/输出表均 {ins['input_shape']}；配置 tie={ins['config_tie_word_embeddings']}，same Parameter={ins['same_parameter_object']}，same storage={ins['same_storage']}，无输出bias；按唯一参数对象总计 {ins['unique_parameters']:,} 参数。tokenizer长度 {ins['tokenizer_length']}，模型保留50304个输出类别，末尾未使用行仍保留；EOS=0、借用EOS做右pad并用mask忽略。tokenizer文件哈希、完整token→ID顺序哈希及checkpoint哈希在 source_manifest。

两端分别用 donor 原表 E∈R^(50304×512)：μ=mean(E)，Q为中心协方差512×512的前128特征向量；CPU float64精确eigh，降序、每列最大绝对值元素正号，z=(E−μ)Q·s。s使每张连续表全局RMS=1/√128；sign零约定正号，C=sign(z)/√128。保存均值、Q、s、特征值及文件哈希，没有 V×V 矩阵。两表解释方差比例：输入 {ins['input_explained_variance']:.6f}、输出 {ins['output_explained_variance']:.6f}。

前向：`H0 = C_in[ids] @ W_in + position`；`H = final_LN(causal_blocks(H0))`；`logits = (H @ W_out) @ C_out.T`。W_in/W_out均128×128且无bias；2层pre-LN、宽128、4头、FF512、dropout0、learned position。无额外unigram输出偏置。学生主干完全随机；公共主干及投影从同一保存初态逐值加载并核对内容哈希 `{m['common_init']['state_sha256']}`。donor主干未用于学生。

A两码表可训，B与A初值逐值相同但冻结，C与B同来源/同变换后sign并冻结。公共初态和训练前缀三臂哈希完全一致。A/B只隔离本接口冻结，B/C只隔离预定norm规则下的二值化；A不是原维可训词表，三臂不是种子完整复现。

| 臂 | 唯一参数 | 可训练参数 | 两码表实际GPU驻留MiB |
|---|---:|---:|---:|
{params}

三臂码表**都实际以float32存储，无bit-pack**。C的二进制语义不带来实际码表存储优势，理论bit数不是本轮显存；bf16 autocast下执行普通浮点矩阵乘法、完整词表softmax，FP32参数/AdamW状态。未开发自定义CUDA或分块CE。

## 4. 数据、数值与执行配置

数据为固定 revision `{m['data']['revision']}` 的 WikiText-2 raw **唯一train parquet**，{m['data']['file']['bytes']:,} bytes。[固定来源卡](https://huggingface.co/datasets/Salesforce/wikitext/blob/{m['data']['revision']}/README.md)的YAML许可为CC-BY-SA-3.0/GFDL，但同卡Licensing Information正文写CC-BY-SA-4.0；这是**来源内部版本冲突**，本轮未独立解决，不能声称精确许可版本已无歧义。保留原始卡和哈希；本轮仅本地研究，没有发布语料。下载整个最小train分片但只选128篇；没有下载validation/test shard或全量WikiText-103。先按一级标题恢复文章，规范化空白全篇精确去重；原分片恢复 {d['source_articles']} 篇、非空唯一 {d['source_unique_articles']} 篇，移除 {d['normalized_duplicate_articles_removed']} 个重复。取前128篇后seed1729置换，96/16/16篇 train/dev/holdout，**先文档划分再分词**。没有做近重复或donor去污染：donor与评测重叠 UNKNOWN。

训练split共 {d['token_counts']['train']['raw_tokens_including_eos']:,} 原始token（含每篇EOS）、{d['token_counts']['train']['blocks']:,} 块；dev共 {d['token_counts']['dev']['raw_tokens_including_eos']:,} token。split内连接EOS后打包，允许跨文档注意力；每块128，块末标签忽略，右pad mask同时约束source和target。每臂使用相同前400个完整块，microbatch2、累积2、有效4序列，100 optimizer steps、50,800有效next-token目标。评估固定前32 train块与前32 dev块，各 {arms[0]['final_dev']['valid_tokens']:,} 有效token，初态和末态均测；这只是预定评估前缀，未声称评估完整dev。holdout仅原文/哈希封存，未分词或调参。

AdamW lr=3e-4恒定、weight_decay=.01、clip1，单seed1729。各臂先batch1/seq32 GPU forward/backward预检（不做optimizer step），随后3次完整batch forward/backward热身（也不更新），再正式100步。100步都检查loss/grad/param有限；各公共参数张量有非零梯度且更新，A两表均更新，B/C两表内容哈希不变。不存在dev拟合PCA/频率或选配置。

CPU `check_numerics.py` 的7项测试全部通过：因果未来扰动/截断；padding key扰动和全pad有限；标签移位/手写有效token NLL及梯度；A/B公共初态与前向逐值相同；三臂公共参数/冻结码更新规则；独立存储与唯一计数；hash稳定和无alias。GPU前另用实际50304词表CPU前向核对A/B差为0。没有分块CE，因此分块CE对照不适用。`artifact_audit.json` 另核对全部文件SHA、128篇不相交、PCA正交/RMS、400块token数和三臂100步记账、最终状态无NaN/Inf。测试、审计均为CPU；短跑资源才为CUDA实测。

## 5. 真实短跑与资源

NLL为自然对数；online train是各更新前batch的token加权均值，不能与末态dev当成同一模型时点。固定train前缀末态NLL为训练内诊断。A/B初态一致是数值核查，短跑最终排序不用于胜负主张。

| 臂 | steps | 训练有效token | online train NLL | 固定train NLL 初→末 | 固定dev NLL 初→末 |
|---|---:|---:|---:|---:|---:|
{table}

| 臂 | 整臂墙钟s | 训练/评估/热身s | 稳态有效token/s | 峰值allocated/reserved MiB | 设备全局峰值MiB | CPU RSS近似峰值MiB | checkpoint MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
{resources}

稳态窗口取step11–100，共45,720有效token/臂，计时前后CUDA同步，含梯度/有限性检查和optimizer。先3次forward/backward热身，再排除头10个训练step；窗口实际只有约2–3秒，**只是短窗口稳定阶段估计，未验证持续热稳态吞吐**。后台桌面及GPU升频可能影响数值，不能把它称为长训速度。训练时长为100步同步计时总和；评估含初/末train/dev；整臂另含构造、同步、哈希、采样、保存等。设备级采样为nvidia-smi**全GPU占用**，含其他应用，不能归因给本进程；PyTorch allocated/reserved为进程allocator峰值。CPU为约0.5秒加查询延迟采样RSS近似峰值，不保证捕获瞬时峰值。最终.pt保留完整两张浮点表，因此三臂大小相近。

本轮只有**1个串行GPU作业**。runner记录的CUDA执行区间UTC `{r['gpu_budget']['jobs'][0]['start_utc']}` → `{r['gpu_budget']['jobs'][0]['end_utc']}`，内部墙钟11.211秒；外部supervisor测得完整进程区间 **{g:.3f}秒（{g/60:.3f}分钟）**，累计预算采用更保守的外部值，硬上限7200秒。外部区间包含child导入/启动，内部区间包含全部预检、热身、评估、CUDA初始化、保存及CPU间隙。原版supervisor只把耗时打印到真实工具输出，本次将其19.406秒（原输出精度0.001秒）封存为supervisor_result.json，并保留runner原始11.211秒日志，没有重跑。监督器现已改为未来运行自动保存；不伪造首次外部起始UTC。OOM={r['oom_count']}，CUDA运行错误={len(r['errors'])}，GPU失败重跑=0。未用满预算，也未扩模型。原始逐步数据/命令和资源见下节。

## 6. 可执行命令与交付

安装详见 [README.md](../README.md) 及 `runs/install_torch.log`、`install_dependencies.log`。实际核心命令：

```powershell
Set-Location F:\\VocabPriorBench
& .\\.venv\\Scripts\\python.exe scripts\\check_numerics.py
& .\\.venv\\Scripts\\python.exe -u scripts\\download.py
& .\\.venv\\Scripts\\python.exe -u scripts\\prepare.py
& .\\.venv\\Scripts\\python.exe scripts\\audit_artifacts.py
& .\\.venv\\Scripts\\python.exe -u scripts\\supervise_v0.py
& .\\.venv\\Scripts\\python.exe scripts\\audit_artifacts.py
& .\\.venv\\Scripts\\python.exe scripts\\build_report.py
```

`runs/COMMANDS.md`记录实际命令/顺序与CPU辅助检查；`runs/numerical_checks.log`、`prepare.log`、`download_verification.log`保留原始输出；`runs/gpu_console.log`和`v0_20261021/events.jsonl`保存种子配置关联、每步NLL/token/同步时间、预检与起止；`resources_A/B/C.json`和`monitor_A/B/C.json`是各臂原始资源；最终状态也在该run目录。`reports/resource_profile.json`、`gpu_budget.json`保存完整测量；`source_manifest.json`保存论文、模型/tokenizer/数据/PCA/源码哈希与环境；`data/manifest.json`保存文档顺序与所有有效token统计。代码按内容寻址，原工作区没有Git历史；执行前hash和最终交付hash分别保留。

## 7. 条件性 V1 候选设计（计划，未运行）

**唯一建议的下一步是先解决新颖性与判别设计。** 先针对最近的组合码、固定binary及种子写出“哪些现有预测不能解释、何种保留结果会反驳本假设”的预注册差异。如果联合问题已被完整回答且无法指出实质差别，STOP_OVERLAP，不再新增GPU比较。当前是 RESOLVE_NOVELTY，不是新颖性成立，也不是启动V1的授权。

若上述审查后仍有实质候选，12 GPU小时可考虑：同donor A/B/C；D原512维可训两表、E原512维冻结两表；R同128维RMS匹配的随机冻结两表；P在训练前按**训练token频率分箱**置换B的token语义对应，保留特殊token、token ID、目标和频率先验。原donor两表不绑定，置换用同一token映射施于两表；其目的是破坏语义对应并保留行/分箱统计。D/E隔离原维冻结，A/D和B/E检查低秩接口代价（投影形状不同，不能宣称唯一因素）；R/P分别排除纯码容量和语义对应解释。对普通全维强基线同样公平调参/预算，不能只选弱基线。

最关键假设：冻结代价/二值化代价与训练词频、预先选定的码区分统计和数据预算有交互，而且其方向/排序可在未参与规则选择的第二语料上预测。使用训练频率、donor-only坐标的抽样近邻margin/碰撞率（不建V×V矩阵），预先限定模型自由度。先在来源语料拟合规则，第二语料只做一次锁定预测；同一训练轨迹按10%/30%/100%预算评估，避免把每个检查点视作独立实验。独立文章holdout本轮未用，但跨语料验证需V1新固定语料，不能把同域拆分包装成跨来源。

测得最慢核心臂稳态速度为 **{slow:.1f}有效token/s**。12h中预留1h在本机校准全维/控制臂、2h评估与检查点、1h失败/热身/系统波动，训练只分配8h。7臂×3独立seed×2语料=42条轨迹。全维臂速度本轮**没有测量**；临时按核心最慢速度的1/4，即 **{planning:.1f}token/s** 做保守规划假设，则每轨迹上限约 **{tokens:,}token（{steps:,}步×508）**，总训练约 {42*tokens:,}token。这个1/4是推断，不是4060全维实测，不保证预算；V1第一小时必须测所有接口后用 `floor(28800×实际最慢速度/(42×508))×508` 重算。若校准、评估或最小有效训练预算不容纳，则缩小主张或停止，不以其他GPU速度替代、不超过12h。此V0速度还只代表短seq128、当前CPU检查频率与后台占用；不能直接外推长上下文/成熟训练。

至少3个独立初始化/数据顺序配对seed只支持带不确定性的探索性方向，不能证明等价、广泛普适或发表质量；正式主张需预先功效/精度分析，通常还要至少5个seed并扩大独立来源，当前12h不保证足够。保持每seed跨臂公共初态和数据顺序配对；不把token/词频bin当独立seed。继续条件为强基线下有稳定且可预先预测的保留语料交互；停止条件为差异只能由尺度/接口/输出先验解释、预测无效，或预设统计精度/资源达不到。单seed100步loss是否下降都不足以支持或否定该研究问题。

## 8. 限制与结论边界

仅英文小语料、一个预训练donor、一个宽度、一个学生、一个seed、100步和固定评估前缀；没有训练原维强基线、随机/置换、多个码宽或跨来源预测。没有验证作者资源端到端完整可复现。精确整文去重不能排除近重复、实体/主题相关或donor污染。输出尾部保留的unused类别、普通浮点码表、检查开销及背景桌面使资源结果只适用于本配置。NLL下降及三臂最终排序均不构成优越、等价、创新或论文可投证据。

V0要求的实现与本地资源核查已完成；尚未解决的阻塞是实质新颖性及能够区分解释的保留验证设计。停止于V0。
'''
(root/'reports/V0_REPORT.md').write_text(text,encoding='utf-8')
print(json.dumps({'report':'reports/V0_REPORT.md','gpu_wall_seconds':g,'stable_min_token_s':slow,'conditional_v1_tokens_per_trajectory':tokens},indent=2))
