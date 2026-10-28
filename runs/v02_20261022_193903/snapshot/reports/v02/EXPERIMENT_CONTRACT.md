# V0.2 输出行范数 × 方向：事前探索契约

2026-10-22（Asia/Shanghai）。完整要求为根目录CODEX_V02_VocabPriorBench.txt。本轮已看过V0.1/dev，封存只保证本轮口径先于本轮结果，不是未见数据的确认实验。只做最后一次基础四格对照，任何结果均停止，NEXT固定PAUSE_CURRENT_CLAIM_FOR_RESEARCH_DECISION。

## 只读复用、证据与版本

工作区明确VPB_WORKSPACE=F:\VocabPriorBench，复用现有.venv/RTX4060 Laptop，不更新环境、下载、重做PCA/打包/抽样。白名单输入为artifacts/codes.pt、data/packed.pt、runs/v01_20261022/private/prepared.pt、该private下两份common_init_{1729,2718}.pt、该run两份block_order和reports/v01/data_selection.json。bytes/SHA对照V0.1 source_manifest的规范化路径；CPU weights_only=True加载，重算common state hash、order完整排列与既有hash、counts/pi/dev_counts一致性。只验证原pi，不替代原文件、重新估计或改平滑。

任一必要文件缺失、SHA/bytes/common state不符，精确列路径与项，BLOCKED_MISSING_OR_MISMATCHED_V01_INPUT，GPU=0并结束。不会用artifacts/common_init.pt、重生成seed或训练终点替代。禁止加载任何V0.1 final checkpoint。V0/V0.1原文件不变。holdout不打开、不重哈希；历史metadata可引用。不得导入旧v01/common来定位目录。

Student、OutputStudent及__init__.py从V0.1实际snapshot逐字节复制入本轮snapshot，保存来源hash。V02训练器由旧run_gpu显式diff改造，保留CE、shift/mask、有效目标、累积、优化器等核心数值循环；仅新路径、表dispatch、冻结设置、计时及预算适配。实际从新snapshot执行；模型/配置/脚本/新测试/完整指令/本契约均封存。派生表、检查和输入证明在CPU生成后另存hash并在GPU前重验。

## 唯一表构造

C为原初始FP32 output，50304×128；r=norm(C.double(),2,dim=1)，CPU FP64。S精确 `torch.where(C>=0,1.,-1.)/math.sqrt(128)`，FP32且sign(0)=+1。非零行U=C.double()/r；严格零行U=S.double()；RS=r*S.double()。U/RS最后转FP32，C/S原FP32值不动。正极小行按真实FP64范数除，不clamp。不额外RMS/中心化/白化/旋转/温度或频率能量对齐。r是L2范数，不是重建最小二乘最佳尺度。

| 条件 | 方向 | 行范数 |
|---|---|---|
| C | 连续 | 原r |
| U | 连续 | 单位（零行用S方向） |
| RS | sign | 原r |
| S | sign | 单位 |

显式 `tables={C:C,U:U,RS:RS,S:S}`，`table=tables[condition]`，四条件统一OutputStudent(input_codes,table,False,logpi)。不得用condition前缀/后缀推断表或先验。输入原连续表冻结；输出均冻结；共同bias为原pi.log().float()的固定buffer，logits=Tq+log(pi)，绝不把bias一起乘r。公共主干/两投影444928参数可训，模型实现不改。50304类别均保留，含EOS=0和27个tokenizer-unused输出行。

CPU统一rtol=2e-6、atol=2e-7：记录C≈rU、U/S单位范数、RS范数≈r及实际最大误差/通过状态、shape/dtype/有限/存储hash；C/S实际hash必须匹配V0.1 Fp/Sp初始表。TABLE_CONSTRUCTION.json保存公式、r来源、各表hash/RMS、原r分位数、严格零/0<r<1e-12的ID/训练目标数/特殊及unused分类，可描述pi加权能量但不调参。toy检查含非单位r/零行/零坐标、明确map与bias不缩放，固定input/output/bias无梯度不变、common有有限非零梯度且更新。沿用V0.1已验证base逻辑，不重跑整套旧实验。

## 配对、预算、评估与资源

两个seed=1729/2718，原3068完整块集合、各seed原block_order逐值复用；每条767步×508有效目标=389636，一次遍历，无额外训练热身/重复补齐。每个条件构造独立对象、加载该seed旧公共随机初态，新的optimizer，不续训。1729顺序C/U/RS/S，2718顺序S/RS/U/C。C/S也重新训练作为锚点。

hidden/code128、2层、4heads、FF512、seq128、dropout0。microbatch2×accum2，每micro254目标、每步508；AdamW lr3e-4常定、weight_decay.01、betas(.9,.999)、eps1e-8、fusedTrue，clip1，BF16 autocast、FP32参数/表/bias；TF32=False及CUBLAS_WORKSPACE_CONFIG沿用封存版。允许一次小型合成32tokenGPU形状/有限预检，无optimizer/语料，不作持续热稳态测速。step0/100/767完整dev493块62601目标，767主要终点，不选最好checkpoint。预计共6136步、3117088更新目标、24评价。

CPU准备/数值检查完整子进程累计≤900秒，外部监督；无V×V、全词表近邻、逐token挖掘或预测器拟合。新增GPU完整子进程累计≤1800秒，外部启动前计时、核验余额与全部CPU证据、预估400秒，预留60秒收尾。只一尝试，串行8条；OOM/非有限/冻结失败/预算或其他实质错误，存证并结束，不改batch/lr/精度、不自动重跑。GPU包含导入/预检/训练/评估/保存/退出/失败；纯训练秒数另列。V0=19.406、V01=312.71703610004624、既有累计332.12303610004624秒单列，项目累计加本轮完整GPU时间。

逐轨迹UTC与monotonic起点在同一代码位置（任何初始化/监测前）取得，区分步同步累计、评价、轨迹、runner、完整supervisor墙钟。记录每步分子/分母、错误、allocator allocated/reserved、全设备占用和CPU采样RSS。FP32表、bit_packed=False、实际字节数，不称1bit存储/加速。最终固定三项hash不变、无梯度；公共参数有有限梯度且更新；checkpoint保存并hash留本地，不进审查包。

## 桥接与唯一主要分析

从旧metrics_raw原精度读取C↔Fp、S↔Sp三评价点形成bridge.csv，容忍0.001 nats/token，仅诊断执行漂移，不是显著性阈值。超限时只核已有hash/初态/order/dtype/backend/版本；不调参匹配、不自动重跑。无法说明则BRIDGE_UNRESOLVED，不确认旧差距；实输入/模型不一致则INVALID_COMPARISON，机制NOT_ASSESSED。四格内部原结果照实保留。

每seed/点有符号G=S−C，D_r=RS−C，D_1=S−U，N_cont=U−C，N_sign=S−RS，I=D_1−D_r=N_sign−N_cont。核对G=N_sign+D_r=D_1+N_cont。它们是差值恒等式，不是因果中介分解，不命名为解释了百分之多少语义。两seed分别报告，不用均值藏冲突；step0/100仅描述。tau=.05 nats/token是探索容忍值，不是统计等价、显著性或发表标准。

只有八条完整有效且桥接无未解漂移时，固定A→B→C→D优先级：

A NORM_DOMINANT_IN_THIS_SCREEN：两seed各满足G>tau、abs(D_r)≤tau、abs(D_1)≤tau、N_cont>tau、N_sign>tau。只说明两种匹配范数下方向差在容忍区间、移除r均增loss；停止把原差距作为独立论文核心，不把恢复范数称创新。

B DIRECTION_RESIDUAL_AT_BOTH_NORMS：两seed各D_r>tau且D_1>tau。只说明原/单位两范数下sign仍有同向代价，不证明语义损失、唯一机制、全部尺度已排除或论文成立。

C INTERACTION_OR_COMPENSATION：两seed各abs(D_r)≤tau、未满足A，且两seed各D_1>tau。只说恢复r在该格补偿方向变化，报告I及两norm effects，不称完整解释。

D INCONCLUSIVE_OR_OTHER_PATTERN：所有其他模式，含seed冲突/反向/单范数残差/G不再>tau；G不另加作B/C门槛。不追加实验、调整阈值或挑结论。BLOCKED/INVALID/INCOMPLETE/BRIDGE_UNRESOLVED优先，机制NOT_ASSESSED。

## 研究决策与停止

补充V01_ERRATA而不改历史：两种margin定义不同；未封存候选的盲态不可由现包独立验证，非独立预注册；旧轨迹起点UTC记录晚于monotonic，外部总时有效，本轮修正不重跑。定向两篇近邻已含norm恢复、符号+幅度与保留预测，设置差异不等创新。

rS的行缩放通常改变列空间，四格非同函数族；C→CA与P→A⁻¹P可保函数却改余弦几何，margin非天然因果/语义量。同初态/步数不证明优化收敛；单donor/语料/宽度、2seed短预算仅探索。原L2控制不代表所有尺度方案。RESEARCH_DECISION一页回答当前差距是否足以独立高质量论文主线，默认不因残差而是。若无精确可证伪且区别近邻的问题则暂存路线，保留工程/负结果，结束基础诊断。不运行margin回归，不打开holdout，不附自动V0.3/V1，不询问续训。
