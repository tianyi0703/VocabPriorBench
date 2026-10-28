# V0.1 输出端 × 共同先验探索性筛选契约

2026-10-22（Asia/Shanghai），在CPU输出拟合和任何本轮GPU训练结果出现之前制定。以根目录 `CODEX_V01_VocabPriorBench.txt` 为完整指令；这是已见过V0/dev背景后的探索性事前约定，不是独立确认实验。seal.py 将本契约、配置、指令与实际源码复制并哈希，之后从封存目录执行；任何改动必须另存版本和被取代日志。

## 输入、数据和一次遍历

复用唯一Pythia-70m revision `a39f36b100fe8a5377810d56c3f4789b9c53ac42`、原50304词类及原PCA128两表、原tokenizer和train/dev打包。不开holdout，不下载任何新模型/数据，不改宽度/架构。现有base model.py及run_v0.py与V0执行哈希一致；prepare/build_report/supervise历史源码若缺失，只报告未知，不重建猜测版本。

仅选原train中按原顺序前3068个完整块，seq128、每完整块127有效next-token目标，N=389636；有效batch4序列，恰767步。预期舍弃3068、3069两完整尾块（不能组成有效batch4），以及3070号不完整尾块（122个有效输入token），准备程序将按实际mask核实并保存理由。每条轨迹训练只遍历这些块一次，无循环补齐；不在train做额外模型评估或更新热身。每seed以独立Torch CPU Generator对同一集合randperm；保存具体顺序和SHA。同seed六条件逐块顺序一致。

对选定训练集合做一次CPU标签统计，使用base next_token_labels：块末忽略，source或target为pad均忽略，计入有效EOS。n_v仅由有效损失目标组成，pi=(n_v+0.1)/(389636+0.1*50304)，alpha=.1、强度1，无调参。pi全程固定；先验第0步已经利用整集合频率，单独记录统计遍历389636目标及成本，不按在线更新token声称严格数据效率优势。完整dev所有493块/62601有效目标，仅评估模型/基线，不用于训练频率、CPU拟合、选配置或选checkpoint。现有packed数据没有可靠逐目标文档映射，禁止伪造文档重采样。

## 六条件与配对

输入都是同一连续PCA128 C_in，FP32存储且冻结。输出连续初值C_out相同；sign=where(C_out>=0,+1,-1)/sqrt(128)，同一既存变换与既定RMS规则。

| 条件 | 输出表 | 独立bias buffer |
|---|---|---|
| L0 | 连续可训 | 固定0 |
| F0 | 连续冻结 | 固定0 |
| S0 | sign冻结 | 固定0 |
| Lp | 连续可训 | 固定log(pi) |
| Fp | 连续冻结 | 固定同一log(pi) |
| Sp | sign冻结 | 固定同一log(pi) |

bias均为独立固定FP32 buffer；六条件都调用相同 `base_logits.float()+output_bias` 路径，不为某臂增加专用可训参数。只有L0/Lp的输出表允许更新；两投影、主干、position、final_LN均可训。S0/Sp只sign输出，不能冒充V0两端sign的C。

每seed（1729、2718）重新构造随机公共状态并保存文件，从同一文件加载六臂；不用V0最终状态，也不顺序续训；不同条件的可更新表不共享storage。跨seed改变初始化和排列，集合相同。原Student宽128、2层、4头、FF512、dropout0、position128，AdamW lr3e-4常定、weight_decay=.01、clip1、microbatch2×accum2、bf16 autocast/FP32参数。新增GPU预检仅短序列forward/backward，不optimizer更新；不持续测速或外推长训。

顺序固定为1729：[L0,F0,S0,Lp,Fp,Sp]；2718：[Sp,Fp,Lp,S0,F0,L0]。此反序仅减轻运行次序混杂，不赋予两个seed普适统计含义。评估点固定step0、100、767，完整dev；767是唯一主要筛选终点，不挑最好dev checkpoint。

## CPU无上下文诊断

用float64，从128维零向量a开始，对连续初始表和sign初始表分别最小化 `L(a)=logsumexp(C_out a)−pi^T(C_out a)`，只拟合训练pi。固定Torch LBFGS、strong_wolfe、max_iter500、max_eval700、history50、tolerance_grad1e-8、tolerance_change1e-12；验收收敛证据为有限梯度inf范数≤1e-6。未达到时只称“当前优化残差”，不得称最优。保存每次closure、实际iter/eval、停止原因、目标值、H(pi)、L−H、梯度inf/l2、耗时、完整dev NLL。稳定logsumexp，手写C^T(softmax−pi)对照autograd；验证a=0且bias=log(pi)时softmax=pi，atol1e-12。若作秩诊断，仅V×128或128×128，永不V×V。两个无上下文拟合不是上下文LM NLL下界。

## 数值、测量和预算

沿用未修改的base7项检查，仅新增输出可训性/固定输入和bias及同bias L/F初态相等的必要检查。实际全词表还核对同seed公共hash、同bias初态前向；训练中loss、grad、params有限，最终输入/bias和冻结输出哈希不变、L输出改变、每个公共参数张量更新。保存每step NLL分子/分母、累计目标和CUDA同步耗时；评估同样保存分子/分母，最终状态文件及SHA留本地，不打包权重。

新增GPU完整子进程累计限3600秒（导入、初始化、预检、全部12轨迹、评估、保存、失败和重试均计）；独立标准库supervisor启动前检查余额、事前估计600秒，超过剩余不启动；外部超时保留60秒清理。一次只一个GPU进程，预计或实际不足则BUDGET_INCOMPLETE，不改臂预算取有利子集。V0的19.406秒单列，项目累计=19.406+本轮新增。GPU allocated/reserved、nvidia-smi设备全局占用、CPU采样RSS不同口径分别报告；sign仍为FP32普通矩阵乘。

CPU数值阶段完整进程合计1800秒，另记封存前必要检查成本；外部CPU监督器和拟合内部deadline保留已有/未收敛结果。不会为了收敛调参或延时。本轮无需用满任一预算。

## 指标、解释和一次性决策

每seed、每预定点计算：freeze0=F0−L0，freezep=Fp−Lp，sign0=S0−F0，signp=Sp−Fp；interaction_freeze=freezep−freeze0，interaction_sign=signp−sign0。全部保留正负号。context_gain=NLL(训练pi→完整dev)−NLL(模型→完整dev)，仅表示超出固定unigram的预测收益，不等于语义能力。

最终点分别按freeze和sign分类，tau=.05 nats/token（约exp(.05)=1.0513 PPL比值，探索预算容忍值，不是文献标准/显著性/等价/发表门槛）：

1. PRIOR_SENSITIVE_IN_THIS_SCREEN：两个无先验差同向且各abs>tau；两有先验差各abs≤tau且各abs≤原差abs的25%。
2. RESIDUAL_TO_EXPLAIN：两有先验差同向且各abs>tau。
3. NO_MATERIAL_GAP_IN_THIS_SCREEN：两seed在有/无先验下各差abs均≤tau。
4. 其余INCONCLUSIVE，不追加seed、改阈值/终点/数据救结果。

独立实验单位是两个训练seed；token、块、文档不是独立训练重复，不作等价检验或普适置信主张。若模型没有在两个seed稳定优于pi，明确尚不足讨论成熟上下文机制，不扩大训练。先验同时加入频率信息并平移logits到仿射空间；敏感性不唯一识别因果机制。固定输出列空间约束、final_LN可经其bias学习列空间内偏置、正pi可表达条件 `log(pi)∈col(C_out)+span(1)`、hidden128 rank界、argmax/唯一性/概率表达三者区别按原指令严格说明，不列为新方法/定理。

只有至少一类RESIDUAL且定向文献没有完整覆盖，报告才附一页后续假设设计；此页不授权训练。若已知干预足以消除信号，暂停当前差距作为独立论文贡献；其他情况明确唯一关键缺口。完成报告、metrics/diagnostics、资源/来源及无语料/权重/环境的审查ZIP后停止，不进V1。
