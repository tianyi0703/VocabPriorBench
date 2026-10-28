# V0.2 复用 V0.1 输入证据审计

状态：**VALID_FILE_EVIDENCE_PENDING_CPU_TENSOR_CHECKS**。八个必要输入的实际bytes/SHA256均与V0.1 manifest匹配，Fp/Sp初始哈希与日志互相印证，未见阻止CPU准备的来源错误。**本审计没有加载张量**；公共状态state_sha256、码表dtype/shape及C/S实际tensor哈希重算留给统一900秒预算下的CPU准备，不能把这里的文件校验写成已完成张量验证。

## 必要输入白名单

仅对下列八个输入流式读取并重算SHA256。manifest键只规范化反斜杠，不修改原文件；完整哈希和预期/实际bytes见同名JSON。没有对历史manifest中的其他输入进行泛化遍历。

| 实际输入 | bytes | 对照 manifest |
|---|---:|---|
| `artifacts/codes.pt` | 51,512,981 | PASS |
| `data/packed.pt` | 4,107,859 | PASS |
| `runs/v01_20261022/private/prepared.pt` | 1,209,450 | PASS |
| `runs/v01_20261022/private/common_init_1729.pt` | 1,788,143 | PASS |
| `runs/v01_20261022/private/common_init_2718.pt` | 1,788,143 | PASS |
| `runs/v01_20261022/block_order_1729.json` | 26,505 | PASS |
| `runs/v01_20261022/block_order_2718.json` | 26,505 | PASS |
| `reports/v01/data_selection.json` | 34,638 | PASS |

两份order都恰为选定3068块的完整无重复排列，文件SHA与data_selection一致；选定集合、389636训练目标、完整dev493块/62601目标的metadata一致。两份**V0.1**公共初态文件哈希与seed记录一致，记录的state hash跨seed不同；未使用V0 artifacts/common_init.pt或重新随机生成初态。必须由后续CPU阶段重算state内容hash才可完成输入验证。

## C/S 桥接初始锚点及原始旧 NLL

`anchor_hashes.json`包含四条 `anchors`，以及方便读取的 `by_seed_condition['1729_C'/'1729_S'/'2718_C'/'2718_S']`。每条提供expected_input/output/bias/common/order_sha256。全部来自对应Fp/Sp的**before_hashes**，并与trajectory_start、data_selection/块序SHA核对；没有读取或哈希任何final checkpoint。Fp/Sp输入、输出和bias在旧运行前后均未变。

| seed | V0.2 / V0.1 | step0 原始 NLL | step100 原始 NLL | step767 原始 NLL |
|---|---|---:|---:|---:|
| 1729 | C / Fp | 7.710583739645483 | 6.852926845998117 | 6.274890513258699 |
| 1729 | S / Sp | 7.751682039636448 | 7.0034112247654665 | 6.443508820995271 |
| 2718 | C / Fp | 7.684166416368024 | 6.865855782914186 | 6.269336835657952 |
| 2718 | S / Sp | 7.711838385965155 | 7.014444109490937 | 6.443635412322125 |

上述值从原metrics_raw.json读取，未使用报告舍入值；还保留对应nll_sum/62601分母。C初始output tensor hash为`dee18b2b89acdd06de3c5fb3df2f50db6ea55d6b6866aff0cbb9487d1ec53215`，S为`e9a4e872117f71b450aeb399e1c82926a730a17987c6004d9bbee7a79fea8f15`；全四条input和log(pi) bias锚点分别相同。这里是旧日志中的tensor hash，实际复用表需新CPU检查重新算并匹配。

## 源码与解释边界

已只读审查指定V0.1报告、契约、配置、环境、Fp/Sp资源及实际snapshot的Student/OutputStudent、prepare、run_gpu、common、supervisor。未导入旧common（其import会创建旧目录），未运行旧prepare、任何GPU或旧audit脚本，未读取holdout/原始语料。

- OutputStudent的input始终冻结，bias是clone的固定FP32 buffer，计算为base浮点logits之后再加bias。V0.2应逐字节复用模型源码，显式四表映射、全部输出冻结；旧condition[0]/endswith('p')分派不能用于新名称。
- 旧轨迹monotonic起点早于模型构造，但start_utc晚到active字典处记录；需补充勘误，不能为此重跑V0.1。旧外部312.71703610004624秒完整进程预算独立于此问题。
- 原报告的“训练pi加权投影margin”和候选稿的未加权最大余弦近邻margin不是同一量。候选稿不在旧snapshot清单，不能由封存来源核实其自述的结果盲态。V0.2不执行两种margin。

V0.1 C/S对应设置的共同输入、prior、初态和顺序是桥接依据；相同块预算与初始化不代表优化收敛或语义机制已识别。本次仅审查必要复用证据，不新增文献或后续训练设计。
