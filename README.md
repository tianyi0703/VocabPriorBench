# VocabPriorBench

**2026-10-23：当前路线暂存，以“冻结／sign 差距”为核心的本次论文立项尝试已结束；没有自动续跑。**

```text
ROUTE = PAUSED
PAPER_PROPOSAL = CLOSED
PAPER_READINESS = NOT_ESTABLISHED
AUTOMATIC_CONTINUATION = NONE
```

An auditable exploratory study of frozen vocabulary tables, fixed unigram bias, and sign directions in small language models.

本项目保存已完成的工程验证与探索记录，新颖性、普适机制和独立论文贡献均未确立。当前决定以 [RESEARCH_STATUS.md](RESEARCH_STATUS.md) 为准；旧报告的 `NEXT`、候选假设及运行命令是历史记录，不表示继续 V0.3、V1、margin 回归或投稿写作。

## 阶段与结果

唯一 donor 为 Pythia-70m；分别从其不绑权的输入、输出表构造 PCA128 码表。学生主干与投影从公共随机初态加载，未加载 donor 主干；2 层、hidden/code128，保留全部 50,304 个输出类别。下列名称仅在各阶段内部解释，尤其 V0 的 C 与 V0.2 的 C 不同。

| 阶段 | 实际执行 | 能支持的结果 |
|---|---|---|
| [V0](reports/V0_REPORT.md) | 连续可训 A、同初值连续冻结 B、双端 sign 冻结 C；单 seed，每臂 100 步、50,800 有效目标 | 验证实现、冻结规则、数值与短跑资源。仅评价固定前缀，不以最终排序主张方法优劣。 |
| [V0.1](reports/v01/V01_REPORT.md) | 固定连续输入；输出可训／连续冻结／sign 冻结 × 零／共同固定 unigram bias；两 seed、12 条轨迹 | 冻结差距对共同先验干预敏感；sign 在共同先验下仍有残差。 |
| [V0.2](reports/v02/V02_REPORT.md) | 连续／sign 方向 × 原行范数／单位行范数；同一固定 prior，两 seed、8 条轨迹 | 两种范数匹配下 sign 方向差均超过预设阈值；恢复原 L2 行范数未消除本轮差距。 |

V0.1/V0.2 每条遍历同一 3,068 个完整训练块：767 步、389,636 个有效目标；完整 dev 为 493 块、62,601 个有效目标，主终点固定 step767。V0 前缀 NLL 不能与后续完整 dev NLL 直接比较。

V0.1 中 L/F/S 分别表示可训连续／冻结连续／冻结 sign 输出；正差值表示增加 dev NLL。两 seed 的主要终点如下（nats/token）：

| seed | freeze0：F0−L0 | freezep：Fp−Lp | sign0：S0−F0 | signp：Sp−Fp |
|---|---:|---:|---:|---:|
| 1729 | +0.357762 | −0.012770 | +0.489389 | +0.168618 |
| 2718 | +0.373957 | −0.010249 | +0.479790 | +0.174299 |

冻结分类为 **`PRIOR_SENSITIVE_IN_THIS_SCREEN`**，sign 分类为 **`RESIDUAL_TO_EXPLAIN`**。固定 bias 同时注入频率信息并平移 logits 仿射空间，不能唯一证明“词频解释完毕”；残差也不自动证明语义损失。

## V0.2 四格终点

令 C 为原连续输出表，`r_v=||C_v||₂`，U 为单位化连续行，S 为 `where(C>=0,1,−1)/sqrt(128)`，RS 为 `r*S`。四条件共享冻结连续输入和同一固定 FP32 `log(pi)`；输出表冻结，公共主干／投影的 444,928 个参数可训。

最终完整 dev NLL（nats/token）：

| seed | C | U | RS | S |
|---|---:|---:|---:|---:|
| 1729 | 6.274890513259 | 6.273991691341 | 6.454264401891 | 6.443508820995 |
| 2718 | 6.269336835658 | 6.268255440614 | 6.454211615124 | 6.443635412322 |

六个有符号差值：`G=S−C`，`D_r=RS−C`，`D_1=S−U`，`N_cont=U−C`，`N_sign=S−RS`，`I=D_1−D_r=N_sign−N_cont`。

| seed | G | D_r | D_1 | N_cont | N_sign | I |
|---|---:|---:|---:|---:|---:|---:|
| 1729 | +0.168618307737 | +0.179373888632 | +0.169517129654 | −0.000898821918 | −0.010755580895 | −0.009856758977 |
| 2718 | +0.174298576664 | +0.184874779466 | +0.175379971708 | −0.001081395044 | −0.010576202802 | −0.009494807758 |

按事前固定 `tau=0.05` 及 A→B→C→D 优先规则，两个 seed 的 `D_r`、`D_1` 均超过 tau，分类 **B：`DIRECTION_RESIDUAL_AT_BOTH_NORMS`**；B 未增加 G 门槛。这是工程筛选，不是显著性或等价检验。C↔V0.1 Fp、S↔V0.1 Sp 在 step0/100/767 的 12 个桥接点最大 NLL 差为 **0.0**。原始分子／分母与全精度值见 [metrics.csv](reports/v02/metrics.csv)、[contrasts.csv](reports/v02/contrasts.csv)、[screening.json](reports/v02/screening.json)、[bridge.csv](reports/v02/bridge.csv)。

共同训练 unigram 在完整 dev 上的 NLL 为 **7.565262378247685**；`context_gain=unigram NLL−模型 NLL`：

| seed | C gain | U gain | RS gain | S gain |
|---|---:|---:|---:|---:|
| 1729 | 1.290371864989 | 1.291270686907 | 1.110997976357 | 1.121753557252 |
| 2718 | 1.295925542590 | 1.297006937633 | 1.111050763124 | 1.121626965926 |

最终四条件均超过该固定 unigram；gain 仅表示预测收益，不是语义信息量。pi 预先统计完整选定训练集合的实际损失目标，包含一次训练统计遍历，不能只按在线更新 token 声称严格数据效率优势。

## 真实资源与解释边界

本机 Windows11、RTX4060 Laptop GPU（8188 MiB）、Python3.12.15、PyTorch2.10.0+cu128、CUDA build12.8；没有云 GPU 或付费训练／推理 API。

| 阶段 | 外部 supervisor 完整 GPU 子进程墙钟 |
|---|---:|
| V0 | 19.406 秒 |
| V0.1 | 312.71703610004624 秒 |
| V0.2 | 156.68108119996032 秒 |
| 累计 | **488.804117300007 秒，约 8.15 分钟** |

完整进程时间包含导入、初始化、预检、评价、保存、退出及 CPU 间隙，不能称为纯 GPU 计算时间。短窗口吞吐不代表持续热稳态；nvidia-smi 全设备采样含桌面／其他程序，不能等同于本进程 PyTorch allocator 峰值。CPU RSS 为采样近似。见 [V0.2 resource_profile.json](reports/v02/resource_profile.json)。

所有 sign 表实际均为 **FP32、普通浮点 matmul、`bit_packed=false`**；单表 25,755,648 bytes。没有实现或测得 1-bit GPU 存储、专用 kernel 或压缩加速。

- 原 L2 行范数恢复不足以解释本轮终点，不代表排除全部尺度方案；r 不是最小二乘最优二值重建尺度。逐行缩放通常改变输出列空间，同初态／同步数不能证明相同函数族或训练收敛。
- 单 donor、单小语料、单架构／宽度、两个训练 seed、有限步终点；token／块不是独立训练重复。未识别普适或因果机制，未确立独立论文贡献，也未推翻原论文。研究去留见 [RESEARCH_DECISION.md](reports/v02/RESEARCH_DECISION.md)。
- V0 保留 **`LIMITED_V0_HISTORY`**：prepare/build_report/supervise 的执行前与交付哈希不同，旧正文未找到，不能恢复精确 diff。19.406 秒来自当时工具 stdout 后归档，首次 supervisor 起止 UTC 为 unknown；不反推伪精确时间。
- **独立 holdout 未分词或用于模型评价。** V0 曾封存其原文与哈希；V0.1、V0.2 和本次发布不打开或重新哈希它，不能概括为整个项目从未读取其原文。
- [V01_ERRATA.md](reports/v02/V01_ERRATA.md) 记录候选 margin 的两种定义不同、候选未纳入原训练前快照且盲态不能独立核实、V0.1 逐轨迹 UTC 与 monotonic 起点不同位；不称候选已独立预注册。外部完整计时仍保留，V0.2 已修正同位记录。

## 公开副本与只读核验

本地发布副本为 `publication/github/`，该目录作为公开仓库根。公开内容为整理代码、配置、执行快照、结果及必要日志；原始报告／日志／ZIP 留在本地不改。公开副本仅删节 Windows 用户主目录中的账户名与 GPU UUID，原 SHA→公开 SHA、大小和删节类别记于 [PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json)。

公开派生审查包：[V01_PUBLIC_REVIEW_BUNDLE.zip](V01_PUBLIC_REVIEW_BUNDLE.zip)、[V02_PUBLIC_REVIEW_BUNDLE.zip](V02_PUBLIC_REVIEW_BUNDLE.zip)。**它们不是原始 ZIP 的原字节。** 公开 payload 由包内 `PUBLIC_BUNDLE_MANIFEST.json` 核验；改名保留的 `ORIGINAL_BUNDLE_MANIFEST.json` 与旧报告哈希描述原始来源，不能用于验证删节后的 payload。发布范围见 [PUBLICATION.md](PUBLICATION.md)；最早 README 保留为 [README_V0_HISTORY.md](README_V0_HISTORY.md)。

在公开仓库根目录，使用系统 Python 标准库执行：

```console
python scripts/publication/verify_release.py
```

这是公开文件／PUBLIC ZIP 校验和既有结果复算入口，**不是训练复现**：不下载数据、不加载 tensor、不启动 GPU 或续训；不需要安装科研依赖。精确检查项与失败路径以脚本输出为准。

公开仓库排除权重、码表／PCA／公共初态 tensor、packed 语料、private 输入、最终 checkpoint、环境与凭据。**fresh clone 缺少私有输入，不能恢复原状态或一键精确重跑。** 封存执行代码保留 Windows 项目硬路径（如 `F:\VocabPriorBench`），未作跨机器训练重构；旧 prepare/seal/supervisor 可能写文件或启动 CUDA，不是只读入口。公开 Git 历史及哈希不能补回缺失 tensor 或旧源码正文。

## 来源与许可记录

- donor：`EleutherAI/pythia-70m`、step143000、revision `a39f36b100fe8a5377810d56c3f4789b9c53ac42`；来源记录为 Apache-2.0，输入／输出实际不绑权。
- 数据：`Salesforce/wikitext` 的 WikiText-2 raw train、revision `b08601e04326c79dfdd32d625aee71d232d685c3`；选 128 篇，先按文档划分再分词。来源卡 YAML 为 CC-BY-SA-3.0/GFDL，正文为 CC-BY-SA-4.0，**来源内部许可版本冲突未解决**；不发布原始／分词语料。
- 文档精确去重不等于 donor 去污染；donor／评测重叠为 **UNKNOWN**。版本与来源哈希见 [V0 source_manifest.json](reports/source_manifest.json)、[V0.2 source_manifest.json](reports/v02/source_manifest.json)。

本次发布未新增或选择项目开源 LICENSE，不替第三方材料统一授予许可。文献核查保留在历史报告中，本 README 不增加文献检索或创新主张。
