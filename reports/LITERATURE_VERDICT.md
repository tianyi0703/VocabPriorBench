# V0 文献总判定（2026-10-21）

`LITERATURE = UNRESOLVED`  
`STOP_OVERLAP = NOT_TRIGGERED`

本轮定向核查了下表 8 篇明确版本的全文，覆盖用户指定起点及两个直接先例；不是穷尽性系统综述。逐项证据、访问失败与版本记录见 [种子核查](literature_seed.md) 和 [近邻核查](literature_neighbors.md)。Bochkov 的两篇属于同一作者的连续研究，不能据篇数当成两次独立复现。

已有工作覆盖固定二进制输入、预训练表压缩后冻结、随机学生重训、词频分析、数据预算变化、多语言和初始化尺度控制。因此，“固定词表能运行”“继承 donor 知识”“换语言”“参数更少”均不足以构成贡献。目前没有核实到某篇已完整回答用户的联合问题，故不触发 `STOP_OVERLAP`；但有限范围内未发现完整重叠，也不能证明新颖性。统一建议仍为 `RESOLVE_NOVELTY`，V0 短跑只用于实现和资源核查。

## 统一差异表

| 论文和版本（原始全文） | 已做比较 | 精确位置 | 本项目候选差别 | 重复风险 | 未核实项 |
|---|---|---|---|---|---|
| [Embeddings Are Overpriced?](https://arxiv.org/html/2610.22428v1)，2610.22428v1，2026-10-15 | 原维可训表、donor 二值冻结表、三值主干；词频、训练曲线、多语言和扩词 | §2–6；表2–6；图1 | 同低维接口分别隔离冻结/符号化，再检验保留来源上的预测 | 高：核心叙事及词频/预算解释已存在 | 同接口连续码对照、跨 donor 预测未在所读版本找到；未端到端复现 |
| [An Empirical Comparison of Vocabulary Expansion and Initialization Approaches for Language Models](https://arxiv.org/html/2407.05841v1)，2407.05841v1，2024-07-08 | RoBERTa/LLaMA2 扩词；CW2V/OFA/分布/均值/随机初始化；多 CPT 检查点 | §3.1–3.4、4.4–4.5、5.1–5.3；表2；图2–4；附录A/G | 保持原 tokenizer，随机学生；控制冻结与离散化 | 中高：初始化与训练预算交互已有实证 | 作者代码未运行；CW2V 分支访问受限；表号限指定 v1 |
| [Beyond Initialization Loss](https://arxiv.org/html/2608.03494v1)，2608.03494v1，2026-08-04 | Hindi 扩表；20余初始化、norm 校准、untied 两端非对称；50步和较长 CPT | §4–4.2；表1；§5.4/图2；§5.5/表2；§5.6/表3；Limitations | 固定两端来源及尺度后检验冻结/符号化的可预测代价 | 高：输入/输出区别和 step-zero 排序失效已明确 | 代码、原始日志、权重齐备性与大规模 CPT 未复现 |
| [Language Models Without a Trainable Input Embedding Table](https://arxiv.org/html/2605.09751v2)，2605.09751v2，2026-10-02（v1：2026-05-10） | learned 输入、canonical/affine Binary16；固定 tile；保留可训 untied 输出；3 seed | §3.1–3.4；§4.1/表1；§5.1/表2；§5.2/表3；§6–7；附录G | donor 几何下同接口 A/B/C，并控制两端来源 | 高：二进制固定码及低维/随机连续控制已有先例或明确提议 | checkpoint、三 seed 原始日志及精确数据顺序未审计 |
| [Do Language Models Need a Trainable Input Embedding Table?](https://arxiv.org/html/2610.04002v1)，2610.04002v1，2026-10-02 | 1.7B级 learned 输入、Binary16、一次 GF2；可训 untied 输出；目标100B tokens | §3–5；表1–2；§7–8；附录B/E | donor 语义码及两端控制；预测词频×压缩×预算交互 | 高：固定输入可行性及 geometry/trainability 分解已有讨论 | GPU 型号、最终实际预算、完整复现未核实 |
| [Pythia](https://arxiv.org/html/2304.01373v2)，2304.01373v2，2023-05-31（v1：2023-04-03） | 不同模型规模/去重设置、同序数据与训练检查点 | 表1；§2.3第5项；§2.4；附录B | 用作可审计 donor；按实际 untied 两张表构码 | 方法重复风险低；不是冻结实验论文 | 下载快照及参数对象须由本项目实测；评测污染 UNKNOWN |
| [All Word Embeddings from One Embedding（ALONE）](https://arxiv.org/html/2004.12073v3)，2004.12073v3，2020-10-23 | 固定 binary/real filters；共享 base/FFN；重建、MT、摘要；base 可训/冻结、factorized 基线 | §2.1–2.3/表1；§3.1/图3；§3.2/表2；§3.3/表3和图4 | donor PCA/sign 的区分能力对冻结代价的预测 | 高：固定码、binary/real、因子化与收敛权衡已有比较 | 官方主分支注明与论文可能不同；历史 release 未锁定、未运行 |
| [Compressing Word Embeddings via Deep Compositional Code Learning](https://arxiv.org/html/1711.01068v2)，1711.01068v2，2017-11-17 | 从预训练表学组合离散码；压缩/重建/任务质量；冻结压缩嵌入，NMT 其余部分随机重训 | §3–4；§5.1–5.3/表2–4；§6.1/表5；§6.2/图3 | 同低维线性接口分离冻结/符号化；保留来源验证解释规则 | 很高：预训练压缩冻结＋随机学生、码利用率与压缩分析已有直接先例 | 全部任务脚本、权重、原始日志及端到端重现未审计 |

## 种子必核事实

种子 §2 使用 Gemma 3 270M 的 `262144×640` 表，中心化后 PCA512→sign；输入和输出共享代码，分别连接可训投影。B 加固定训练 unigram log 先验，C 从同先验起点继续训练。§3：随机学生主干，FineWeb-Edu 2.05B tokens，每臂一张 RTX5090、14–16小时。§5 已分析20种语言及未见 token；B 整体只在5种语言领先，修改先验后中文仍由 A 领先。§6 已披露单 seed、A 词表数据不足、donor 知识/许可、C 的主干/计划/先验三重变化、扩词质量代价及单 CPU/不同实现比较限制。这些均非本项目新发现。[种子全文](https://arxiv.org/html/2610.22428v1)。

## 官方资源核验的界限

种子 [HF 文件树](https://huggingface.co/QLNI/embeddings-are-overpriced/tree/main) 当前根提交 `af839128c84a5dee79017a50fe80ecbd0b21aacb`；已读训练源码和数据 checksums，已看到 A/B/C 权重、21训练 shard、validation、码表、先验及评估目录的文件元数据。没有逐个下载、复算或运行全部 checkpoint，不能写“完整可复现已验证”。其 [README](https://huggingface.co/QLNI/embeddings-are-overpriced)约17 RTX5090 GPU-hours/臂与论文14–16小时并列保留，不用于估算本机4060速度。

[种子训练代码](https://huggingface.co/QLNI/embeddings-are-overpriced/blob/main/code/train.py)将 packed bits 展开为 float32，再作浮点矩阵乘法；16MB码表不能解释为训练显存。代码还披露跨机器 PCA 重建近零位可能翻转，且 saved-table 相对路径与发布目录不同；这些只是静态发现，未复现。其[许可文件](https://huggingface.co/QLNI/embeddings-are-overpriced/blob/main/LICENSE.md)为非商业研究，衍生表/权重另受 Gemma 条款约束。

2610.04002 的 [learned](https://huggingface.co/Bochkov/ab_ext_learned/tree/main)、[Binary16](https://huggingface.co/Bochkov/ab_ext_binary16/tree/main)、[GF2](https://huggingface.co/Bochkov/ab_ext_gf2/tree/main) 均实际列出大权重、模型代码、tokenizer 及训练补充 zip；只核验目录/元数据，没有下载大权重、解压或训练。Pythia [官方项目](https://github.com/EleutherAI/pythia)和[70M配置](https://raw.githubusercontent.com/EleutherAI/pythia/main/models/70M/pythia-70m.yml)可读，配置明确不绑权；本项目仍须检查实际参数对象。ALONE [代码](https://github.com/takase/alone_seq2seq)及组合码[代码](https://github.com/zomux/neuralcompressor)可读，但未核验全部任务资源齐备。2608.03494 本轮只核实论文全文，不能补写代码已开源。

## 可保留的候选问题与停止规则

候选问题限定为：同 donor、两端来源及输出先验受控，同一低维接口中 A/B 初值相同，分别估计冻结与符号化；再以训练词频和码区分统计预先形成解释规则，预测未参与规则选择的语料或 donor 上的结果。现有证据不足以说明这一联合预测已做完，也不足以说明它必然是实质贡献。单独增加低维/随机连续/尺度控制尤其不算新颖：Bochkov 已明确提出这些控制。

若后续原始文献已完成相同受控比较和保留验证，且不能指出实质差别，应记录 `STOP_OVERLAP` 并停止新增 GPU 研究比较。当前缺的是可审计的新颖性判断和正式判别设计，不是更多单 seed 短跑。V0 的 A 仅为低维可训码；本表不能替代 V1 中同 donor 原维可训/冻结、随机及训练前语义置换强基线。

方法工具说明：本次合并参考本机 literature-review 技能的来源、版本和覆盖边界检查。工具文献：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*，当前版本v2（2026-09-02）；[arXiv记录](https://arxiv.org/abs/2609.00065)。这是工具来源，不计入上表8篇研究全文。
