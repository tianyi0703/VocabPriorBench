# V0 直接近邻核查（2026-10-21）

本文件记录实际读取的原始来源；种子论文及 2610.04002 由另一份核查覆盖。下面的“未见”只针对所读版本，不能当作领域不存在某项工作的证明。来源访问失败、网页存在、仓库 README 存在均不能证明新颖性或完整可复现性。

| 论文和已核验版本 | 原文已做的比较 | 精确章节或表号 | 本项目可能多回答的问题（尚属候选） | 重复风险 | 尚未核实项 |
|---|---|---|---|---|---|
| [An Empirical Comparison of Vocabulary Expansion and Initialization Approaches for Language Models](https://arxiv.org/html/2407.05841v1)，2407.05841 **v1，2024-07-08** | RoBERTa/LLaMA2 扩词表；CW2V、OFA、单/多变量 Gaussian、mean、random；初始化后及 CPT 多检查点评估。理论从 LM head 出发，未绑定时也分别初始化输入表。 | §3.1–3.4；§4.4–4.5；§5.1–5.3；Table 2；Figures 2–4；Appendix A/G | 保持原 tokenizer 和随机学生主干，隔离同 donor 低维码的可训/冻结及 sign 化；用训练词频和码区分统计预测数据预算的收益边界。 | **中高**：初始化先验、简单基线追上复杂方法、随训练预算变化均已有实验；不能把“初始化重要/优势随训练变化”称新。其任务是扩词表后的已有主干适配，非所拟三臂。 | 本轮不运行作者代码；CW2V GitHub 分支通过浏览工具返回 restricted URL。正式 CoNLL 版本另存在，本表精确定位以指定 v1 为准。 |
| [Beyond Initialization Loss: A Systematic Study of Token Embedding Initialization Strategies for LLM Vocabulary Extension](https://arxiv.org/html/2608.03494v1)，2608.03494 **v1，2026-08-04** | Nemotron-3-Nano-30B-A3B 的 Hindi 扩表；20 余初始化、norm 校准、两端非对称；初始 loss/BPB、50 步 CPT 和更长训练。明示两表 untied，初始化后全参数共同训练。 | §4（Decoupling the Matrices）、§4.1–4.2；§5 setup；Table 1；§5.4/Figure 2；§5.5/Table 2；§5.6/Table 3；Limitations | 隔离冻结/符号化，不把输入先验和输出先验的不同作用混作“冻结收益”；测试可跨来源预测的边界。 | **高（控制与诊断层面）**：双表区别、norm、step-zero 排序不可靠、早期 CPT probe 已明确做过。作者已声明单语言/架构及长训差异缩小，不能重新包装为新发现。 | 未验证发布代码、训练日志或权重齐全；本轮仅核实论文全文，未复现其大规模 CPT。 |
| [Language Models Without a Trainable Input Embedding Table: Learning from Fixed Minimal Binary Token Codes](https://arxiv.org/html/2605.09751v2)，2605.09751 **v2，2026-10-02**（v1：2026-05-10） | learned input vs fixed canonical/affine Binary-16；固定 tile lift；保留标准 untied 可训输出头。三训练 seed，约 16–17B tokens；Table 2 的均值差异明确仅描述性。 | §3.1–3.4；§4.1/Table 1；§5.1/Table 2；§5.2/Table 3；§6 Controls needed for mechanism；§7；Appendix G | 由同 donor 连续坐标构建 A/B/C，排除维度、输入/输出来源和可训性混杂；研究有预训练几何的符号码，而非纯 token-ID 编码。 | **高**：“没有可训输入表也能学习”“二进制固定码”“参数减少”均已做。作者 §6 已主动提出 learned rank-16、fixed random continuous 等缺失对照，不能把这些限制称新发现。 | 未下载或运行该论文的三个 HF checkpoint；卡片/权重、三 seed 原始日志和完整训练数据顺序未独立审计。 |
| [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/html/2304.01373v2)，2304.01373 **v2，2023-05-31**（v1：2023-04-03） | 同序数据、不同规模/去重设置及训练检查点；不是冻结/压缩对照论文。70M 为 6 层、512 宽、8 heads。 | Table 1；§2.3 第 5 项明确 untied embedding/unembedding；§2.4；Appendix B 的版本更正 | 作为可审计 donor；实际参数对象必须再次检查，保留两张表分别构码。 | **基线来源，无直接方法创新**：公开多检查点及数据顺序本就是其贡献。 | 论文不等于当前下载快照。实际 revision、文件 hash、表形状与对象共享关系应由本项目下载/数值日志提供。donor 对任一后续评测文本的污染不能由学生去重排除。 |
| [All Word Embeddings from One Embedding](https://arxiv.org/html/2004.12073v3)（ALONE），2004.12073 **v3，2020-10-23**，NeurIPS 2020 | 固定随机 binary mask vs real filters，经共享 base vector/FFN 得嵌入；GloVe 重建、MT、摘要；含 factorized 基线、base vector 可训/冻结；MT 与 pre-softmax 共享。 | §2.1–2.3/Table 1；§3.1/Figure 3；§3.2/Table 2；§3.3/Table 3 与 Figure 4 | 同 donor PCA→sign 的几何及码碰撞如何与词频/预算共同决定冻结代价；两端 untied 时的分别控制。 | **高**：固定随机码、binary/real、低维/因子化强基线和收敛速度权衡都有先例。它的共享 base 可训性对照不等同于逐 token 连续码 A/B，但不能泛称“固定 vs 可训第一次比较”。 | [官方代码](https://github.com/takase/alone_seq2seq) 可读，README 明示主分支结果可能与论文有差异，需用 released version；本轮未锁历史 tag、未运行。 |
| [Compressing Word Embeddings via Deep Compositional Code Learning](https://arxiv.org/html/1711.01068v2)，1711.01068 **v2，2017-11-17**，ICLR 2018 | 从 GloVe 或先训 NMT 表学组合离散码与 codebook；比较压缩率/重建损失/任务质量。IMDB 两者嵌入冻结；NMT 压缩嵌入冻结、其余参数从随机初始重训。 | §3–4；§5.1–5.3/Tables 2–4；尤其 §5.3 retraining 段；§6.1/Table 5；§6.2/Figure 3 | 在同一低维线性接口区分冻结和符号化；将训练词频、区分能力、预算形成先注册的预测并在保留来源验证。 | **很高（研究叙事）**：“预训练嵌入压缩后冻结＋随机学生”已有直接先例；码本利用率、语义相近码和压缩强度也已分析。仅换 LM/PCA/sign 或加图不足以建立实质区别。 | [作者代码](https://github.com/zomux/neuralcompressor) 可读，含 quantize/train/export 示例；未检验完整 MT/分类端到端实验脚本、权重或原始日志齐备。 |

## 与 V0 契约的关系

这次核查**未找到上述近邻完整实现并回答以下联合问题的证据**：同 donor、分别保留输入/输出来源、同连续低维接口、A/B 初值相同、单独隔离冻结与 sign 化，再以词频和码区分能力预测数据预算的交互，并在未参与规则选择的语料或 donor 上验证。这是有限检索下的候选差异，不是新颖性证明；三臂单 seed 短跑只验证实现，不能完成这个问题。

对照含义须写窄：A/B 只隔离本接口内的可训性，B/C 只隔离预先固定尺度规则下的符号化。A 不是全维可训表；若同 donor 两张表 untied，不能改绑一表后声称输入输出先验相同。V1 必须考虑原维可训/冻结、随机及训练前语义对应置换，并明确全维先验损失、秩约束、离散化和优化正则化各自的解释。

**本子集判定：UNRESOLVED，未触发 STOP_OVERLAP。** 上述工作已覆盖许多拟用机制和方法部件；仅“固定词表能运行”“压缩更小”或相同三臂上的短期 loss 排序都不足以支持新贡献。是否有 DISTINCT_CANDIDATE 必须结合种子及 2610.04002 全文后统一判断。

## Pythia 官方资源的实际访问范围

- [官方 GitHub README](https://github.com/EleutherAI/pythia) 可读：Quickstart 给出按 revision 加载的示例，并说明 `step143000` 对应 main 的最终检查点；实际执行仍应把分支解析为不可变 commit。
- [官方 70M 配置原文](https://raw.githubusercontent.com/EleutherAI/pythia/main/models/70M/pythia-70m.yml) 可读：`num-layers=6`、`hidden-size=512`、`num-attention-heads=8`、`no-weight-tying=true`。目录页被浏览工具拒绝后改读这个明确文件；没有把目录读取失败解释为文件缺失。
- README 的 License 段将 repo 代码及 Pythia models 等 artifacts 标为 Apache-2.0；这不能替代本轮所下载模型卡及 checkpoint 的具体 revision 核验。

## 访问审计和边界

2026-10-21 通过浏览工具实际读取了表内六篇 arXiv HTML 正文、相关实验及局限段；2605.09751 和 2304.01373 的未版本 abs 地址分别解析到 v2。补充 ALONE 时先读 v2，核验 abs 历史后改以 v3 定位；没有把两个版本混为一个表号来源。

原始 OpenReview PDF 访问遇到浏览器校验，NeurIPS 官方 PDF 返回不可访问；已成功读取相同题名作者的 arXiv 明确版本全文补足，不依赖二手解读。CW2V GitHub 分支和 Pythia models 目录页为工具 restricted URL，已如实保留，不能据此推断仓库不存在。未执行任何此类论文的训练/推理，也未下载额外模型。
