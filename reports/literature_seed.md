# 种子与直接扩展核查（2026-10-21）

本文件记录实际读取的原始网页与代码；未运行作者实验、未下载大权重或全量语料。读取了以下两篇全文的实验、结果与局限。资源“可用”在此仅指公开目录及指定文件内容/元数据可读，不表示已成功重现实验。

## 存在性与版本

| ID | 核验结果 | 原始来源 |
|---|---|---|
| S | Sai Kiran Bathula, *Embeddings Are Overpriced? A Frozen 16 MB Vocabulary for Small and Ternary Language Models*, arXiv:2610.22428v1；v1 submitted 2026-10-15 06:34:18 UTC，9 pages / 6 tables | [版本页](https://arxiv.org/abs/2610.22428v1), [全文](https://arxiv.org/html/2610.22428v1) |
| E | A. Bochkov, *Do Language Models Need a Trainable Input Embedding Table? Fixed Minimal Token Codes at 1.7B-Class Scale*, arXiv:2610.04002v1；v1 submitted 2026-10-02 20:06:37 UTC | [版本页](https://arxiv.org/abs/2610.04002v1), [全文](https://arxiv.org/html/2610.04002v1) |

## 已读证据：S

| 核查项 | 精确证据摘要 | 位置 |
|---|---|---|
| 输入来源 | Gemma 3 270M, 262144×640; centered PCA512 then sign; shared input/output codes with separate learned projections | §2 |
| 输出先验 | B: fixed training-unigram log prior; C: trainable prior initialized identically | §2–3 |
| 训练 | Random body; FineWeb-Edu 2.05B tokens; one RTX5090, 14–16h/arm | §3 |
| 已做比较 | Learned full table versus frozen binary; ternary arm also changes schedule/prior; frequency strata and training curves | Tables 2–4, Figure 1 |
| 多语言 | 20 languages; B better overall in five. Modified unseen-token priors improve selected languages; Chinese still favors A | §5, “Tokens the model never trained on” |
| 作者已披露局限 | One seed; insufficient data for A; donor knowledge/terms; three C confounds; single-CPU/runtime comparison; vocabulary expansion costs quality | §6 |

出处：[S 全文 §2–6](https://arxiv.org/html/2610.22428v1)。这些是原作者已报告的观察或局限，不能作为本项目的新发现。

## 紧凑差异表

| 论文和版本 | 原文已做的比较 | 精确章节/表号 | 本项目可能多回答的问题（候选） | 重复风险 | 尚未核实项 |
|---|---|---|---|---|---|
| S, v1 | Full learned versus binary frozen; frequency, learning curves, multilingual, vocabulary additions | §3–6; Tables 2–6; Fig.1 | Same-interface continuous learned/frozen/sign separation; held-out prediction across donors/budgets | HIGH: frequency/budget/donor dependence already explicit | No matched continuous-code arms or cross-donor predictive test found; full reproduction unverified |
| E, v1 | Learned input versus Binary16 and one GF2 recoding; full learned output retained | §3–5, Tables 1–2; §7–8; Appendix B | Donor-derived semantic codes and both interfaces; predictive frequency/compression/data interaction | HIGH for generic “fixed input works”; proposed geometry controls already discussed | Hardware model, realized final budgets, complete reproduction unverified |

## 已读证据：E

E uses V=49152 and 16 zero/one ID bits, tiled to width 2048 without a learned input projection. GF2 uses one invertible matrix (seed 12345). All arms retain a trainable, untied, bias-free output matrix; there is no unigram-bias term in the stated interface. Models are randomly initialized; SmolLM2 supplies tokenizer artifacts, not donor model weights. Two GPUs are specified, with a target 100B prediction tokens per model, but a GPU model is not specified in the read text. The learned-input model leads important evaluations; the evidence establishes viability, not parity. Limitations include one run, unmatched input scale/geometry/capacity, one tokenizer/data/scale, and no measured systems-efficiency advantage. Section 7 already proposes learned low-dimensional, frozen random continuous and norm-matched controls. [§3–8; Tables 1–2; Appendix B](https://arxiv.org/html/2610.04002v1).

## S 官方资源的实际可读性

[官方仓库根目录](https://huggingface.co/QLNI/embeddings-are-overpriced/tree/main)可读，当前根提交为 `af839128c84a5dee79017a50fe80ecbd0b21aacb`，目录显示 83.1 GB。没有整仓下载。

| 类别 | 实际核验 | 证据 URL |
|---|---|---|
| 实验代码 | `train.py` 内容可读；`prep_data.py`, `eval.py`, `freq_eval.py` 等有文件与大小 | [code/](https://huggingface.co/QLNI/embeddings-are-overpriced/tree/main/code), [train.py](https://huggingface.co/QLNI/embeddings-are-overpriced/blob/main/code/train.py) |
| 权重 | `final/armA`, `armB`, `armC` 目录存在；armB `model.safetensors` 显示 406 MB，并可读文件元数据；未下载/加载 | [final/](https://huggingface.co/QLNI/embeddings-are-overpriced/tree/main/final), [armB file](https://huggingface.co/QLNI/embeddings-are-overpriced/blob/main/final/armB/model.safetensors) |
| 数据 | 8.24 GB 目录，train_000…020 与 val.npy；checksums 内容可读；大文件仅元数据核验 | [data/](https://huggingface.co/QLNI/embeddings-are-overpriced/tree/main/data), [checksums](https://huggingface.co/QLNI/embeddings-are-overpriced/blob/main/data/SHA256SUMS.txt) |
| 码表/先验 | npz 18.1 MB，counts 2.1 MB，seal 1.05 MB；仅元数据 | [tables/](https://huggingface.co/QLNI/embeddings-are-overpriced/tree/main/tables) |
| 评估/日志 | `eval/`含频率、B2、控制token、CPU与多语言相关文件；`logs/`目录存在；未逐项复算 | [eval/](https://huggingface.co/QLNI/embeddings-are-overpriced/tree/main/eval), [logs/](https://huggingface.co/QLNI/embeddings-are-overpriced/tree/main/logs) |

模型卡声明所有中间 checkpoint 与 optimizer state 已发布；本轮没有逐个下载验证完整性。[README “What is where” / “Reproducing the training”](https://huggingface.co/QLNI/embeddings-are-overpriced)写约17 RTX5090 GPU-hours/arm；这与论文14–16小时存在口径差别，保留原值，不换算为4060速度。

代码阅读补充：`frozen_table`, `Model.hidden`, `Model.logits` 显示 packed bits 被 `np.unpackbits` 展开为 float32 buffer，训练 readout 使用浮点矩阵乘法。代码的 seal 是 add-one 平滑 `log((counts+1)/(sum(counts)+V))`。saved-table 路径寻址脚本同目录，而发布文件在 `tables/`；注释还说明跨机器重建约0.008%近零位可能翻转。模型卡声称重建相同比特，因此应固定发布表和哈希。代码定义 D 小模型及 random-table 函数，不能据此声称对应实验已经完成。[官方 train.py](https://huggingface.co/QLNI/embeddings-are-overpriced/blob/main/code/train.py)。这些是静态阅读发现，没有尝试修复或复现作者代码。

许可实际可读：非商业研究/教育，B/C与码表还承受 Gemma 条款，tokenized 数据另含 FineWeb-Edu ODC-By 1.0 条款。[LICENSE.md](https://huggingface.co/QLNI/embeddings-are-overpriced/blob/main/LICENSE.md)。本项目不复用其模型或数据。

## E 官方资源的实际可读性

Appendix E 指向以下三个 Hub 项目；model card 与文件树均已实际访问。首次直接打开树链接失败，随后从模型卡点击 Files 成功；失败不解释为资源不存在。

| 官方项目 | 目录实际列出 | 未完成验证 |
|---|---|---|
| [ab_ext_learned](https://huggingface.co/Bochkov/ab_ext_learned/tree/main) | model.safetensors 7.25GB; model/config Python; tokenizer; preprocessing_train_material.zip 46.5kB；head `858d3c0` | 未加载权重、解压补充材料或运行 |
| [ab_ext_binary16](https://huggingface.co/Bochkov/ab_ext_binary16/tree/main) | model.safetensors 6.85GB; 同类文件；head `f9544a1` | 同上 |
| [ab_ext_gf2](https://huggingface.co/Bochkov/ab_ext_gf2/tree/main) | model.safetensors 6.85GB; 同类文件；head `a2d4285` | 同上 |

Binary16 model card 提供权重 SHA-256 `c3e78f24f03dff36b6174985dc5189e8ce37a52685d022ade36fa68c3239801b` 与 tokenizer revision `effd688a12921b4cc83e3312b6feb579f70f9c71`；这是发布者声明，未由本机下载复算。[Verification and integrity / Model specification](https://huggingface.co/Bochkov/ab_ext_binary16)。没有看到可据以宣称全数据、全部训练日志与 optimizer state 齐备的逐文件验证证据。

## 对停止条件的判断

`STOP_OVERLAP = NOT_ESTABLISHED`。这两篇不足以证明用户的完整问题已被回答；同样不证明创新成立。候选差别是受控低维接口中分离可训练性/二值化，并将预先选定的解释规则用于未参与规则选择的来源或语料。V0 三臂短跑只能验证实现与资源，不能验证这种预测差别。

本子核查建议 `LITERATURE = UNRESOLVED`，待合并其余直接近邻后判断；不得把已有词频图、20语言结果、donor 外部知识或单seed局限重新包装成贡献。未发现论文未做某项实验，也只是本次全文阅读范围内的阴性结果，并非领域空白证明。
