# V0 冻结契约（在 GPU 短跑前写定）

日期：2026-10-21，seed=1729。唯一 donor 为 Apache-2.0 的 EleutherAI/pythia-70m，最终 step143000，commit `a39f36b100fe8a5377810d56c3f4789b9c53ac42`。配置标明 untied；准备程序另检查真实参数对象、形状和 storage，禁止把两端绑在一起。学生不加载 donor 主干。

输入与输出各自取 donor 的 50304×512 表 E。仅用来源表，以 CPU float64 计算 mean μ、512×512 中心协方差并精确 eigh，取最大的128个特征向量 Q（512×128）。每列最大绝对值元素定为正，保存 μ、Q、全局缩放和特征值。连续表 Z=(E−μ)Q·s，s 使全表 RMS=1/√128；符号表 sign(Z)/√128，零约定正号。两端独立 PCA，不跨端混用。禁止使用 dev、holdout 或学生训练统计拟合变换。没有 V×V 相似度矩阵。

前向为 H0=Zin[token] Win + learned_position；H=final_LN(two pre-LN causal Transformer blocks(H0))；logits=(H Wout) Zoutᵀ。Win、Wout 均128×128，无偏置；学生宽128、2层、4头、FF512、dropout0。不存在额外 unigram/logit bias。Transformer 内部 Linear 和 LayerNorm 的普通偏置允许训练，不等同输出频率先验。

A：Zin/Zout 均可训练；B：与 A **完全相同数值**的两表冻结；C：与 B 同一对来源表及已保存 PCA 变换产生的 sign 两表冻结。两表都以 float32 常驻 GPU，C 没有 bit-pack，没有二进制 CUDA 算子；bf16 autocast 下执行普通矩阵乘法。三臂的公共主干、position、两投影加载同一份 common_init.pt，并核对按 tensor 内容的 SHA256；共同状态不依赖仅重设 seed。

数据：固定 revision `b08601e04326c79dfdd32d625aee71d232d685c3` 的 WikiText-2 raw 唯一 train parquet（约6.36 MB，CC-BY-SA-3.0/GFDL）。按一级标题恢复完整文章，规范化空白做整篇精确去重，先选前128篇，再以1729置换分96 train/16 dev/16 holdout，**先划分再分词**。保留集仅保存原文与哈希，不分词、不评估。dev 不用于选择配置。本文档级去重不代表 donor 去污染，重叠状态 UNKNOWN。

每篇追加 EOS=0，split 内按固定文档顺序连接打包，允许 EOS 后跨文档注意力；块长度128，块末标签忽略，因此完整块127个有效预测。右侧 padding 借用 EOS id 但 attention_mask=0，source/target 任一无效则 label=-100。无额外 BOS。保留50304输出类别（tokenizer未使用的末尾行也保留），不重新编号词表。

CPU 数值检查先通过，然后 GPU 短序列32、batch1的三臂预检（forward/backward，不做 optimizer step），随后共同短跑。每臂100 optimizer steps、microbatch2、梯度累积2、有效4序列，前400完整块共50800有效训练预测；三臂逐块使用完全相同前缀。AdamW lr=3e-4恒定、weight_decay=.01、clip=1、FP32参数与bf16 autocast。每臂3次仅forward/backward热身，不更新权重。稳定吞吐统计 step11–100，每step CUDA同步。

train online NLL按整个已训练前缀有效token加权；另在相同前32 train块（训练内诊断）及前32 dev块报告初/末态 NLL，实际有效token数从mask计算。保存原始逐step JSONL、最终状态与代码哈希。若中断，任何比较只使用共同完成的前缀，不能将不同步末态直接作质量比较。

累计 GPU 作业从 CUDA runner 进程的 job_start 到 job_end 记账，包含初始化、预检、热身、评估、保存和失败。仅一个 runner、串行三臂；初始预估600秒，不足则不启动；总限7200秒，每步更新账本并设置预留，外部 supervisor 进程于累计7100秒终止 runner 进程树，留100秒清理；runner 内线程 watchdog 为附加防线。禁止自动进入 V1。

解释范围：A/B只隔离本128维接口中的冻结，B/C只隔离预定norm规则下的sign。A不是原维可训表；三臂不是种子完整复现，单seed短跑不能证明有效、等价或新颖。当前文献重合高，新颖性保持UNRESOLVED；没有证据触发已完整回答同一问题的 STOP_OVERLAP。

执行后来源校验补注（2026-10-21）：固定WikiText来源卡YAML写CC-BY-SA-3.0/GFDL，而Licensing Information正文写CC-BY-SA-4.0；精确版本为SOURCE_INTERNAL_CONFLICT，原始卡按哈希保留，未独立消除歧义。最终预算采用外部supervisor实测19.406秒，runner内部CUDA阶段11.211秒另存。此补注不作为训练前已知事实。
