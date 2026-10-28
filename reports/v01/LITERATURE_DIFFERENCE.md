# V0.1 定向文献差异与数学边界

核查日：2026-10-22（Asia/Shanghai）。以下为四篇指定原始全文及种子 §2/§6 的定向阅读；未扩展检索、读取语料/holdout、下载模型或运行 GPU。`FULL_OVERLAP = NOT_ESTABLISHED`：未核实到下述剩余诊断已被完整覆盖，相关性不等于完整覆盖，也不等于本项目有新颖性。V0.1 可按已限定契约完成机制筛选，不以泛泛的新颖性待决替代实际诊断。

## 紧凑差异表

| 原始来源与精确位置 | 已有结果 | 本轮限定诊断 | 不可主张的创新 | 仍需验证 |
|---|---|---|---|---|
| [A Natural Bias，ACL 2023](https://aclanthology.org/2023.acl-short.22/)；[PDF](https://aclanthology.org/2023.acl-short.22.pdf) §2.2式3–4、§4.1、§4.2图2–4、附录A.6 | 训练 unigram 初始化输出 bias；**bias 训练中可更新**（§4.1，p246），NMT 的学习效率/质量与频率分析 | 六条件共同使用固定 log(pi)，看输出可训/冻结/sign 的差距交互 | unigram bias、context×frequency 分解、早训加速均非新 | 固定输入、同低维两端来源下的冻结/sign×共同固定先验交互；该文未直接检验 |
| [Breaking the Softmax Bottleneck，1711.03953v4，ICLR 2018](https://arxiv.org/html/1711.03953v4) §2.1命题1/推论1、§2.3–2.5、§3.1–3.3表1–3/6–7 | log条件概率矩阵按行平移后的秩条件；MoS 与 softmax/MoC 比较 | 区分固定列空间和共同128维瓶颈 | 线性输出秩限制、加宽/换映射不必解除限制均非新定理 | 本轮固定表、先验与优化残差的数值量；理论不直接给本机差值 |
| [Small LM Saturation，2404.07647v1](https://arxiv.org/html/2404.07647v1) §3–4图1–5、§5.1图6、§5.2定理5.3、附录A.2/B.1 | Pythia 饱和/谱分析；冻结预训练主干、约150M tokens 重训不同秩的 head；理论界较松 | 固定码宽/架构，仅测输出冻结/sign 对先验干预的敏感性 | 小模型输出维度/谱可能限制质量非新；不能把谱相关性当因果 | 本轮短轨迹未检验长期饱和；常量拟合不是该文 head 重训 |
| [Multi-mode Word Distributions，ACL 2022](https://aclanthology.org/2022.acl-long.554/)；[PDF](https://aclanthology.org/2022.acl-long.554.pdf) §2.2定理1–2、§3–5表1–4、附录D/E | 在明确线性依赖/几何等假设下限制部分候选排序；MoS/MFS、GPT-2/BERT和歧义模板 | 分开检查码唯一性、argmax 可达性与分布拟合 | 固定输出几何会限制分布、单 hidden 不能任意排序非新 | 不能从128维或近邻直接推出本项目某 token 永远不可预测 |
| [种子2610.22428v1](https://arxiv.org/html/2610.22428v1) §2、§6 | Gemma 表 PCA512→sign、双端共享；B固定训练 unigram，C可训先验；B2先验可训短程消融；作者披露C主干/计划/先验混杂 | 连续输入固定，L/F/S输出×0/pi全交叉，两 seed 终点评价 | 固定先验、donor知识、原文已披露混杂均非新发现 | 未在所读版本找到同接口六条件交互；本轮也不能复现或推翻种子 |

## 数学边界（本轮接口的直接推导，不作为新定理）

列向量记号：`C=C_out∈R^(V×k)`，`P∈R^(k×d)`，`q(x)=P h(x)`，`ell(x)=C q(x)+b`；本轮 `k=d=128`。冻结 C 时，无独立 bias 的 logits 位于 `col(C)`（实际在 `col(CP)`），不等于完全学不了词频：`final_LN` 的可训偏移 beta 可产生 `CP beta`，但仍在同一子空间。加固定 b 后可达集合平移为 `b+col(CP)`。

对严格正 pi，**自由常量 a** 的精确可表示条件为 `softmax(Ca+b)=pi ⇔ log(pi)-b∈col(C)+span(1)`；这是 softmax 对常数平移不变的结果。真实学生若不能遍历所有 a，该条件仅必要。取 `b=log(pi), a=0` 即精确得到 pi；这既注入训练频率，又改变仿射空间，差距变化不能唯一归因于频率机制。

任一固定 checkpoint 上 `rank(CP)≤min(k,d)=128`；未来即使 k=512，d仍128且线性读出时也未解除该限制。此处约束的是 logits/按行常数平移等价类，不能直接把 softmax 概率矩阵的秩写成128；固定 b 还可为原始 logits 矩阵增加一个上下文不变方向。对应经典边界见 [Yang等 §2.1](https://arxiv.org/html/1711.03953v4)。

码唯一、token可成为唯一argmax、任意概率可表示互不等价。无bias且a自由时，v可唯一argmax要求存在a使 `(c_v-c_u)^T a>0` 对所有u≠v成立；例如同范数且互异的sign行取a=c_v即可满足，尽管维度远小于词表。实际a可达性、重复码、固定bias均需另查，不能凭低维/邻距断言不可预测。

CPU常量拟合中 `L(a)-H(pi)=KL(pi || softmax(Ca))≥0`，`grad L=C^T(q-pi)`，Hessian为 `C^T[diag(q)-qq^T]C`，故为凸目标但未必严格凸。有限迭代只报告实际残差与停止证据；这是单一无上下文分布的拟合，**不是有上下文LM的NLL下界**，也不解释完训练差距。剩余经验问题只是在约定终点两个seed的冻结/sign差值及交互是否跨过筛选阈值；若有残差，再讨论竞争解释，当前不预先命名机制或创新。

方法技能引用已核验：Timothy Kassis、Vinayak Agarwal、Yuhuan He、Darshil Patel、Aubrey M. Brueckner（2026），*Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*；[arXiv当前记录](https://arxiv.org/abs/2609.00065)，当前v2（2026-09-02），DOI `10.48550/arXiv.2609.00065`。用于 experimental-design/scientific-critical-thinking 等方法技能来源，不是本轮研究证据；该记录未提供期刊引用或出版商DOI。
