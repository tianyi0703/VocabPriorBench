# V0.1 补充勘误（V0.2，2026-10-22）

本文件补充解释已交付证据，不修改 V0/V0.1 报告、候选稿、配置、快照或日志，不为此重跑旧实验。以下三点依据实际文本与封存源码核对。

| 项目 | 原记录与文件证据 | 准确口径及影响 |
|---|---|---|
| 候选指标不一致 | `reports/v01/V01_REPORT.md:173` 描述“训练pi加权投影margin变化”；`reports/v01/RESIDUAL_HYPOTHESIS_CANDIDATE.md:7–8` 定义 `u_v=T_v/||T_v||₂`、`m_v(T)=1−max_{j≠v}u_vᵀu_j`、`x_S=m(C)−m(S)` | 后者是**未加权最大余弦近邻margin**。训练频率仅用于分箱/抽样；不是在max/cosine中加pi权重。两种描述不是同一指标，不能互换或据其中一种补写另一种的验证。本轮不计算、拟合或执行任一种margin诊断。 |
| 候选盲态与注册边界 | 候选稿自述初稿02:26 UTC时生成者未读本轮loss/metrics。原 `snapshot/SNAPSHOT_MANIFEST.json` 封存于02:22:16 UTC，文件列表不含该候选稿；GPU supervisor记录02:23:25–02:28:38 UTC | 只能称 **V0.1探索中提出的候选**。当前包不能独立核实其自述的结果盲态，亦不能称已验证的独立预注册假设。这里既不认定自述为假，也不把自述当成独立证据。先前封存的V0.1实验契约与筛选规则不因此被改写。 |
| 逐轨迹时间起点不同位 | 封存 `scripts/v01/run_gpu.py:80` 先设置 `trajectory_start=time.perf_counter()`，再启动monitor及构建/加载模型、CUDA迁移、optimizer；`:90` 才写 `start_utc=utc()`；`:134` 的wall从前者计算 | `end_utc−start_utc` 与 `wall_seconds` 不是同一开始区间，不应强行对齐或倒推出旧起点。外部完整子进程计时 **312.71703610004624秒** 及原预算记账仍保留。V0.2的修正要求是在同一程序位置邻接记录UTC和monotonic起点，明确训练/评价/轨迹/runner/supervisor口径；实际落实以本轮执行快照与新日志为证，本补充文件不代替执行验证。 |

证据路径均相对项目根目录 `F:\VocabPriorBench`。上述snapshot指 `runs/v01_20261022/snapshot`；外部计时来源为 `runs/v01_20261022/supervisor_result.json`。本次未读取语料、holdout或私有tensor。

为便于复核，以下是读取时旧文件的SHA-256（大小写不影响摘要值）：

| 旧文件 | SHA-256 |
|---|---|
| `reports/v01/V01_REPORT.md` | `60e08e58fc65aff2af47c54462da0dc8ecb7866152a1b751123b8b9c216a2d63` |
| `reports/v01/RESIDUAL_HYPOTHESIS_CANDIDATE.md` | `5e89e21feb0adb9d473dfa1f6500a7d1c812272641b37bd595561fca3c35f992` |
| `runs/v01_20261022/snapshot/scripts/v01/run_gpu.py` | `20d763e8f737a1de6d856b902532ce83bc802be05d53f4efdc6b2bcbc560816f` |
| `runs/v01_20261022/snapshot/SNAPSHOT_MANIFEST.json` | `c30719be6966c6221a0801b5887cbbbf6c7af6fcf83da63a3578cf5e2b4a23d2` |

原snapshot tree SHA-256为 `35d17fbdd376690a8b49458581f1ac14c0219ce8a322a2cc9cfbf137814757c9`。单个文件/树哈希能证明具体字节对应关系，不能独立证明研究者当时看过哪些结果。
