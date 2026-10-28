# V0.2 后处理与交付代码只读复核

结论：**PASS；未发现必须阻止本轮打包的错误。** 本次只读审查当前后处理代码、封存契约及已生成文本/JSON，另用标准库从 raw metrics 复算标量差值和桥接值。未导入/执行 finalizer 模块，未导入 PyTorch、读取 tensor 对象、运行 GPU、修改训练快照或访问 holdout。本文件是训练后的审计补充，不属于训练前契约。

审查的最终版本：

- `scripts/v02/finalize.py` SHA-256：`3b6db12f2debfdfb14f94dbc07d3e03fdfb596c1d3a1300aaacf85ba77cdc204`
- `scripts/v02/analysis.py` SHA-256：`f05fc532b2b90a9ae9739c82497cde1c49702f32881a9bd5a68c0e744ef31192`

## 有效性 gate 与执行来源

后处理在分类前要求：训练快照 bytes/SHA 与树 hash 不变；八份复用输入和派生产物仍匹配；V0.1 模型依赖逐字节一致；CPU/GPU 阶段成功且预算内；24 条唯一评价覆盖两 seed×四条件×0/100/767；八轨迹顺序、767 步、每步 508 目标、累计 389,636 目标及训练分子/同步时间一致；冻结 hash 与初始 proof 相同，公共参数已更新；评价 raw metrics 与事件逐字段一致；新 checkpoint 只按文件字节核验、不反序列化。

只有全部 12 个 C↔Fp、S↔Sp 桥接格通过，才调用机制分类；否则输出 `BRIDGE_UNRESOLVED / NOT_ASSESSED`。本轮桥接最大绝对差为 **0**，不存在被阈值隐藏的漂移。

除静态检查 gate 外，本审查直接读取唯一 `job_start` 原始事件，确认其 `snapshot` 等于 source manifest 和训练快照的 `ee3b1075a2e6fb0bc89f9e203307dea8784b673f6affc6c218e611eb8c90be00`，`config_sha256` 也一致。训练前快照为 16 个文件；后写的 analysis/finalize 明确放入独立 `report_snapshot`，没有冒充训练前封存内容。包排除私有 tensor，不能被描述成可独立恢复完整训练状态的包，现报告已明确这一边界。

当前 builder 对本次完整有效运行使用严格断言；不应将本次 PASS 当作它能为任意缺失或失败运行生成完整报告的保证。

## 算式、阈值与当前结论

`analysis.py` 的六差值为 `G=S−C`、`D_r=RS−C`、`D_1=S−U`、`N_cont=U−C`、`N_sign=S−RS`、`I=D_1−D_r`，同时检查 I 的另一表达和 G 的两条恒等式。未取绝对值覆盖符号、未先舍入、未取均值合并 seed。A→B→C→D 优先级、`>tau`/`<=tau`、C 的“不满足整体 A”以及 B/C 不另加 G 门槛，均与封存契约一致。

从本轮 raw metrics 独立复算的主要终点：

| seed | G | D_r | D_1 | N_cont | N_sign | I |
|---|---:|---:|---:|---:|---:|---:|
| 1729 | +0.168618308 | +0.179373889 | +0.169517130 | −0.000898822 | −0.010755581 | −0.009856759 |
| 2718 | +0.174298577 | +0.184874779 | +0.175379972 | −0.001081395 | −0.010576203 | −0.009494808 |

两 seed 均满足 B 的 `D_r>0.05` 且 `D_1>0.05`，均不满足 A/C。故 `DIRECTION_RESIDUAL_AT_BOTH_NORMS` 正确；上述恒等式在本轮原精度标量下误差为 0。该代数检查不是因果中介证明。step0/100 未用于选择终点或修改阈值。

## 报告与资源

现 `V02_REPORT.md` 的最终 NLL、六差值、两条负 norm effect、桥接、范数舍入和 EOS/unused 类别说明，与 CPU 审计及原始结果一致。直接核对 raw metrics 后，step0 全部 context gain 为负、step100/767 全部为正，报告对应描述成立。

GPU 账本与 supervisor 的完整进程耗时均为 **156.68108119996032 秒**；与此前累计相加为 **488.80411730000657 秒**。没有用纯训练时间代替预算。CPU 准备/检查完整进程累计 23.772711600002367 秒；报告将额外后处理算术内部时间另列，没有冒充完整进程计时。设备、8188 MiB、Torch/CUDA/cuDNN 和 driver 字段与 environment JSON 对应。

报告保留了关键边界：两 seed/单 donor/语料/宽度的有限步探索；同初态/步数不证明收敛；rS 可改变列空间；可逆右变换可保函数而改行余弦；L2 尺度不代表所有尺度方案；全部 FP32、无 bit packing；近邻中的尺度恢复等不是本项目创新；残差不证明语义损失或论文成立。`PAPER_READINESS=NOT_ESTABLISHED` 与固定停止状态符合本轮指令。

## 打包名单与禁入内容

`pack()` 只选新 REPORT/RUN 中规定的文本后缀，排除 `private`、`__pycache__`、`.venv`；旧来源使用明确文件名单。已确认新增的三份旧原始日志 `events.jsonl`、`gpu_console.log`、`supervisor_result.json` 在明确历史名单中。最终来源补充还纳入 `reports/source_manifest.json`：只对这份旧 JSON 文件本身核验 SHA，与 V0.1 `preserved_v0.source_manifest_sha256` 相同（`1cfe2ef82ff3118cfd5a898bc4984b62ea47197acd8c5df207b1098ad2af8784`），未遍历或读取其中引用的 data/holdout。此最后增量已单独静读与实际文本 SHA 复核，没有算法或训练变化。没有枚举 `data/`、`artifacts/` 或旧 private 来收集内容，也没有 holdout 读取路径。

本次仅清点的当前新文本树候选全部位于报告/运行目录，无权重或语料文件。旧 metadata 内存在的历史 holdout hash 不是本轮打开或重新哈希 holdout。按当前目录内容和后缀规则，权重/码表 `.pt/.bin/.safetensors`、数组及 parquet 不会入包；块索引/排列 JSON 是来源证明，未包含 tokenized corpus。

包内 manifest 排除自身；ZIP 自身 SHA 在包外；pack 会检查 CRC、成员集合及每个成员的 bytes/SHA。此复核未自行创建 ZIP，最终成员和完整性以 pack 输出及 `BUNDLE_MANIFEST.json` 为准。建议按已完成审计直接执行一次 build/pack，随后停止，不因残差追加实验。
