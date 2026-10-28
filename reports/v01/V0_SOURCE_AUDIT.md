# V0 关键源码与产物审计（V0.1 前补充）

结论：`LIMITED_V0_HISTORY`。当前模型、runner、基础测试和配置与 V0 执行前记录逐字节一致；当前输入产物及三个最终 checkpoint 与既有哈希记录一致。`prepare.py`、`supervise_v0.py`、`build_report.py` 的历史文本未找到，无法恢复真实 diff，不能将其变化直接断言为“仅元数据修改”。这一历史缺口本身不要求重跑 V0；V0.1 应使用当前已核对产物，并完整封存新执行版本。

本审计只读取源码、配置、日志与指定二进制产物，未执行原 V0 脚本、未调用 GPU、未打开或哈希 `data/holdout_documents.json`、未运行会读取 holdout 的 `audit_artifacts.py`。既有 `artifact_audit.json` 只作为历史检查记录读取，不能当作本轮重新执行的检查。

## 三个源码哈希阶段

完整 SHA-256、逐文件比较和产物复核见 `V0_SOURCE_HASH_AUDIT.json`。`source_files` 与 `code_files_before_gpu` 对共有路径记录相同哈希，后者由 runner 在实际训练前写入。

| 文件 | 当前相对执行前版本 | 当前相对交付版本 | 能支持的判断 |
|---|---|---|---|
| `src/vocabpriorbench/model.py` | 相同 | 相同 | 当前因果与接口实现等同记录版本 |
| `scripts/run_v0.py` | 相同 | 相同 | 当前训练、计数、同步计时和保存实现等同记录版本 |
| `tests/test_numerics.py`、`scripts/check_numerics.py` | 相同 | 相同 | 当前基础测试等同原检查版本 |
| `configs/v0.yaml`、`src/vocabpriorbench/__init__.py` | 相同 | 相同 | 配置与导出未变 |
| `scripts/prepare.py` | 不同：`1ac476fd…` → `e3a802d2…` | 相同 | 原执行文本缺失，不能给真实 diff |
| `scripts/supervise_v0.py` | 不同：`69b90891…` → `b739afea…` | 相同 | 原执行文本缺失，不能给真实 diff |
| `scripts/build_report.py` | 不同：`8360ae24…` → `bd958ba3…` | 相同 | 原执行文本缺失，不能给真实 diff |
| `scripts/archive_supervisor_measurement.py` | 执行前无记录 | 相同 | 后补归档脚本，不能称为原 GPU 执行代码 |

已搜索工作区源码/日志/报告路径，未找到这三个旧 `.py` 文本、源码快照或源文件归档；工作区无 `.git`。存在的 `__pycache__` 不等于可核验的旧源码文本。哈希只证明字节相同或不同，不能从中恢复修改内容。所有 13 个当前交付源码路径均匹配 `delivery_source_files`；这些本地记录不是独立签名的运行证明。

## 与比较有关的实现核对

- 因果和 padding：`model.py:41–49` 在有 mask 时使用下三角与有效 key 的交集；无 mask 时使用 SDPA 的 `is_causal=True`。基础测试分别检查未来扰动、截断前缀、内部 padding key 扰动和全 padding 有限性。
- 标签与分母：`next_token_labels` 将下一个 token 对齐到当前 logits，最后位置及任一相邻无效 mask 对置为 `-100`。runner 所有训练/评价 NLL 使用完整 logits、交叉熵求和与 `labels != -100` 的分母。原训练明确断言前 400 块都是完整块，因此各 microbatch 的有效数相等，梯度累积按每 microbatch 平均再除 accumulation 与该前缀的 token 平均一致。V0 不包含跨块的末位到下一块首位目标。
- 冻结与公共初态：输入/输出码表是不同 Parameter 和不同 storage；A 两表可训练，B/C 两表冻结，C 同时 sign 输入及输出。每臂加载同一 `common_init.pt` 后校验公共状态哈希；最终核对每个公共参数更新、所有可训练参数非零有限梯度与码表前后哈希。公共初态、训练前缀在三臂的原始事件/资源记录中相同。V0.1 的 S0/Sp 只改变输出端，不能当作原 C 继续训练。
- 同步计时：runner 在评价段、热身段、每个训练 step 的计时边界使用 `torch.cuda.synchronize()`；保存最终模型前也同步。稳定吞吐来自预定 step 11 起的有效目标数除以这些同步后的 step 时间；不是整个进程的吞吐。
- 保存对应：重新计算 `final_A.pt`、`final_B.pt`、`final_C.pt` 的文件 SHA-256，分别同时匹配 `resources_A/B/C.json` 和 `arm_end` 事件。每臂恰有 100 个 step 事件，其有效目标求和均为 50,800，匹配资源汇总；不把此检查冒充重新训练。
- 现有输入：`codes.pt`、`transforms.pt`、`common_init.pt`、`vocab_order.json`、`packed.pt` 和 V0 配置的当前文件 SHA-256 均匹配原 manifest 对应记录。这里仅哈希明确列出的 train/dev 打包文件，不遍历并读取 manifest 中全部 data 路径。

原 `runs/numerical_checks.log` 保留 7 项 CPU 测试通过（2.898 秒）的原始输出。V0.1 新 CPU 日志 `runs/v01_numerics.log` 再运行未改动的 7 项基础测试及 3 项输出干预测试，10 项均通过（2.474 秒，PyTorch 2.10.0+cu128，测试张量全部位于 CPU）。

## 时间字段规范化

以下独立字段写入 `V0_SOURCE_HASH_AUDIT.json` 的 `normalized_v0_timing`，不改写任何 V0 原记录：

| 字段 | 值 | 来源/范围 |
|---|---:|---|
| `runner_start_utc` | `2026-10-21T12:15:17Z` | `job_end` 原始事件；秒级 UTC |
| `runner_end_utc` | `2026-10-21T12:15:28Z` | 同上 |
| `runner_elapsed_seconds` | `11.211335600004531` | runner 单调计时，导入后开始 |
| `supervisor_start_utc` | `null` | 初始 supervisor 未记录；不反推 |
| `supervisor_end_utc` | `null` | 初始 supervisor 未记录；不反推 |
| `supervisor_elapsed_seconds` | `19.406` | 原工具 stdout 测量后由 `archive_supervisor_measurement.py` 归档，显示精度 0.001 秒，包含子进程导入/启动 |

原预算 ledger 中 `wall_seconds=19.406` 与保留的 runner `start_utc/end_utc` 属不同范围，不能把这对 UTC 误称为 supervisor 起止时间。V0 的项目预算仍按 19.406 秒保留，V0.1 新 GPU 完整进程墙钟应另行累计。
