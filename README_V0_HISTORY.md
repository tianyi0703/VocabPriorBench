# VocabPriorBench — V0

**当前状态（2026-10-23）：当前路线暂存，以“冻结／sign 差距”为核心的本次论文立项尝试已结束。** 决定与停止范围见 [RESEARCH_STATUS.md](RESEARCH_STATUS.md)；最新完成证据见 [V0.2 报告](reports/v02/V02_REPORT.md)。以下保留 V0 的历史执行说明。

本仓库执行文献差异核查、实验契约、A/B/C 最小实现、数值检查和本地 RTX4060 Laptop 资源测量。V0 结束即停止，未授权 V1 或正式 pilot。研究问题与新颖性结论见 [V0 报告](reports/V0_REPORT.md)，运行契约见 [EXPERIMENT_CONTRACT.md](reports/EXPERIMENT_CONTRACT.md)。

## 本机安装（PowerShell）

实际目录 `F:\VocabPriorBench`。已有全局 Anaconda 的 torch 是 CPU 版；项目单独使用 Python 3.12.15 venv，未卸载旧环境或修改驱动。以下命令来自本轮执行记录：

```powershell
Set-Location F:\VocabPriorBench
& 'C:\Users\USER\AppData\Roaming\uv\python\cpython-3.12.15-windows-x86_64-none\python.exe' -m venv .venv
& .\.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
& .\.venv\Scripts\python.exe -m pip install transformers==4.57.1 tokenizers==0.22.1 safetensors==0.6.2 pyarrow==22.0.0 numpy==2.2.6 PyYAML==6.0.3 psutil==7.1.1 requests==2.32.5 huggingface-hub==0.36.0
```

依赖最终解析版本另存 `runs/pip_freeze.txt`。需要约10GB额外磁盘空间容纳含CUDA的环境和缓存；模型/数据本身约175MB，PCA码表与最终状态另占空间。只下载配置允许的一个 immutable donor checkpoint 和一个 WikiText-2 train shard，绝不调用托管训练/推理 API。

## 检查与复现

```powershell
& .\.venv\Scripts\python.exe scripts\check_numerics.py
& .\.venv\Scripts\python.exe -u scripts\download.py
& .\.venv\Scripts\python.exe -u scripts\prepare.py
& .\.venv\Scripts\python.exe scripts\audit_artifacts.py
& .\.venv\Scripts\python.exe -u scripts\supervise_v0.py
```

`prepare.py` 只在 CPU 构造独立输入/输出 PCA 表、文章划分、token blocks 和公共随机初态；已缓存文件也必须匹配 pinned Hub revision 的 LFS SHA256 或 Git blob。`check_numerics.py` 保留原始 unittest 日志。`supervise_v0.py` 是实际 GPU 入口，串行预检及 A/B/C，每臂100步；外部 watchdog 与累计7200秒账本覆盖失败/评估/热身。开始前至少需要1.5GiB设备剩余显存。

已有 `runs/v0_20261021/events.jsonl` 时 runner **拒绝覆盖或自动重跑**。本轮已执行后的复查只运行 tests/audit；重新实验应另建明确run目录并保留旧账本，在人工检查剩余预算后修改run目录。不要删除日志来绕过预算。

实际配置为 [configs/v0.yaml](configs/v0.yaml)。默认seed1729，码宽128，学生2层×128，microbatch2×累积2，序列128；每块127个有效 next-token labels，每臂400块/50800有效token，bf16 autocast、FP32权重、完整softmax。无分块CE或二进制kernel。A两表可训，B相同数值冻结，C同源sign后冻结；输入输出始终两张表。

## 文件

- `reports/V0_REPORT.md`：状态、证据、实际结果、限制与条件性V1设计。
- `reports/source_manifest.json`：donor/data revision，来源文件校验，词序、PCA、初态和代码哈希。
- `reports/resource_profile.json` / `gpu_budget.json`：实际各臂资源和累计GPU作业墙钟。
- `reports/literature_*.md` / `LITERATURE_VERDICT.md`：8篇原始全文核查及公开资源审计。
- `runs/`：安装、下载、准备、CPU测试、逐step事件、设备采样、依赖与控制台原始日志。
- `artifacts/`、`data/`：本地可重建制品；`data/holdout_documents.json` 未参与分词/评估/调参。

WikiText 固定来源卡的YAML标为 CC-BY-SA-3.0/GFDL，正文却写CC-BY-SA-4.0，精确版本冲突已保留在报告/manifest，未宣称无歧义；donor 来源 Apache-2.0。没有把学生文档去重当作 donor 去污染；donor/评估重叠为 UNKNOWN。本V0不是原论文完整复现，也不以单seed短跑排序作效果或新颖性主张。
