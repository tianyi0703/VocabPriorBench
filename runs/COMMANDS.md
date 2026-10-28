# V0 实际命令与顺序

2026-10-21，PowerShell，工作区 F:\VocabPriorBench。此记录列出实际已执行的核心命令；shell输出、程序逐步JSONL与hash为原始证据，本文不是模拟日志。

1. `Get-Content -LiteralPath 'C:\Users\USER\.codex\attachments\a8e162c5-0a58-435c-9b1f-0e2bd4d056c3\已粘贴的文本.txt' -Raw`：读取用户完整任务。
2. 检查 `F:\AGENTS.md`、项目AGENTS、目录、`git status --short`、OS/RAM/磁盘、`python/py/conda/nvidia-smi`。起始空项目，git报告不是仓库；默认Python为Anaconda3.13.5/torch2.10.0+cpu，无可用CUDA build。硬件为真实RTX4060 Laptop。系统检查原始文件为 system_inventory.json、initial_nvidia_smi.csv，完整CUDA环境由GPUrunner记录。
3. `& 'C:\Users\USER\AppData\Roaming\uv\python\cpython-3.12.15-windows-x86_64-none\python.exe' -m venv .venv`
4. `& .\.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 *> runs\install_torch.log`
5. 与下载CUDA轮子并行执行：`& .\.venv\Scripts\python.exe -m pip install transformers==4.57.1 tokenizers==0.22.1 safetensors==0.6.2 pyarrow==22.0.0 numpy==2.2.6 PyYAML==6.0.3 psutil==7.1.1 requests==2.32.5 huggingface-hub==0.36.0 *> runs\install_dependencies.log`。两条安装均成功，仅项目venv。最终解析版本保留pip_freeze.txt；pip_check.txt无依赖错误。
6. `& .\.venv\Scripts\python.exe -u scripts\download.py *> runs\downloads.log`：初始显式allowlist下载，只有一个donor safetensors和一个数据train shard。审查后补充revision元数据校验，再运行同命令、输出到download_verification.log，文件SHA256/Git blob均匹配；没有重复下载权重。
7. 辅助CPU测试：全局只读 `python -m unittest discover -s tests -v`，torch2.10.0+cpu，7/7通过，2.992s（此项没有冒充项目环境/GPU）。
8. `& .\.venv\Scripts\python.exe scripts\check_numerics.py`：正式项目环境CPU测试，7/7通过，2.898s，原始numerical_checks.log。
9. `& .\.venv\Scripts\python.exe -u scripts\prepare.py *> runs\prepare.log`：一次CPU数据/PCA/初态准备成功；prepared_utc、参数inspection和token计数见日志/manifest。
10. `& .\.venv\Scripts\python.exe scripts\audit_artifacts.py`：GPU前审计通过。
11. `& .\.venv\Scripts\python.exe -u scripts\supervise_v0.py`：唯一GPU作业；supervisor启动runner，外部timeout7100s。runner串行三臂短序列预检、100步主训及评估保存；gpu_console.log、v0_20261021/events.jsonl为原始日志，gpu_budget.json为全作业记账。没有GPU失败或重跑。
12. `& .\.venv\Scripts\python.exe scripts\audit_artifacts.py`：GPU后审计通过，包括冻结/更新、全参数有限、相同prefix与50800token/臂、100step/臂。
13. `& .\.venv\Scripts\python.exe scripts\build_report.py`：从真实JSON与已读文献证据生成V0报告，并在manifest保留执行快照及交付快照。

执行后审计：真实supervisor输出为 `Runner exit=0; supervisor wall=19.406s`，初版没有自动保存该输出；`python scripts\archive_supervisor_measurement.py` 将这个已观测值按原1ms显示精度封存为supervisor_result.json，预算改用包含导入/启动的19.406秒，保留原始runner内部11.2113356秒及其日志。未编造外部起始UTC。监督器现已加保存与旧证据覆盖保护；没有再次运行GPU。随后重跑CPU artifact audit和build_report均成功。最终源码hash与执行前hash分别保存。固定数据卡许可版本的内部冲突已加入报告与manifest，没有改动训练数据。

开发期CPU静态检查中一次使用 `python -m py_compile scripts\*.py`，Windows没有展开通配符，返回Invalid argument；随后改用 `python -m compileall -q scripts src tests`成功。此错误发生在GPU前，不是训练失败。未隐去OOM或改用其他硬件结果。
