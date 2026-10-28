# V0.2 实际执行命令（已完成，不是续训指令）

工作目录 `F:\VocabPriorBench`，PowerShell；已有 `.venv`，未安装/升级依赖。

```powershell
$env:VPB_WORKSPACE='F:\VocabPriorBench'
& .\.venv\Scripts\python.exe scripts\v02\create_runner.py
& .\.venv\Scripts\python.exe -m compileall -q scripts\v02 tests\test_v02_tables.py
& .\.venv\Scripts\python.exe scripts\v02\seal.py
& .\.venv\Scripts\python.exe runs\v02_20261022_193903\snapshot\scripts\v02\cpu_supervise.py prepare
& .\.venv\Scripts\python.exe runs\v02_20261022_193903\snapshot\scripts\v02\cpu_supervise.py numerics
& .\.venv\Scripts\python.exe runs\v02_20261022_193903\snapshot\scripts\v02\supervise_gpu.py
& .\.venv\Scripts\python.exe scripts\v02\finalize.py build
& .\.venv\Scripts\python.exe scripts\v02\finalize.py pack
```

GPU supervisor 账本另保存实际 child 完整 argv、PID、UTC、exit code 和完整墙钟。
CPU supervisor 账本另保存 prepare 与 unittest 的完整 argv/起止/耗时。
封存脚本和 GPU supervisor 会拒绝复用现有执行输出；本轮已结束，不重跑。
finalize.py 是训练后报告/校验/打包代码，未替换训练前契约或执行快照。
