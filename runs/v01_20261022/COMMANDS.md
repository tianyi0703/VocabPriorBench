# V0.1 实际执行命令

日期2026-10-22，工作区 F:\VocabPriorBench，既有 `.venv`；没有安装依赖或下载模型/语料。

完整指令：`CODEX_V01_VocabPriorBench.txt`。开始时根目录尚无该文件，向用户询问后文件于10:04新增，完整读取15955 bytes；未在文件缺失时猜seed或启动GPU。

1. 读取V0必要源码、配置、报告、原始numerical/gpu_console/events/resources/monitor与supervisor归档记录；只读核验对应hash。未运行旧audit_artifacts.py（它会读holdout），未重跑V0GPU。
2. 项目环境CPU执行新增3项+旧7项 unittest；10/10 PASS、内部2.474秒，原输出为 `runs/v01_numerics.log`。这次封存前检查外层进程墙钟未记录，CPU预算保守预留60秒，未把预留声称为实测。
3. `& .\.venv\Scripts\python.exe -m compileall -q scripts\v01 src tests`：封存前静态检查通过。
4. `& .\.venv\Scripts\python.exe scripts\v01\seal.py`：复制27个指令/契约/配置/源码/测试文件至 `runs/v01_20261022/snapshot`，树hash见SNAPSHOT_MANIFEST.json。后续从该快照执行，不修改快照。
5. `$env:VPB_WORKSPACE='F:\VocabPriorBench'`。
6. `& .\.venv\Scripts\python.exe runs\v01_20261022\snapshot\scripts\v01\cpu_supervise.py numerics`：快照版本10/10检查通过、内部2.406秒、外部完整进程5.8456654秒，原输出numerics.log。
7. `& .\.venv\Scripts\python.exe runs\v01_20261022\snapshot\scripts\v01\cpu_supervise.py prepare`：完整进程4.4110412秒；只读train/dev packed和既有码表。选3068完整块，tail两完整块及一partial舍弃；计数389636；dev62601；每seed保存公共初态和具体块顺序。训练统计一次pass成本单列，未打开holdout。
8. `& .\.venv\Scripts\python.exe runs\v01_20261022\snapshot\scripts\v01\cpu_supervise.py diagnose`：完整进程6.2400870秒，两float64常量输出拟合完成；目标与梯度、closure、停止原因见diagnostics.json和diagnostic_events.jsonl。未根据dev改优化设置。
9. `& .\.venv\Scripts\python.exe runs\v01_20261022\snapshot\scripts\v01\supervise_gpu.py`：唯一新增GPU子进程，外部timeout3540秒并保留60秒清理；实际进程预算含导入、预检、12轨迹训练/完整dev评估/保存。精确起止与完整秒数由监督器直接写supervisor_result.json，不再从工具输出手工归档。

实际执行的CPU/GPU命令数组及起止/exit code另由cpu_budget.json、gpu_budget.json保存。固定顺序、seed、配置hash与snapshot树hash保存在events.jsonl；逐step均保留NLL分子/有效目标/同步时间。最后由同一快照 `finalize.py` 复核计数/配对/hash、输出36行metrics、6行contrasts、分类/报告与审查ZIP；该命令结果见同目录finalize.log。

V0报告、源码、配置与原始结果保持原样；本轮没有打开holdout、运行七臂V1、追加数据/模型/码宽/seed或调阈值。
