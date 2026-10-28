# GitHub 发布记录

发布日期：2026-10-28（Asia/Shanghai）。研究结束决定仍为2026-10-23。

目标公开仓库：[tianyi0703/VocabPriorBench](https://github.com/tianyi0703/VocabPriorBench)。本地发布副本位于 `publication/github/`，该目录作为独立 Git 工作区；原项目工作区保留未删节的执行证据与私有输入。

此次发布用于总结已完成的工程筛选和探索结果。以“冻结／sign 差距”为核心的论文立项尝试已结束；发布不启动实验，不恢复该路线，也不改写原报告结论。

公开内容包括总结、代码、配置、执行快照、结果表、必要日志、来源记录和两份 **PUBLIC** 审查包。公开副本仅删节 Windows 用户主目录中的账户名及 GPU 唯一硬件标识。`PUBLICATION_MANIFEST.json` 记录各文件原字节与公开字节的大小/SHA256和删节类别；原始 ZIP 的 SHA256 作为来源记录保留，但原始 ZIP 本身不上传。

`V01_PUBLIC_REVIEW_BUNDLE.zip`、`V02_PUBLIC_REVIEW_BUNDLE.zip` 是由本地原始审查包生成的公开派生包，不是原包的原始字节。其 `PUBLIC_BUNDLE_MANIFEST.json` 验证公开内容；旧 `BUNDLE_MANIFEST.json` 改名为 `ORIGINAL_BUNDLE_MANIFEST.json` 保留为历史来源，不能用于验证删节后的 payload。公开文件中的旧哈希仍描述原执行产物，不伪装成新的执行记录。

环境目录、donor/model 权重、PCA 码表 tensor、packed 数据、独立 holdout、全部 private 目录、安装/下载日志和凭据不发布。本次导出不打开、读取或重新哈希 holdout，也不加载任何 tensor 或启动 CUDA。

发布树可用系统 Python 的标准库只读核验：

```powershell
python scripts/publication/verify_release.py
```

它核验公开文件及派生 ZIP 的字节/SHA，并复算既有结果表与桥接值；不下载数据、不训练模型。仅有公开仓库不足以恢复原公共初态、训练 prior、码表和训练块 tensor，不能据此承诺一键精确重跑。训练前执行快照和训练后公开 Git 提交是不同证据。

未为整个项目另选开源许可证或替上游模型/数据授予许可。上游 revision、许可记录冲突及评测污染 UNKNOWN 沿用原来源记录。当前项目状态以 [RESEARCH_STATUS.md](RESEARCH_STATUS.md) 为准；V0/V0.1/V0.2 的旧 `NEXT` 字段是历史状态。
