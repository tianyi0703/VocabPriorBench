# V0.2 分析模块结果盲态审查

本文件对应运行后独立报告构建器的纯算术模块，不修改已封存训练代码、契约或判定规则。实现与以下合成测试期间没有读取本轮loss、metrics、语料或holdout，也没有导入torch、旧common或初始化GPU。

API：`contrasts({C,U,RS,S})` 返回六个有符号差值、三个浮点恒等误差及最大绝对误差；`classify(final_rows, tau=.05)` 接收两个含seed及六差值的行，返回label、逐seed A/B/C布尔细节/未满足原因及跨seed谓词。root先独立处理BLOCKED/INVALID/INCOMPLETE/BRIDGE_UNRESOLVED，只有有效完整桥接才调用分类。模块本身不替代证据gate。

公式逐项为 G=S−C、D_r=RS−C、D_1=S−U、N_cont=U−C、N_sign=S−RS、I=D_1−D_r；另验G=N_sign+D_r=D_1+N_cont以及I=N_sign−N_cont。保持正负号，不舍入/夹断。

严格A→B→C→D：A要求两seed所有五条件；B只要求两seed D_r、D_1均严格大于tau；C要求两seedabs(D_r)≤tau、两seed D_1>tau且全局A未通过。B/C没有新增G门槛。没有给tau加epsilon，也不对两seed先取均值。

静态AST检查模块无import且没有open/exec/eval/compile调用；没有文件IO。合成分类通过项：

- A norm dominant → `NORM_DOMINANT_IN_THIS_SCREEN`
- B accepts negative G → `DIRECTION_RESIDUAL_AT_BOTH_NORMS`
- C accepts compensation and negative G → `INTERACTION_OR_COMPENSATION`
- D other direction pattern → `INCONCLUSIVE_OR_OTHER_PATTERN`
- A includes abs(D_r)=tau and abs(D_1)=tau → `NORM_DOMINANT_IN_THIS_SCREEN`
- B requires strict > tau; C permits abs(D_r)=tau → `INTERACTION_OR_COMPENSATION`
- A requires strict G/norm effects > tau → `INCONCLUSIVE_OR_OTHER_PATTERN`
- Two seed disagreement is not averaged away → `INCONCLUSIVE_OR_OTHER_PATTERN`

另验证缺少第二seed、重复seed和非有限值均明确拒绝，共3项。所有合成差值恒等误差<1e−12。

标准库子进程测试内部墙钟 **0.063287300秒**，范围为Python脚本开始计时后至上述检查完成，含标准库加载/源文件读取；不是完整进程GPU计时。未运行任何训练或数值拟合。

分析模块SHA256：`f05fc532b2b90a9ae9739c82497cde1c49702f32881a9bd5a68c0e744ef31192`。本审查只证明公式与边界分派在合成例上的实现符合契约，不预示本轮真实数据会归入哪个标签，不支持因果中介、统计等价或新颖性主张。
