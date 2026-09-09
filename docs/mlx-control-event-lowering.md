# 完整控制窗口的 C++ 事件接入

本增量将现有描述符绑定的 RV64 控制窗口接入独立事件模型。它不是 Rocket CPU 模型：原生窗口从描述符 ROM 取运算叶程序，用 MemoryPort 访问张量，不执行系统 ELF 取指或 RISC-V load/store。因此窗口对齐不能代替 Chipyard 整模验收。

## 资源和时序

schema v6 的 controller 描述符显式区分 `domain=control` 与 `domain=memory`。控制前端拥有固定 512 B 寄存器（32 个 GPR、32 个 FPR）及最多 32 条叶指令模板，最多一个控制源驻留。ALU、乘法、浮点、分支共用单条在途控制指令资源；不借用 PE 的 RF/SPM/ROM、计算/SFU 或独立搬运转换器。

控制运算、搬运转换和阵列计算可并发；三者的 DMA 事务竞争同一有限通路，响应消费后下一边沿才重新可用。控制前端请求/响应节拍以当前准入为起点；指令完成后下一边沿才发射后继。逐条延迟来自 MLX 控制选项，不引用 DFU 指令拍数，也不输入原生测得耗时。

空 add/argmax 等窗口执行时间为 0，但源完成发布仍需要一个边沿，并且占用控制前端直到发布。它不同于不占独立搬运前端的 memory view。arange(0) 仍执行 init；空 all 执行 init 并回写 true，均不能作为无工作跳过。

## 编译与数据相关控制

`src/mlxsim/model_control_events.py` 重新核对 canonical RV64 机器字、dtype、输出形状、广播、轴与模式，覆盖 arange、整数 add/mul、le/ge、Boolean and/all/guard、整数和 FP16/FP32 argmax。init/body/advance/compare/select 均保留所有动态实例；循环使用紧凑 repeat IR，非规律选择分组保留原顺序，超过 IR 容量时拒绝而非截断。

argmax 的 select 分支决定是否执行两条更新指令，不能仅按元素数量估算。当前窗口接口要求真实分支取向序列；对照从实际原生执行的 select PC 路径提取布尔值，不传递时间戳、持续时间或输出 logits。guard 要求实际布尔输入与期望一致；未知/失败 guard 不生成成功路径预测。独立重放重新验证这些见证与实际执行对应。

这是成功执行路径的**时序编译接口**，不是在事件核心里重新计算数据或执行分支。完整源图中的见证生成、guard 失败传播和依赖发布仍需整图接入；不能把未验证分支值当作模型正确性证据。

编译 CLI：`scripts.compile_mlx_matrix_events --family control`（沿用旧模块名）。独立重放：`scripts.verify_mlx_control_event_alignment`。两者均明确不提供整模或 Rocket 通过标志。

## 验证范围

- `control-event-regression-001`：221 项回归通过，其中控制新增 45 项。
- `control-event-safety-001`：202 个事件输入的 ASan/UBSan/LSan 重放（168 正常、34 预期拒绝）；源、输入和完整报告摘要核对，trace 另核对服务互斥、控制驻留、端口节拍和在途积分。
- `control-event-alignment-001`：28 个完整控制窗口独立重跑。覆盖 9 类入口的默认/非默认时序、FP16/FP32 NaN 与首次并列最大值、strided 输入、浮点 literal 比较及空形状。数值输出与原生功能合约一致，事件周期/指令/流量一致；回归还逐条比较独立计算的发射时刻。
- 另有控制／搬运／PE 同时执行、共享 DMA 仲裁、空控制发布和多控制窗口节拍重新起算、资源扩容及跨域操作拒绝检查。
- 当前引擎重新核对原有 22 个矩阵、48 个向量及 18 个搬运窗口，旧 schema 行为保留。

数值窗口使用固定延迟本地 TensorMemoryPort；不声称覆盖 cache、外部错误重试或真实 CPU 取指时序。安全重放和窗口误差为零只验收上述范围。

## 下一步与优先级

matrix/vector/memory/control 四类独立窗口现已具备事件入口，但完整 batch/源图、逐实例块间依赖、跨源原生 whole-tick 仲裁顺序、模板装载与系统访存边界尚未对齐。整模事件覆盖和 `(C_event-C_e2e)/C_e2e` 仍没有结果。

既有六个完整运行及捕获来源保持不变。先推进 Chipyard 与 MLX P0 主线验收，GPU 新实验等待，RTL/PPA 最后；本增量无 RTL 修改。
