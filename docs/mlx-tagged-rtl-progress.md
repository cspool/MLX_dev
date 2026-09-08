# M4 实施状态：tagged PE 前端与共享 RF

状态：第一、二阶段证据已复核，M4 已开始。当前完成的是单 PE 的可综合控制/RF 前端组件，不是完整加速器 RTL，也不是 M4 或 PPA 验收。完整实施要求仍见[总方案](mlx-gpgpu-toolchain-scheduling-gap-plan.md)和[交接要求](mlx-native-to-rtl-handoff.md)。

后续增量已加入独立的功能 FP16 FU，见下文；但用户新要求以完整模型推理验证系统，目前优先转入[模型级验证](mlx-model-e2e-validation.md)，不继续扩展 RTL/PPA。以下组件证据均不能替代模型级端到端通过。

## 已实现的电路

入口为 [`mlx_tagged_pe_control.sv`](../rtl/mlx/tagged/mlx_tagged_pe_control.sv)。该文件独立于旧 `mlx_array_pe_tile.sv`、旧 Chipyard RTL 后端和系统阶段的 DPI 包装。开始编译该组件时，runner 会检查 M0–M3 的证书、源码、构建身份及系统证据仍然对应。

| 部分 | 实际状态或数据通路 |
|---|---|
| 程序与配置 | 32 个 64-bit 指令字、配置有效位，以及最多 4 个块描述符；准入检查模板范围、规范编码、寄存器范围、重复 ID 和 RF 分区重叠 |
| 每上下文状态 | 16-bit block/layer、8-bit epoch、16-bit iteration/trip count、5-bit 相对 PC、active/inflight、寄存器有效位与 RF base |
| 选择与发射 | 按 `(layer, block)` 选择操作数就绪且外部资源可用的候选；选中槽实际选择指令字及三个源操作数；每周期至多一次接受发射 |
| 完成反馈 | 逐槽核对 block/epoch/iteration/PC；只有匹配的完成事件推进 frontier 或循环/退休，重复、过期和未发射完成均报错 |
| 共享 RF | 全 PE 只有 16 个物理向量寄存器，三个读端口、一个写端口；上下文通过 RF base 分区，不为每个槽重复分配 16 个向量 |
| 数据可见性 | load/compute 完成必须拥有该周期写回；xfer 输入到达检查实例与邮箱空闲；迭代推进清空该上下文有效位 |
| 故障处理 | 错误事件不推进状态、不写 RF，置 sticky error；忙时不允许覆盖配置。运行中故障当前需要外部复位恢复 |

候选使用周期开始的有效位和 inflight 状态；本周期写回不能使同一周期的新候选读取刚到达的数据。`resource_ready_i` 由后续 FU/SPM/网络仲裁决定，不能绑成常数来冒充完整发射闭环。`issue_valid_o` 是每周期重新选择的候选，`issue_fire_o` 才表示接受；未接受的候选不改变任何上下文进度。

完成接口用于已经获得完成/写回仲裁的事件，不是把所有 FU 的待完成 valid 无条件送入控制器。多个不同槽可以同周期完成，但物理 RF 写回只能有一个。接收 input 写回与本地运算写回采用不同检查规则，不能将输入邮箱覆盖与普通临时寄存器复用混为一谈。

## 前端首版如何验证（历史范围）

首版 [`tagged_pe_control_driver.cc`](../tests/tagged_pe_control_driver.cc)加载公共 PE 机器字，并按同一 Program 的块信息设置本地配置端口。该历史报告中 RF 数据和上下文状态由 RTL 保存，C++ 服务提供 FU 运算、有限 SPM 请求与单 PE loopback 包的延迟/返回；所有请求都来自 RTL 实际接受的指令。当前默认 FU 路径已更新，见下文“FU 增量”。

独立的 C++ `Simulator` 在每个有效执行周期后用于比较状态，不向 RTL 提供候选、预排发射顺序或 golden 结果。测试同时比较 admit/issue/complete/retire 事件、输出、资源守恒和重叠计数；额外 MLIR 用例还使用源表达式的独立期望值检查三源操作与临时值复用。

这证明了控制与 RF 的闭环，不证明 C++ 测试服务已成为硬件：

- FU 算术仍在测试端计算；没有新增可用的 FP16 运算 RTL。
- SPM 和 loopback 服务仍在测试端维护；没有实现完整多 PE 网络及全阵列准入。
- 当前测试实例为单 PE、4 槽、16 向量 RF、32 指令字，分别编译 SIMD4 和 SIMD32；其他几何不能静默借用这两个实例。
- 本地程序/描述符装载周期单列为 `configuration_cycles`；逐周期对照的是含 admit 的组件执行区间，不能把装载开销删除后称为系统总周期。

## 前端首版证据

[`component-report.json`](../artifacts/tagged/rtl-control/component-report.json)记录了源码、运行程序、编译二进制及测试产物摘要。38 项回归通过，覆盖：

- SIMD4/SIMD32、2/4 个活动上下文、存储反压与串行/重叠策略。
- 40 个逻辑块复用两个槽、16-bit 大 ID、257 次迭代、32 字完整模板达到 PC31 后重放。
- loopback 依赖事件、共享写回仲裁、compute II 限制、旧 epoch/iteration/PC 和错误完成拒绝。
- RF 全 lane 数据保持、MLIR 生成的三源 FMA/后续指令与静态临时寄存器复用。

一个两上下文、四次迭代的受控用例，在相同资源和 24 条动态指令下，组件执行区间由串行 117 周期变为重叠 77 周期，观测到 29 个重叠 PE-cycle；两边本地装载均为 9 周期。以上使用 C++ 测试服务时序，不是完整 RTL 加速比或论文性能点。

Verilator 严格 lint 通过。Yosys 的 generic RTLIL 检查无 latch，SIMD32 前端保留一个 `16 × 512-bit`、三读一写 RF，逻辑容量为 8192 bit；其余顺序状态为 2550 bit（包含 2048-bit 指令存储），合计 10742 bit。RF 仍是 generic memory 表示，并未映射成特定 SRAM 宏或标准单元。这些数值仅检查存储及端口是否符合合约，不是面积、时序或功耗结果。

```bash
.venv/bin/python -m scripts.run_mlx_tagged_control \
  --regression --structural --output artifacts/tagged/rtl-control
.venv/bin/python -m pytest -q tests/test_tagged_rtl_control.py
```

每次回归使用新的工作目录，保留逐用例程序、机器字、日志和比较结果。该报告明确标为 `rtl_pe_frontend_component_not_M4_acceptance`，不能作为整体完成证明。

## 已完成的 FU 增量

`mlx_tagged_fp16_lane.sv`、`mlx_tagged_exp.sv` 和 `mlx_tagged_fu.sv` 实现了真实整数/位运算构成的 FP16 ADD/MUL/非融合 FMA/MAX/EXP/DIV，以及向量 shuffle 和受限 lane 规则，不调用 DPI 数学或返回工作负载查表结果。EXP 使用 Q40 范围缩减、64 项通用数学常量及四阶多项式；常量表有独立 Decimal 定义检查。

[`fu-report.json`](../artifacts/tagged/rtl-fu/fu-report.json)记录 65,536 个 EXP 编码穷举、其他运算各约 131 万个边界/随机检查，均与原生 C++ 合约一致；SIMD4/32 各 3,594 个握手事务检查了延迟、II、数据/身份反压保持和 reset。标量算术已展开为基本逻辑门（72,481 个层次化 generic cells），不是 PPA 或时钟频率证明。

FU 目前在发射时锁存组合计算结果，再保持至声明的服务延迟，只有一个操作在途；并未宣称这些组合路径已达到论文 1 GHz。前端联调 runner 默认连接实际 RTL FU，`--cpp-fu-service` 仅保留显式旧参考。SPM、loopback 及测试端仲裁仍由 C++ 提供。为与独立组件的执行域对齐，测试中的 FU 在本地装载期间停钟，装载周期另列，不能由此推导完整系统总周期。

上面的 38 项前端报告保留其历史范围；新增 FU 独立证据及联调代码不改变“完整阵列/系统/PPA 尚未完成”的状态。

## 下一步（服从模型级门槛）

先完成完整模型的算子/精度编译路径和 C++/系统正确性验证；这尤其要求处理实际模型中的 FP32、索引、广播、归约和 cache，而不只是既有 FP16 向量指令。满足模型级门槛后，再继续全镜像 RTL 装载、全阵列窗口准入、真实 SPM 和有界 NoC，将组件接成完整 RTL 后端，完成真实 ELF 对照及 PPA。不能删除操作、放宽正确性条件或增加未记账资源来取得表面通过。
