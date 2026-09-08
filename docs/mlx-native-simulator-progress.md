# MLX 原生 C++ 调度模拟器：实施状态与运行合约

更新：2026-09-07。原方案：[编译链与调度缺口方案](mlx-gpgpu-toolchain-scheduling-gap-plan.md)。

当前阶段：M0/M1/M2 的登记范围已经通过[第一阶段验收](../artifacts/tagged/stage1-acceptance/certificate.json)，M3 C++ 模型的 Chipyard 集成已通过[第二阶段验收](../artifacts/tagged/stage2-acceptance/certificate.json)。M4 已开始，控制/共享 RF 前端完成组件回归；完整功能 RTL、系统对照及 PPA 仍未完成，见[RTL 实施状态](mlx-tagged-rtl-progress.md)。

用户随后将完整模型端到端推理设为系统正确性的验收依据。上述证书只覆盖已登记的小负载，不能支持这一更强结论。当前工作优先级转为[模型级编译与验证](mlx-model-e2e-validation.md)，完成全算子 C++/系统执行与结果对照之前，不开始模型推理性能验证，也不继续扩展 RTL/PPA。

模型级新增进展：完整公开 Llama2-7B 的 31 种 ATen overload、6,181 个节点已编译到独立 C++ 张量语义执行器，真实权重 mmap、KV 依赖和自生成 token 链均执行；20 项组件回归通过。但 logits 原门槛仍失败，且这一 OpenBLAS 功能语义执行器尚未接入 MLX 指令调度或 Chipyard 内存系统。它不能冒充模型级 MLX 通过；数值诊断与逐算子映射缺口见上述模型文档。

后续增量已完成全部矩阵节点的受限 FP32 微程序功能执行，严格模式禁用 BLAS，见[矩阵微程序说明](mlx-matrix-microcode.md)。完整尺寸首层 Q/K/V 的 98,304 个输出与独立 K 顺序参考逐位一致，但完整模型 logits 在该矩阵合约参考下仍有超差。另已将矩阵 batch 内的固定 PE 映射、多上下文准入/发射、共享 SPM 和写回反压接入同一张量执行器，组件证据位于 [`matrix-window-005`](../artifacts/tagged/matrix-window-005/report.json)。当前共 53 项回归通过；跨算子层折叠、余下算子的有界目标 lowering 与 Chipyard 集成仍未完成，不计为模型级 M3 通过。

最新浮点增量见[向量微程序说明](mlx-vector-microcode.md)：完整 Llama2-7B 的 2,622 个浮点源调用已进入有界微程序，结合矩阵路径完成了实际全模型执行。对显式、版本化数值合约，三个 logits 向量共 96,000 值逐位一致，token/cache 数据流核实；原子 libm 共享及原 GPU 差异均公开保留。当前 114 项回归通过，但其余功能语义调用、向量周期调度、跨算子层折叠与完整 Chipyard 链仍未完成，模型推理性能和 RTL 扩展仍暂停。

内存/控制继续补齐：[内存计划完整执行](../artifacts/tagged/model-e2e/llama2-memory-001/numeric-conformance.json)保持 96,000 个 logits 逐位一致；所有 6,181 个源调用现都有显式编译路径，包含新增的 30 个 RV64 控制调用。158 项回归和 191 个 Spike 叶指令对照通过，加入控制路径后的完整重跑仍在进行。详见[内存/控制实施记录](mlx-memory-control-plans.md)；仍不是完整系统或推理性能通过。

该完整重跑已经结束并通过声明数值合约：[`llama2-all-lowered-001`](../artifacts/tagged/model-e2e/llama2-all-lowered-001/numeric-conformance.json) 中全部源调用实际执行，旧功能辅助入口为0。新增[向量多上下文周期组件](mlx-vector-window-simulation.md)已接入同一入口，并能通过可替换端口在无本地Tensor数据指针时完成计算；200项相关回归通过。完整系统和跨算子执行仍未完成，不推进推理性能或RTL。

矩阵现也通过同一内存端口取数，新增region→物理地址/权限验证、跨kernel token隔离和有界重试桥；221项相关回归通过。证据见[公共内存端口](mlx-shared-memory-ports.md)。这是系统连接前的组件验证，不冒充真实HellaCache接入或完整模型系统执行。

搬运路径现新增 `--memory-backend scheduled`：C++状态机从响应取得索引/predicate与源数据，经转换后发出写请求，并等待确认；保留合法视图消除与存储所有权。269项广泛回归、76项绑定源码的相关组件回归及三组ASan/UBSan检查通过，详见[搬运状态机](mlx-memory-control-plans.md)。新版本的[完整公开Llama2重跑](../artifacts/tagged/model-e2e/llama2-memory-scheduled-001/numeric-conformance.json)已结束：6181个调用全部执行、96000个logits在声明数值合约下逐位一致；32446193次memory请求/响应排空，BLAS和未编译功能回退均为0。原GPU门槛仍失败，系统与完整周期门槛未通过，不能据此开放推理性能。

## 实施方向

最新镜像准备增量：[完整系统镜像/profile](mlx-complete-system-image.md)已生成全部6181源任务、初始资产及CPU输出检查器，完整编译/参考绑定和共同RAM初始化审计通过。真实Rocket小图也验证了显式4×4参数与驻留资产模式，184项相关回归通过。完整镜像尚未执行，不将准备或初始化算作模型系统通过。

最新大地址空间增量：[16GiB系统配置](mlx-wide-system-memory.md)已通过实际高地址的正常/扰动图，受检ELF初始化与正常CPU/DMA执行分开计数。新内存类也完成完整360资产/13476831558 bytes的初始化摘要核对，但没有执行模型。170项相关回归通过；完整系统镜像和全模型执行仍未完成。

最新实际系统增量：[RoCC连接](mlx-clocked-rocc-integration.md)已经通过两张正常/扰动45源节点图及三个阻塞WAIT数据链。真实Rocket执行CPU控制，C++后端按系统边沿运行，数据经DCache接口而非插件私有数组；159项相关回归通过。修正并保留了窄store展开导致最终数值失败的证据。该系统仅配置256MiB外存，尚未执行完整模型，不放行推理性能或RTL。

2026-09-08外部时钟增量：[C++设备控制器](mlx-clocked-device-controller.md)独立接收系统边沿和总线响应，描述符与张量均走注册请求队列，状态查询不推进时钟；15项组件检查及154项相关回归通过，ASan/UBSan对照一致。它尚未连接实际Rocket/SimpleHellaCacheIF，也未替换任何活动模型运行；两个完整尝试的冻结实现保持不变。

2026-09-08独立主机桥增量：通用C/RV64调度器已执行45节点三步生成图；插件内存改用按需4KiB页存储，16GiB逻辑窗口的低/跨4GiB数据偏移均实际执行，29个历史联合用例与dense表示的数值、事件和计数一致。证据与明确边界见[稀疏映射测试内存](mlx-sparse-mapped-memory.md)。这不修改在跑完整模型的冻结后端，也未完成大模型CPU装载或真实系统时钟。

最新共享原生物理执行见[四类后端集成](mlx-shared-physical-model.md)。控制scheduled选择已进入主编译与C++入口；Arena/pin、全局请求身份、同一物理背板和写入有效位已连接四类后端，524项相关回归通过。它仍预装载资产、跨算子串行，尚无新路径完整模型或真实Rocket/HellaCache证据，不将组合图通过提升为系统模型通过。

完整模型重跑期间，另新增独立[控制访存/单步组件](mlx-controller-window-simulation.md)和[物理缓冲所有权](mlx-model-physical-storage.md)，开发时未修改冻结来源。独立阶段证据分别是241项控制回归与15项分配器测试/完整图地址重放；后续的主入口和共享物理集成是新的证据范围，见上段，不改写原运行身份。

用户在实施过程中明确要求优先用 C/C++ 模拟器实现调度机制，而不是修改电路。早期提前进行的 RTL 及 RoCC、Chipyard Scala 配置、安装脚本、主机 runtime 修改已经撤回。调度执行主线是原生 C++；Python 用于输入编译、独立参考和测试。

固定实施顺序为：**C++ 模拟器实现与验证 → 系统模拟集成 → RTL 实现与对照**。编译链在第一阶段共同完善。系统集成和 RTL 仍属于后续目标，但只有前一阶段通过才进入下一阶段；不能在 C++ 调度模型仍有未解决问题时提前推进电路。

这里的硬件调度指“在 C++ 状态模型中模拟硬件调度行为”：维护计算单元、驻留上下文、资源容量、执行流水线和网络状态。系统阶段新增的 Scala 接口与 SV/DPI 包装仅把 Chipyard 信号接到该模型，不包含新的 PE/FU/调度器功能 RTL。真实 Chipyard ELF 结果见下文；RTL 对齐及电路成本仍未验证。

系统阶段遵循[ISA 设计依据](mlx-native-isa.md)：指令集由论文硬件及约束支持的执行模式推导，区分明确语义、重建推测和项目编码。先确定架构契约，再在 C++ 中验证，并不意味着提前实现电路。

## 已实现的结构

| 部分 | 入口 | 职责 |
|---|---|---|
| C++ 程序合约 | `simulator_ext/tagged/program.cc` | 独立解析与验证 ABI、资源、数据依赖、模板容量、SPM 访问及波次 |
| C++ 周期模拟核心 | `simulator_ext/tagged/simulator.cc` | 块组准入、每 PE 多上下文选择与发射、独立在途操作、有限 SPM/路由资源、完成回收 |
| C++ 数值执行 | `simulator_ext/tagged/fp16.cc` | FP16 转换、逐步舍入算术、SIMD shuffle 与受限 lane 的 EXP/DIV |
| C++ 可执行入口 | `simulator_ext/tagged/main.cc` | 直接加载程序 JSON、运行模拟并导出计数/trace；可关闭 trace 并在进程内重复运行 |
| C/C++ 逐周期 API | `mlx_tagged_simulator.h`、`c_api.h/.cc` | 模型拥有独立程序/状态；逐周期 tick、无副作用快照、完成查询、错误边界和句柄生命周期 |
| 原生 MLIR 前端 | `simulator_ext/tagged/mlir_frontend.cc` | LLVM/MLIR 14 的已注册 `mlx.input/compute/output` 操作、类型校验、CSE 与 canonicalize |
| 空间与块编译 | `src/mlxsim/tagged_compiler.py`、`tagged_regions.py`、`tagged_mlir.py` | 保留源层/region，融合多条算术，分配临时寄存器生命期、PE/SPM，生成传输与层边界搬运 |
| Python 检查参考 | `src/mlxsim/tagged_simulator.py` | 无时序 DAG 参考和可读的周期参考，用于核对 C++，不作为性能执行主线 |
| 运行与审计脚本 | `scripts/run_mlx_tagged.py` | 构建/调用 C++ 二进制，检查输出，保存源码和程序摘要 |
| 第一阶段验收 | `scripts/verify_mlx_native_stage1.py` | 执行行为回归、数值测试、MLIR 负载与对照，将证据绑定到前后不变的源码摘要 |

源图 v2 的 `region` 与 `layer` 标明计算语义边界，不包含物理 PE 或寄存器选择。一个完整 BSMM 蝶形的四条算术现在处于同一 tagged block，FFT 也以复数蝶形为块；内部临时值通过静态生命期复用寄存器，不再把每条 SSA 算术当成一个上下文。源图 v1 的单算子块仍用作历史反压回归与编译对照。

MLIR CSE 允许合并同一 region/layer 内的相同纯运算，不能跨调度块合并。输出别名被保留，FP16 非融合 FMA 等数值语义不变。当前是有限向量源语言和 MLX dialect 的编译前端，不宣称支持任意 C/PyTorch 或通用 Linalg 到 MLX 的自动转换。

### 两级调度的具体语义

编译器固定块到 PE 的空间映射。程序分成有界波次（wave），每个波次的块在容量检查后共同驻留，避免生产者等待一个因资源不足而无法进入的消费者。波次内按真实数据到达推进，不等待整层结束。波次排空后才复用物理上下文槽；这是当前明确采用的有界执行策略，不宣称已支持任意动态迁移或跨波次重叠。

PE 内按 `(logical_layer_id, block_id)` 排序候选，跳过未就绪或目标资源不可用的上下文。每个块持有独立 PC、迭代号、数据有效位和 inflight 状态；每 PE 每周期最多发射一条指令。不同块可同时占据 compute、load/store 或 xfer 等不同执行阶段。

当前每 PE 最多 4 个上下文、总计 16 个向量寄存器、32 个去重后的指令字；全阵列 SPM 为 128 个向量，默认 SIMD32、4×4 PE。模型对资源不足明确拒绝，不自动增加容量。一个 compute 操作在途、全阵列一个存储请求在途、每个路由器一个向量缓冲；latency、compute II 和反压时序分别定义。

所有候选、完成和路由动作根据周期开始的状态决定，再统一提交。当前周期完成的数据在下一周期可用于发射。每 PE 每周期最多一个 RF 写回，存储返回、compute 返回和网络交付发生竞争时按固定顺序处理。

### 身份与数据生命期

逻辑块 ID、层号、PE、物理上下文槽、wave/epoch、iteration 和 PC 分别保存。xfer 指向逻辑块及其局部输入寄存器；路由中保留生产者指令身份和迭代号。只有目标上下文处在匹配迭代且输入寄存器未被占用时才交付；目的数据可见后完成发送者的 xfer。

输入寄存器在本次迭代内保持有效，迭代结束后清除有效位。生产者不能覆盖尚未被消费者推进下一迭代的数据。波次退休要求所有块、在途操作和路由包排空；寄存器数据已经复制到消费者后，生产者资源才可回收。

多迭代 BSMM 验证曾暴露两类网络阻塞：未来迭代的包过早注入，以及满缓冲之间不能同时交换。现已增加目的输入的注入 credit 检查，并在每周期路由仲裁中允许同一注册缓冲同时出队/入队；依赖消除到不动点后，保留可以同时完成的交换环。每路由器仍只有一个向量缓冲，未通过增加无限缓冲绕过问题。该行为由 C++ 与参考模型的逐事件一致性及多迭代回归共同检查。

当前支持同波次、相同循环次数的无环跨块数据流。跨波次值通过显式 SPM spill/reload 保存。SSA 编译入口当前接收固定的 SIMD 向量输入，循环重复执行同一组输入；需要每迭代变化数据时，低层程序可使用经过容量验证的 SPM stride。更通用的算子前端和自动循环数据布局仍需扩展。

## 直接运行 C++ 模拟器

```bash
cmake -S simulator_ext/tagged -B build/mlx-tagged-cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build/mlx-tagged-cpp -j4
build/mlx-tagged-cpp/mlx-tagged-sim \
  --program /path/to/program.json --output /path/to/native.json
```

构建依赖 C++17 编译器、CMake、pkg-config 和 jsoncpp。`mlx_tagged` 是可链接的 C++ 静态库；`mlx-tagged-sim` 是独立可执行文件，无需 Python 执行周期模型。

默认编译入口还需要 LLVM/MLIR 14 开发包；本环境安装了 `mlir-14-tools`、`libmlir-14-dev`、`llvm-14-dev`。`mlx-mlir-front` 完成原生解析、校验和优化，`libmlx_tagged_c.so` 提供 C ABI。可以使用 `-DMLX_ENABLE_MLIR=OFF` 仅构建已编译程序的执行核心，但这不替代包含编译链的第一阶段验收。

常用参数：`--serial` 禁用同 PE 多块重叠；`--compute-ii N`、`--load-latency N` 设置服务时序；`--memory-period N`、`--link-period N`、`--receive-period N` 注入有界反压；`--no-trace --repeat N` 用于减少输出开销并重复测量原生运行时间。

已有程序格式可由 `Program.to_dict()` 导出。编译图输入与运行 C++ 可以用以下辅助脚本组织：

```bash
.venv/bin/python -m scripts.run_mlx_tagged \
  --graph /path/to/vector_graph.yaml --output artifacts/tagged/my-run
```

已经提供四个可重放的源算子样例：

```bash
.venv/bin/python -m scripts.run_mlx_tagged \
  --workload fft_cmp --output artifacts/tagged/stage1/fft_cmp
.venv/bin/python -m scripts.benchmark_mlx_tagged \
  --output artifacts/tagged/native-throughput
.venv/bin/python -m pytest -q tests/test_tagged_scheduler.py
ctest --test-dir build/mlx-tagged-cpp --output-on-failure
.venv/bin/python -m scripts.verify_mlx_native_stage1
```

脚本对源图/登记负载默认使用 MLIR 前端，然后调用 C++ 分别运行重叠与串行策略，并用无时序及源算子参考核对输出；结果分类为 `native_cpp_component_cycle_simulation`。`--frontend python` 可运行优化前的空间编译对照，`--mlir source.mlir` 可直接输入已登记 dialect 的 MLIR。原始/优化后 IR、程序、运行日志和 JSON 结果保存在指定目录。

## 验证与尚未完成的工作

回归入口为 `tests/test_tagged_scheduler.py`。原生测试对照 C++ 与 Python 参考的输出、逐事件 trace、周期和资源计数，覆盖局部多上下文、等待隐藏、循环、反压、40 个逻辑块复用少量槽位、独立的 C++ ABI/容量拒绝及自动编译。

截至本次记录，56 项回归通过；新增覆盖同层融合、内部临时寄存器复用、源 region 收缩成环的拒绝、逐周期 C API、不同实例隔离、MLIR 类型/CSE/输出别名及资源受限波次间的 spill/reload。原生 CTest 检查全部有限 FP16 编码的往返、舍入边界、非融合 FMA 和 MAX 的带符号零语义。AddressSanitizer/UndefinedBehaviorSanitizer 构建通过该数值测试和四个融合后的源算子样例，未报告错误。

默认 4×4、SIMD32、两次迭代的小样例记录如下。这些是原生组件模型的周期，不是 Chipyard 测量，也不是论文尺寸的性能点。

| 样例 | 同层 tagged block 数 | 动态指令数 | 原生模型周期 | 同 PE 跨块重叠 PE-cycle |
|---|---:|---:|---:|---:|
| BSMM | 4 | 80 | 165 | 0 |
| FFT-CMP | 4 | 136 | 228 | 0 |
| SWA | 4 | 50 | 183 | 0 |
| 组合 block | 13 | 212 | 408 | 0 |

新证据由 `artifacts/tagged/stage1-acceptance/` 保存，旧单算子块证据仍在 `artifacts/tagged/stage1/`，两类数据不混用。默认 16 PE 大于这些小样例的块数，均未发生同 PE 多层折叠，因此串行/重叠周期相同；T12 另外把真实 BSMM 的两层各两个蝶形块折叠到两个 PE，每 PE 两个上下文、共 14 个向量寄存器，四次迭代时从串行 380 周期降到重叠 367 周期，观测到 39 个重叠 PE-cycle，计算和访存工作量相同。

SWA 样例是 SIMD 前四分之一 lane 上的单特征 attention，其余输入/可见输出为零；组合 block 明确为 Fourier mixing → butterfly projection → SWA → residual，不冒充完整 Transformer 模型。

[融合后的运行速度记录](../artifacts/tagged/native-throughput-fused/benchmark.json)对同一 408-cycle 程序关闭 trace，C++ 重复 1000 次、Python 参考重复 10 次。本机记录约为 C++ 0.356 ms/次、Python 51.8 ms/次，约 145.7 倍运行时间差。C++ 计时包含程序验证和实例构造，Python 只计 run；两端模拟周期、资源计数和输出相同。这个数值衡量模拟器自身运行速度，不是 MLX 硬件加速比，也不作为跨机器性能保证。

## M3：C++ 行为模型接入 Chipyard

`system_sim/native/device.cc` 独立解码二进制镜像、描述符和 PE 机器字，检查规范编码、静态路由、资源与数据流。`config/launch/wait/status` 通过 `system_sim/chipyard/MLXNativeRoCC.scala` 和 `system_sim/native/MLXNativeRoCC.sv` 的 DPI 桥进入 C++。加速器调度与算术仍由第一阶段同一个 `mlx::tagged::Simulator` 逐周期执行，不调用 Python 周期参考，也不使用旧串行 SV cycle model。

当前系统协议是输入 DMA → kernel → 输出 DMA，64-bit 请求、一个 outstanding；缓存请求的接受与返回驱动实际数据装载和完成。主机缓冲区按 SPM 槽位升序紧密打包，输出通过真实写请求返回主机内存。原生 `system_cycles = dma_cycles + kernel_cycles`，不沿用旧控制器的 `+2`；host 的 config 和 launch/wait 是 Rocket `rdcycle` 测得的区间，不能与上述 busy 周期混加为同一种统计。

三层检查入口如下，不能互相替代：

- `tests/test_mlx_native_device.py` 与 `tests/test_mlx_native_system_gate.py`：共 20 项通过，覆盖四个负载的二进制装载与 DMA、输入内存改变、存储/响应反压、非法镜像拒绝、直接 C++ 与 DPI 一致性、连续启动/恢复及验收器对损坏证据的拒绝。这是协议和验收回归，不是 CPU 执行。
- `scripts/run_mlx_native_chipyard.py`：MLIR 编译、机器镜像封装进 bare-metal ELF，再由真实 Rocket 发出 RoCC 命令。逐项核对 host 输出与完成状态、DMA 字节/请求守恒，以及系统内 kernel 与独立 C++ 的事件、周期、资源计数和输出。
- RTL 对照：尚未开始，不能用 DPI 包装的存在作为调度电路已实现的证明。

独立 checkout 位于 `build/chipyard-native`，固定 Chipyard commit `b5d013190d637e634113cb5179f8c8885df1945a`。本次使用 OpenJDK 11、SBT 1.4.9、RISC-V bare-metal GCC 10.2 和 Verilator，已补齐本次 Rocket 构建所需子模块与 FESVR。安装脚本只加入独立的 native-model 桥，并记录兼容补丁；`native-model-finalize.patch` 确保退出时调用 DPI final 导出报告。

[本次验收的真实 ELF 执行记录](../artifacts/tagged/stage2-acceptance/run-mtr8kiaw/workloads/results.json)如下，单位均为对应定义的模拟周期。旧 `chipyard-native/` 中的首批记录保留为历史证据，不与本次 ELF 的 host 开销混用：

| 负载 | host config | host launch/wait | 系统 busy | DMA | kernel | 实际 DMA 字节 |
|---|---:|---:|---:|---:|---:|---:|
| BSMM | 703 | 560 | 549 | 384 | 165 | 704 |
| FFT-CMP | 990 | 879 | 868 | 640 | 228 | 1216 |
| SWA | 465 | 479 | 447 | 264 | 183 | 448 |
| 组合 block | 1535 | 827 | 793 | 385 | 408 | 640 |

全部输出正确，kernel 的逐事件 trace、周期、计数与独立原生模型一致。此处系统周期由运行中的 C++ 设备采样并由 ELF 查询，不是把独立 kernel 周期代填为系统测量。负载规模与第一阶段相同，不能外推到论文尺寸或当作 RTL 测量。

[两 PE 折叠 BSMM](../artifacts/tagged/stage2-acceptance/run-mtr8kiaw/folded/results.json)与[串行对照](../artifacts/tagged/stage2-acceptance/run-mtr8kiaw/serial/results.json)也已通过真实 ELF：四次迭代、相同程序摘要和资源，均执行 160 条动态指令并传输 704 字节。启用多块重叠后，kernel 从 380 降至 367 周期，系统 busy 从 764 降至 751 周期，DMA 均为 384 周期；host launch/wait 为 775/762 周期。39 个重叠 PE-cycle 对应第一阶段的同一调度行为，不是扩大资源或减少工作量产生的收益。

真实 ELF 的三类拒绝测试已通过：[不兼容镜像版本](../artifacts/tagged/stage2-acceptance/run-mtr8kiaw/reject-magic/results.json)、[错误路由](../artifacts/tagged/stage2-acceptance/run-mtr8kiaw/reject-route/results.json)、[非零保留位](../artifacts/tagged/stage2-acceptance/run-mtr8kiaw/reject-reserved/results.json)。均在执行前返回可查询的错误，DMA 请求和 kernel 周期为零，主机输出哨兵未改变。它们验证的是当前项目 ISA/ABI 的非法输入处理，不证明论文未公开的原始编码。

可重放命令（需已准备固定版本的工具链与 checkout）：

```bash
.venv/bin/python -m pytest -q tests/test_mlx_native_device.py
.venv/bin/python -m scripts.run_mlx_native_chipyard --phase all
.venv/bin/python -m scripts.run_mlx_native_chipyard --phase run \
  --workloads bsmm_folded --output artifacts/tagged/chipyard-native-folded
.venv/bin/python -m scripts.run_mlx_native_chipyard --phase run --serial \
  --workloads bsmm_folded --output artifacts/tagged/chipyard-native-folded-serial
.venv/bin/python -m scripts.run_mlx_native_chipyard --phase run --reject magic \
  --workloads bsmm --output artifacts/tagged/chipyard-native-reject-magic
```

`--reject route` 与 `--reject reserved` 分别测试非法路由和保留位。负例的 ELF 必须确认错误可见、输出哨兵未改变，runner 还要求 DMA 请求及 kernel 周期为零；不能以超时或崩溃充当拒绝成功。

系统 ABI、命令接受条件、busy/complete 的区别、计数有效性及错误排空规则见[系统接口合约](mlx-native-system-contract.md)。连续启动回归发现并修复了一个统计污染问题：前次成功之后的非法 launch 仍暴露旧周期/字节计数。当前每次启动尝试在解码前清空本轮执行计数和结果快照；非法地址、错误 tag 后排空恢复与 reset 也有独立回归。

完整 M3 验收入口为 `.venv/bin/python -m scripts.verify_mlx_native_stage2`。它以新目录重新运行全部证据，核对编译进实际设备的构建身份、二进制/源码/链接库摘要，独立重放每笔 DMA 并对照系统与独立 kernel。`--skip-build` 只省去构建动作，不跳过构建身份校验。构建脚本通过身份头文件补齐 C++ 库变化到 Chipyard 重建的依赖，避免误用旧二进制。

尚待完成并验证：

- M3 已形成独立系统阶段证书，包含 20 项回归和 9 次真实 ELF 执行；后续修改所绑定源码、库或构建产物后必须重新验证。四个默认负载全部空间展开，多上下文收益的证据来自单独的两 PE 折叠用例。
- 继续按 ISA profile 核验未覆盖模式与推测边界；不把可解码或功能正确等同于完整论文架构复现。
- RTL阶段恢复后，按[RTL 交接要求](mlx-native-to-rtl-handoff.md)继续实现 FU、SPM/NoC 和阵列准入，再做完整 RTL 系统对照与物理成本评估。按用户追加门槛，当前先补完整模型的C++/系统正确性，不执行这项后置RTL工作。
- 任意 C/PyTorch、一般 Linalg 自动映射、更大规模或更多调度策略属于后续扩展。现有通过的登记范围不能外推成所有合法图和映射的无死锁证明。

本文件报告阶段事实，不替代原方案的完整验收，也不将测试通过数视为整项目完成证明。
