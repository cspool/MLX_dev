# 完整模型矩阵路径：受限微程序与明确数值合约

2026-09-07。属于[模型级 C++/系统链路](mlx-model-e2e-validation.md)的实施增量；不是旧 FP16 ISA 的静默扩展，也不是已验证的完整论文 ISA。

## 目的与当前边界

此前完整 Llama2 的 675 次 `linear`、195 次 `matmul` 虽然在独立 C++ 中执行，但计算依赖 OpenBLAS，尚未消费 MLX 算术指令。本增量为这些调用生成有界矩阵微程序，在 C++ 中逐条执行装载、类型转换、MUL、ADD 和存储，并禁止该路径调用 BLAS。

已实现**功能微程序、张量 lowering，以及矩阵 batch 内的多 PE/多上下文周期组件**。周期组件复用既有 tag 调度原则，并接入同一个张量执行器，但尚无跨算子 CDC 层折叠、一般 xfer 网络或 Chipyard 内存接口；其 C++ DMA 行为端点不是 HellaCache。模型与性能门禁保持关闭，不能将矩阵组件周期当作完整推理周期。

## 依据、推断与编码分开

| 类别 | 内容 |
|---|---|
| 论文事实 | Principle 3 说明 FP16 是最低稳定精度，超越函数单元为部分 SIMD 宽度；Power & Area 说明精简版移除了高精度流水线；端到端部分提到完整设计支持 RMSNorm/位置编码 |
| 本项目已有容量 | FP16 子集的 RF 是 16 个 64-byte 向量，指令容量为 32 words，SPM 为 128 个 64-byte 向量。这些实现参数不自动等于作者完整芯片规格 |
| 本次重建推断 | 全精度模式可将同一 64-byte RF 向量解释为 16 个 FP32 lane，提供独立 FP32 MUL/ADD、FP16↔FP32 转换；不增加 RF/SPM 字节容量 |
| 本次编译选择 | 输出 tile 为 M=2、N=16，K 分块为 64，K 内按递增顺序执行；它是可检验的 lowering 选择，不是论文规定的唯一映射 |
| 本次编码选择 | `mlx-matrix-f32-kasc-v1`、下述 opcode/字段、prologue/body/epilogue 划分均为本项目编码，不复用已验收系统镜像的 opcode 空间 |

论文来源为仓库中的[MLX 原文](../MLX%20Multi-Layer%20Execution%20for%20Structured%20LLM%20Workload%20Acceleration%20on%20Spatial%20Architectures/MLX%20Multi-Layer%20Execution%20for%20Structured%20LLM%20Workload%20Acceleration%20on%20Spatial%20Architectures.md)。论文未公开足以固定本模式的 FP32 累加/舍入细节；因此不能仅因模拟器实现了它就称为作者实现。

## 数值合约

对每个输出元素：

```text
acc = +0.0f
for k = 0 .. K-1:
    a32 = exact_conversion(input_a[k])
    b32 = exact_conversion(input_b[k])
    product = round_binary32_RNE(a32 * b32)
    acc = round_binary32_RNE(acc + product)
if bias exists:
    acc = round_binary32_RNE(acc + exact_conversion(bias))
output = round_to_output_dtype_RNE(acc)
```

明确禁止浮点重结合、MUL/ADD contraction 和 FTZ/DAZ。K=64 分块边界不舍入 accumulator 到 FP16，不重置 accumulator，不改变 K 的先后。尾部只启用实际 M/N lane；没有独立 lane PC 或 GPU 分支重汇合。

MUL 后 ADD 的两次舍入不同于 fused FMA 的一次舍入；FP16 输入乘积虽通常可在 FP32 中精确表示，也不能因此对一般 FP32 输入偷偷融合。关于显式区分融合与舍入模式的参考写法见 [PTX floating-point instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-fma)，此处只是数值合约的表达参考，不是复用 PTX/SASS 后端。

原来的 GPU/CPU 框架 logits 门槛与失败证据继续保留。新合约参考是另一个明确标注的实验，不允许替换历史结果来宣布通过。矩阵逐位一致也不等于整个模型已一致：RMSNorm、softmax、非线性及其 FP32 归约仍需分别验证。

## 微程序与受限工作集

RF 只使用 6 个向量：`r0=A broadcast`、`r1=B`、`r2=product`、`r3=bias`、`r4/r5=两行 accumulator`。寄存器带有效/类型状态，未初始化或类型不符的读取会失败。

SPM 中每个 K tile 包含 `B[64,16]`、`A[2,64]`、`bias[16]`。数据保持原 FP16/FP32 类型，逻辑字节数分别为 2,336 / 4,672，均小于 8,192；未来按 64-byte 物理向量分配时，FP16 需向上对齐为 37 个向量，FP32 为 73 个。禁止据此给所有驻留上下文各自一份完整 8 KiB SPM。

```text
初始化 r4/r5
  → 从实际全局张量拷贝本 K tile 到有界 SPM
  → 重放 body：B 装载/转换；各行 A 装载/转换 → MUL32 → ADD32
  → 下一个 K tile（保留 accumulator）
  → bias 加法（若有）→ 输出转换 → 实际存储
```

FP16 无 bias 模板共 16 words，有 bias 时 20 words；FP32 无 bias、同精度输出时 11 words。所有阶段合计不得超过 32 words。程序字由编译器生成并由 C++ 解码消费，解释器不是根据算子名称直接跳到预存的矩阵结果。修改合法 MUL 指令为 ADD 会改变结果，已有测试验证这一点。

当前编码使用 32 个低位，未来系统映像若沿用 64-bit word slot，高 32 位必须保留为零；本次尚未实现该系统编码/解码桥。

| 字段 | 位 | 规则 |
|---|---|---|
| opcode | 7:0 | 1–13，见下表 |
| dst | 11:8 | 当前帧只分配 r0–r5 |
| src0 | 15:12 | 未使用时必须为 0 |
| src1 | 19:16 | 未使用时必须为 0 |
| row | 21:20 | 0/1 表示对应输出行，2 表示共享指令；3 非法 |
| reserved | 31:22 | 必须为 0 |

| opcode | 指令 | 职责 |
|---|---|---|
| 1 | ZERO32 | 初始化 accumulator |
| 2 / 9 | LOAD_A16 / LOAD_A32 | 从 SPM 装载当前行的标量并广播到有效 lane |
| 3 / 10 | LOAD_B16 / LOAD_B32 | 从 SPM 装载当前 K 的列向量 |
| 4 / 7 | CVT16_32 / CVT32_16 | 显式类型转换；缩窄时 RNE |
| 5 / 6 | MUL32 / ADD32 | 分别执行和舍入的 FP32 向量算术 |
| 8 / 11 | STORE16 / STORE32 | 向已分配的实际输出张量存储；同一输出行不得重复存储 |
| 12 / 13 | LOAD_BIAS16 / LOAD_BIAS32 | 可选 bias 的 SPM 装载；无 bias 绑定则拒绝 |

`microcode` 模式的全局地址访问仍是功能接口：它确实读取 mmap 权重/前序张量并写入 SPM，但没有总线握手、缓存延迟或一致性完成事件。`global_read_bytes/global_write_bytes` 记录实际功能搬运量，不冒充 Chipyard DMA 计数。新增 `scheduled` 模式使用下述周期行为端点，也仍不是实际 Chipyard DMA。

## 新增周期状态层与两级调度

[matrix_schedule.cc](../simulator_ext/matrix_schedule/matrix_schedule.cc) 实现 `tick/done/result`；`Kernels::matrix` 可直接进入该状态机，数据来自同一 SSA 张量/真实权重，输出经实际写请求回到下游张量，不调用功能执行器回填结果。

1. **准入**：输出 tile 的目标 PE 固定为 `block_id % PE_count`，每周期至多准入一个块；每 PE 至多两个 6-vector RF 帧。相同矩阵 batch 共用一份不超过 32-word 的模板，SPM arena 则从**全阵列共享**的 128 个向量中分配。
2. **驻留状态**：每块分别保存槽位、epoch、PC、K 分块/迭代、有效行/lane、RF 类型/有效状态、输入填充/输出排空进度。在途指令和 DMA 带完整块/槽/epoch/phase/PC/iteration 身份，DMA 还绑定当前搬运项编号。
3. **PE 内发射**：按逻辑块顺序检查候选；遇到 FU/SPM 未就绪可检查另一个上下文，但每 PE 每周期只接受一条指令。每 PE 一个 compute 在途，全阵列一个 SPM 指令在途及一个 DMA 请求在途。
4. **数据与写回**：DMA 从绑定存储读取或写入实际 2/4-byte 元素；读返回提交到该上下文的 SPM arena。PE 装载、存储和 DMA 提交竞争共享 SPM 端口；SPM 装载返回与 FU 完成竞争每 PE 一个 RF 写口。忙/未就绪时保留原请求，不提前移动 PC。
5. **边沿**：选择依据周期开始状态，完成/准入在周期末提交，完成的数据下一周期才可发射使用。输出存储先进入 SPM，再由输出 DMA 写回；最后一个写响应到达后才退休并释放 RF/SPM，完成后的 `tick` 幂等。

每个 FP16 arena 对齐后为 37 个 SPM 向量，因此 4×4 阵列同时最多容纳三个这样的 arena（111/128），而不是 16 份独立 8 KiB。FP32 arena 为 73 个向量，当前 K=64 lowering 只能在全局 SPM 中驻留一个。这个限制会影响利用率；不能通过复制 SPM 或忽略对齐来掩盖它，后续可在保持数值顺序的前提下优化分块/共享。

周期参数全部是可检查的重建选项，不声称来自作者物理测量：默认 DMA latency=8、SPM=3、MUL=4、ADD=2、转换=2、ZERO=1，compute II=1；可注入有界 request/response/SPM/writeback ready 周期。报告明确区分 wall-cycle、PE-cycle 和 blocked-response-cycle，后者可能同周期计入多个被阻塞的响应，不能当作墙钟停顿相加。

`--matrix-backend scheduled` 选择此路径，`--schedule-options /path/to/options.json` 固定几何/时序/反压；默认关闭大规模 trace，单个矩阵 batch 默认 cycle 上限为 10,000,000。大模型的周期执行尚未完成，不能因为编译可生成程序就声称全模型已跑过这一模式；若需扩大运行界限，必须显式配置，不能外推 FLOPs 代替执行。

## 对应实现与验证

- 编译器：[model_matrix_program.py](../src/mlxsim/model_matrix_program.py) 生成模板；[model_tensor_compiler.py](../src/mlxsim/model_tensor_compiler.py) 为每个矩阵节点附加 `matrix_program`，并保留源算子 ID。
- 执行器：[matrix_program.cc](../simulator_ext/tensor_model/matrix_program.cc) 解码、检查容量/字段/类型、执行受限 RF/SPM 数据流；[kernels.cc](../simulator_ext/tensor_model/kernels.cc) 处理批广播和矩阵形状后进入微程序。
- 严格运行：传入 `--matrix-backend microcode`，原生 BLAS 参数固定为 `none`；缺少矩阵 lowering 会拒绝，不允许回退。完整运行还核对实际 MUL 有效 lane 总数等于矩阵 MAC 总量。
- 周期运行：`--matrix-backend scheduled` 也固定 `none`；逐 batch 保存源算子/forward/layer 身份与周期组件报告。非矩阵算子尚无完整目标时序，因此全局仍标记 `timing_mode=unmodeled`，不合成虚假的模型总周期。
- 独立数值参考：[model_matrix_reference.py](../src/mlxsim/model_matrix_reference.py) 使用 NumPy 显式逐 K FP32 MUL/ADD。只用于验证，从不进入 C++ 目标路径。
- 测试：[test_model_matrix_program.py](../tests/test_model_matrix_program.py) 已通过 16 项：FP16/FP32 逐位一致、M/N/K 尾块、空 K、bias、非连续批广播、指令修改影响结果、非法 opcode/ROM/SPM/寄存器读取拒绝、禁止 BLAS 回退，以及参考模式的插桩等价/调用计数。

完整功能运行 [`llama2-microcode-001`](../artifacts/tagged/model-e2e/llama2-microcode-001/comparison.json) 已完成 6,181 个源调用，870 个矩阵调用展开为 6,822 个实际 batch 窗口；执行 65,175,028,352 次有效 lane MUL 和同量 ADD，BLAS=0。计数来自实际微程序执行，不是 FLOPs 推算。生成 token 与原参考一致，但原 GPU logits 仍失败；[独立矩阵合约参考对照](../artifacts/tagged/model-e2e/llama2-microcode-001/kasc-numerical-diagnosis.json)也仍有 `58/257/11` 个 logits 超差。因此功能全模型数值验收尚未通过。首层完整 Q/K/V 的[独立数值检查](../artifacts/tagged/model-e2e/llama2-microcode-001/matrix-conformance.json)为 98,304 个元素逐位相同，仅证明所登记矩阵的数值合约。

周期/张量图集成回归位于 [test_matrix_window_scheduler.py](../tests/test_matrix_window_scheduler.py)，验证资源守恒、迟到响应归属、写回反压、同 PE 发射上限、ready 后一周期可用、实际 DMA 数据重放、混合精度输出、非法阶段访问拒绝，以及模拟器自身 token 驱动下一步矩阵。测试也暴露并补齐了 `aten.mm` 与 `_to_copy` 的前端路由；后者规范化为强制新分配的 cast，不能误消除为别名。

当前[周期组件证据](../artifacts/tagged/matrix-window-005/report.json)绑定了源码、两个 C++ 二进制、17 项测试及 12 个成功矩阵用例的数据/程序摘要。矩阵 `[5,5] × [37,5].T` 的机制对照中，两个上下文都占用 74 个共享 SPM 向量，均完成 815 个 DMA 请求、读取 1,260 bytes、写出 370 bytes；允许同 PE 重叠时为 7,479 cycles，禁止时为 8,980 cycles，trace 核对出 591 个同 PE 跨上下文重叠 PE-cycle。这只是调度机制回归，不是模型推理性能报告。重叠计数还从独立的 issue/complete/request/response 区间重算，避免由计数器自行证明计数器。

矩阵后续已支持[公共外部内存端口与物理地址绑定](mlx-shared-memory-ports.md)。其计算状态只保留Tensor布局/容量，实际值经请求/响应进入SPM；默认C++行为端点与向量共用。外部测试使用null数据指针、4GiB以上地址和真实重试，不能再由本地张量取数代替端口返回。以上旧周期证据保留历史版本，新的接口证据见公共端口报告。

ASan/UBSan 构建另外通过 K=65、反压、FP16 输入→FP32 输出三个用例，数据、完整 trace 和计数均与普通构建相同，结果保存在 `artifacts/tagged/matrix-window-asan-002/`。完整模型尚未运行这一周期后端；已有完整微程序功能运行和周期组件用例不能相互替代。

```bash
.venv/bin/python -m scripts.run_mlx_tensor_semantics \
  --inventory artifacts/tagged/model-e2e/llama2-reference-005/inventory.json \
  --matrix-backend microcode \
  --observe-operators tests/fixtures/llama-first-layer-observations.json \
  --output artifacts/tagged/model-e2e/llama2-microcode-new
```

显式数值合约参考通过 `capture_mlx_model_reference --device cpu --matrix-reference kasc` 启用。它改变参考的矩阵执行顺序，因此**不是透明插桩**；透明追踪等价检查仍分别在该数值模式内进行。补丁范围仅限矩阵数值入口，request/forward/layer/operator 连接键、缓存状态和生成逻辑保持原有追踪方案；不收集更多不必要的模型中间张量。

## 接下来必须补齐的系统职责

1. 将矩阵 batch 内已经实现的状态层扩展为跨源算子的提交/依赖管理，加入真实 CDC 层折叠、必要的 PE 间 xfer 和共享数据生命期；当前同步 `Kernels::matrix` 返回仍是算子边界。
2. 为归约、非线性、索引/布局/转换等余下算子增加有界目标 lowering，继续定位全模型数值差异，不能把矩阵通过当作全算子通过。
3. 将 C++ 行为 DMA 端点替换/接到实际系统内存请求、HellaCache 响应和 host 配置/同步；再验证完整编译产物的系统结果。现有小负载 M3 证书不自动覆盖这一新模式。
4. 只有全部要求模型、输入与系统正确性门禁通过后，才进入推理周期/吞吐验证；RTL 实现仍在这些步骤之后。
