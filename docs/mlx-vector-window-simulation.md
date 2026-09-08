# 向量周期状态与可替换内存端口

2026-09-07。继[向量微程序](mlx-vector-microcode.md)后，把功能执行推进到有界、多上下文的周期组件。它仍不是完整 Chipyard 模型，也不提供未经系统验证的推理性能结论。

## 已实现的状态机

[vector_schedule.cc](../simulator_ext/vector_schedule/vector_schedule.cc) 消费原有 `mlx-vector-fp32-v1` ROM/阶段描述，而不是先调用功能模型算出整张量再附加周期。

- 逐元素算子以最多 16 个输出元素为一个块；mean/softmax 以一行归约为一个块，跨分块保留部分和、max、sum 和程序阶段。
- 块固定映射到 `block_id % PE_count`；每 PE 至多两个 8-vector RF 帧，每个上下文分配 5 个 SPM 向量。全阵列共享 128 个向量，因此最多 25 个 arena，不为每个 PE 复制 8 KiB。
- 每 PE 每周期至多一条发射。常规向量 FU 与四分之一宽度 SFU 分别保持一个在途操作，并共享一个 RF 写回口；SPM 全局至多一个在途指令，DMA 全局至多一个在途请求。
- 所有操作保留 block/slot/epoch、phase/PC、base、归约层级和搬运位置。结果及 lane mask 在完成前保持，不在发射时提前写 RF 或推进依赖。
- 选择基于周期开始状态；完成/准入在末尾提交，下一周期才可使用新数据。输出先存 SPM，再经过 DMA 写响应排空后退休，`done` 后 tick 幂等。

装载前先执行操作数填充协议。常量/虚拟 padding 通过受限 SPM 初始化写入，张量值则只能从内存响应进入 SPM；全部所需元素就绪后才能发射 PE 的 LOAD。归约的部分和读写同样竞争 SPM，softmax 的三个流式遍历都实际发生。

默认常规延迟沿用矩阵重建参数；SFU 的 EXP/COS/SIN=8、DIV/SQRT=12，均是显式可配置的模型参数，不冒充论文或芯片测量。支持独立 DMA 请求/响应、SPM、RF 写回反压与两个 FU 的 II。报告分别标注 wall-cycle、PE-cycle 和 blocked-response-cycle，不能混加为总时延。

## 内存端口是数据的唯一来源

[model_io/memory_port.h](../simulator_ext/model_io/memory_port.h) 定义公共接受/返回合约：

```text
request_ready → submit(id, region, byte_offset, bytes, read/write, data)
response(id, data, error) → consume_response
```

`submit` 表示请求已经被端口接受。系统适配器应使用有界请求缓冲，将下游 valid 与下游 ready 解耦，不能等到看见 ready 才凭空生成/丢弃请求。响应一旦出现，必须在被消费前保持 id/data/error 稳定。错 token、无请求响应、错误响应、回压时数据变化或撤回都会被拒绝。

向量组件已有两种端口：

1. 内置 C++ 行为端口，按显式延迟读取/写入绑定存储，用于组件回归。
2. 外部端口，由调用方绑定 region 与真正的内存服务。输入和输出 Tensor 可以只有 shape/stride/容量描述、**没有本地数据指针**；计算值只能来自外部响应，结果只能通过写请求到达外部存储。

[外部内存测试](../tests/vector_external_memory.cc) 正是使用无本地 backing 的虚拟 Tensor。外部端口提供 `[1,4,9,16]` 后，RSQRT 得到 `[1,0.5,1/3,0.25]`；测试还故意注入错误 token、错误响应、无请求响应以及响应变化/撤回。这个证据排除了把本地预存输入或结果当成系统响应的路径，但尚未证明 HellaCache/TileLink 实际接入。

## 已接入同一张量入口

`--vector-backend scheduled` 在 `Kernels::vector` 内直接运行周期组件；`--vector-schedule-options /path/to/options.json` 固定参数。编译器保留源算子/forward/layer 身份，执行报告逐算子输出 `vector_windows`。矩阵 `scheduled`、向量 `scheduled`、内存 `planned` 和控制 `rv64_leaf` 已能在同一生成/cache 组合图中共同运行，结果与参考一致且旧功能辅助入口为零。

尚缺内存计划和控制绑定层的完整时序、跨算子同时驻留与实际 host/系统接口，所以整张量程序仍标记 `timing_mode=unmodeled`。不能把 `matrix_windows` 与 `vector_windows` 相加就宣布完整推理时延，也不能把局部并行测试代替跨层 CDC 重叠验证。

## 验证证据

[源码绑定的组件报告](../artifacts/tagged/vector-window-002/report.json) 保存 33 个成功向量周期用例、独立内存端口用例及对应程序/结果摘要；41 项向量周期测试和25项内存/组合图测试共同通过。全体相关回归为 [200 项](../artifacts/tagged/model-e2e/vector-window-tests-002.xml)。

同一 SiLU 输入、两个上下文和10个共享 SPM 向量的机制对照中，启用/禁止同 PE 重叠分别为2406/3001 cycles，微指令数与读写量相同；trace 观测到292个跨上下文重叠 PE-cycle和41个常规向量/SFU重叠 PE-cycle。这里仅验证调度机制，不是模型推理性能报告。

4×4阵列用例的驻留上限为25个上下文，实际占用125/128个SPM向量，准入在容量不足时停顿。归约测试涵盖1/3/17/65/4096宽度、FP16/FP32、尾部及独立反压；数值和微指令计数与功能执行器逐项一致。ASan/UBSan 另外通过归约65、向量/SFU重叠和外部虚拟内存三例，结果在 `artifacts/tagged/vector-window-asan-001/`。

## 完整模型与下一步

[`llama2-all-lowered-001`](../artifacts/tagged/model-e2e/llama2-all-lowered-001/numeric-conformance.json) 已实际完成全部6181个源调用：870 matrix +2622 vector +2659 memory +30 control，`functional_entry_calls=0`、BLAS=0。三个 logits 共96000个值在声明数值合约下逐位一致，参数、token与cache链核实；原GPU误差记录保持原状，libm共享范围仍公开。

这份完整模型证据是功能/数值执行，不是完整周期执行。下一步必须把公共内存端口用于矩阵、搬运与控制路径，建立统一地址绑定及实际系统响应，再实现跨算子依赖/驻留和完整模型系统对照。其他要求模型、输入集合以及原硬件精度/时序依据也仍未完成。推理性能和RTL扩展继续受门禁约束。

后续已经完成矩阵对公共端口的接入，并增加物理 region 权限/地址检查、跨kernel请求身份和有界重试队列，见[公共内存端口增量](mlx-shared-memory-ports.md)。矩阵与向量的无本地数据指针测试通过，但实际 HellaCache 连接及其余后端的接口迁移仍待完成。
