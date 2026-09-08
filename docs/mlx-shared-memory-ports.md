# 矩阵/向量/搬运公共内存端口与物理地址绑定

2026-09-07。承接[向量周期组件](mlx-vector-window-simulation.md)，将公共内存合约扩展到矩阵，并补齐系统对接前必须具备的地址与传输约束。未修改 RTL；本页不声称真实 Chipyard 缓存已经接入。

## 已接入的计算与搬运后端

矩阵和向量周期执行器均通过 [MemoryPort](../simulator_ext/model_io/memory_port.h) 接受数据，矩阵不再在计算引擎中直接读取 `Tensor.storage->data` 或直接写回目标张量。矩阵的 region 为 A/B/bias/output，向量为 operand0/operand1/output。

不传外部端口时，三者共用 [TensorMemoryPort](../simulator_ext/model_io/tensor_memory_port.cc)。它是有一个在途请求的 C++ 行为内存端点：到达配置延迟时采样实际输入或提交写入，随后保持响应直到消费者接受。写入提交和发起方收到确认是两个时刻；上下文仍要等写响应才能退休，不能以提交代替完成。

新增的 `--memory-backend scheduled` 将copy/cast、cache拼接、embedding和where搬运也接入公共端口；region为按名称排序的输入，随后是输出。它从响应取得实际索引/predicate，再发数据请求，不直接访问Tensor数值。视图仍是检查过的存储别名，不产生虚假DMA。详见[搬运状态机与证据](mlx-memory-control-plans.md#响应驱动的-c-搬运状态机)。

传外部端口时，Tensor 可以只有 dtype/shape/stride/容量，data/writable 均为 null。矩阵测试覆盖 FP16/FP32、bias、K=0/5/65、尾部、batch 偏移、非连续 B 布局；计算只能用响应数据，结果只能到外部存储。测试中的物理地址均高于 4 GiB，排除地址被静默截断为 32 bit 的实现。

## 逻辑区域到物理请求

[AddressSpacePort](../simulator_ext/model_io/address_space_port.cc) 将 `(region, byte_offset, size)` 转为物理地址，并在发出请求前检查：

- region 有效、读写权限正确；只接受已登记的1/2/4/8-byte自然对齐访问。
- offset/size 不越过区域，base+size 和 base+offset 不发生64-bit溢出。
- 只读区域可以显式别名；任何可写区域与其他非空区域重叠则拒绝，避免意外覆盖输入或权重。
- 本地请求ID被转换为共享 `RequestTokens` 分配的物理ID；即便新kernel复用本地ID，也不会接受前一个kernel的迟到响应。
- 响应id/data/error在回压期间必须稳定；无匹配请求的响应、错误身份或撤回/变化都会被拒绝。

每个适配器只保存一个在途映射，不使用无界事务表。`cycle_origin` 可将组件相对周期映射到传输的绝对周期；实际系统仍需由统一时钟驱动，不能让多个独立局部时钟自行代表同一缓存。

## 有界传输队列与重试

[QueuedPhysicalPort](../simulator_ext/model_io/queued_physical_port.cc) 是无数据计算、无内存读写的注册式传输桥：

```text
提交到有界队列 → 下一周期呈现请求 → 下游接受
  → nack：保持同一请求身份，下一周期重试
  → response：保持结果，直到模型消费
```

它不会因为下游未 ready 而丢请求，也不会在收到 nack 后让模型再次提交同一逻辑请求。响应必须对应已被下游接受的请求；尚未呈现或同接受周期的响应被拒绝。忙时 reset 明确拒绝，必须先排空，不能清空表后误接旧响应。

这是通用 cache-like 协议队列，不是 HellaCache/TileLink 实现。真实系统仍需绑定请求尺寸、数据对齐/掩码、响应及 nack 信号，并证明 SoC reset/error 路径；这些内容不能从 C++ 队列存在而推断完成。

## 当前证据

[源码绑定验证报告](../artifacts/tagged/model-io-002/report.json)覆盖矩阵/向量周期回归、外部矩阵执行、地址权限/别名/溢出、跨kernel token隔离、响应稳定性和注册队列。新增21项后，[整套221项回归通过](../artifacts/tagged/model-e2e/model-io-tests-002.xml)。

带nack的真实数据用例共有315个模型请求、378次下游接受、63次重试；最终315个响应均被消费。测试直接记录内存提交，确认只有258次读与57次写，而不是仅靠最终数值相同掩盖重复写入。所有请求/响应和队列均排空，输出与独立 K 顺序参考逐位一致。

ASan/UBSan 另外通过公共端口合约、K=65外部矩阵、非连续/batch寻址和带重试外部矩阵。输出、事件及计数与普通构建对应结果核对，文件位于 `artifacts/tagged/model-io-asan-001/`。

## 下一步与边界

新增[模型物理缓冲所有权组件](mlx-model-physical-storage.md)，用Storage共享所有权和显式pin防止过早地址复用。新[共享物理执行器](mlx-shared-physical-model.md)已将四类后端、pin和同一请求队列连接起来，完成组合生成、权重扰动及重试验证；完整Llama2仍只有编译/地址生命期核对，新共享路径的完整推理及真实系统事务尚待验证。

公共接口、物理绑定和传输队列只解决系统对接的一部分。搬运周期组件已接入；控制绑定层仍需迁移。模型程序/缓冲的系统物理地址分配与装载、实际 Rocket/HellaCache 连接、跨算子 CDC/路由和完整模型周期执行也未完成。搬运的最新独立组件证据在[此报告](../artifacts/tagged/memory-window-001/report.json)，不能据此更新历史矩阵报告的源码身份。

原有完整 Llama2 功能/数值证据仍有效于其声明版本和输入；它不能替代这些新系统路径的执行证据。所有当前端口报告继续保持 `mlx_system_verified=False` 和 `inference_performance_eligible=False`，不以组件周期或请求数量推算模型性能，也不提前扩展 RTL。
