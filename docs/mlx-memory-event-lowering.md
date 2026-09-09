# 独立搬运控制器的 C++ 事件路径

本增量补齐独立 memory 窗口，不是整模事件模拟或 Chipyard 验收结果。运行核心仍为 C++ `simulator_ext/event_schedule`；Python 只负责 canonical 计划的事件编译与独立对照，不推进模拟周期。既有整模运行的来源和二进制保持冻结，没有修改 RTL。

## 参考与资源约束

[模拟器卡片](mlx-dfu-simulator-reference.md)的 Q2、D005/D017 强调编码、执行、状态和后继消费一致；Q4 要求 DMA、计算、应用和宿主计时分开。本实现采用这些检查原则，资源与时序取自当前 MLX C++ 搬运控制器，不复制 DFU 的硬件参数或内部代码。

schema v5 新增独立 `controllers` 域，与阵列 `blocks` 并发。搬运控制器拥有 32 B 数据寄存器、128 B staging、8 B 转换结果锁存和 8 B 请求数据锁存；最多一个非空搬运源驻留。它不占用 PE 上下文、RF、SPM 或 ROM，转换不占用 PE 算术/SFU。容量是独立资源，不能隐含扩大阵列容量。

阵列与搬运控制器共享同一条有限 DMA 事务通路：请求、响应消费和下一边沿释放参与竞争。转换可与阵列计算重叠，但不能因此绕过共享总线。控制器请求/响应周期以本次准入为起点，区别于阵列全局端口节拍。

## 完整窗口编译

`src/mlxsim/model_memory_events.py` 从实际 `memory_program` 和输入布局重新生成 canonical 计划；拒绝计划不一致、训练 dropout、错误 cast dtype 和未登记的 new_ones 布局/pinning。覆盖类型转换、连续化/布局实体化、cat、embedding、advanced index、new_ones、where 与无搬运 view/split。

- 索引、谓词、数据读取、转换、回写均保留实际事件数量；循环按需遍历，不裁剪实例。
- 零宽 embedding 仍读取实际索引，不能简单按输出字节为零跳过工作。
- view/真正空搬运的原生窗口为 0 周期；整图源完成发布仍占一个边沿。分别记录 `window_cycles` 和发布区间，不把这一个边沿误报为窗口误差。
- where 两分支若数据类型或 tensor/literal 路由不同，必须提供真实谓词选择序列。当前组件对照只从实际执行的响应提取布尔值，丢弃时间戳和延迟；独立重放重新验证该序列。缺失选择时拒绝预测，不以参考输出或测得耗时填充事件。

事件核心不计算 Tensor 数值，也不验证索引值合法性。当前预测仅适用于已通过实际数值/索引检查的成功执行；固定延迟内存端点不根据地址建模 cache/DRAM。整模仍须接入真实动态控制与地址相关访存，不能以本窗口的计数覆盖替代。

## 已验收范围

`memory-event-regression-002`：176 项回归通过，其中新增 28 项。`memory-event-safety-002`：166 次 ASan/UBSan/LSan 重放通过（136 正常、30 预期拒绝）。

`memory-event-alignment-002` 独立重跑 18 个实际数值搬运窗口与事件窗口：8 个 strided cast、6 个选择器/常量案例、4 个 view/空形状案例；周期及读写字节一致，原生完整报告与输出重放一致。测试另核对指令数、索引/谓词读取、请求/响应/转换等待和零阵列存储占用。原生使用 `TensorMemoryPort` 的固定延迟模式（`external=false`），不是 Rocket 缓存访问对照。

另保留混合控制器/PE 并发、共享总线竞争、两个控制器节拍起点、资源越界拒绝和数据相关路由拒绝检查。现有 22 个矩阵与 48 个向量窗口已在当前事件引擎重放，分别见 `matrix-event-alignment-003`、`vector-event-alignment-002`。这些窗口误差为零不表示整模误差为零。

编译入口沿用 `scripts.compile_mlx_matrix_events --family memory`；独立重放入口为 `scripts.verify_mlx_matrix_event_alignment --family memory`。归档包含真实输入/节点、谓词见证、输出、报告、来源摘要，不包含模型 checkpoint、运行中整模证据或私有卡片正文。

## 仍缺的主线

完整 RV64 控制流、batch/源图路由、逐实例块间事件、模板装载与缓存时序仍待接入。事件引擎全局 completion-first 顺序与原生多源 whole-tick 顺序尚未完全对齐，需要独立解决和验证。`full_model_event_lowering_complete=false`；暂无 `(C_event-C_e2e)/C_e2e` 整模结果。Chipyard 整模验收仍最优先，GPU 新实验等待 P0，RTL/PPA 最后。
