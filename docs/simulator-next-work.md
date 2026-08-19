# MLX 模拟器下一阶段工作

当前 Fig.23、Fig.19、Fig.20 已达到 15% 数值误差目标，统一 workload lowering
工具链也已可运行。下一阶段用于提高模拟器的独立性、周期精度和通用性，不涉及
RTL、功耗或面积。

## 1. 周期级物理化

当前 Fig.23 使用保留 `raw_cycles` 的 latency service，Fig.19/20 使用 Python
性能组合服务。下一步把启动填充、SPM round-trip、带宽拥塞和算子服务时间实现为
逐周期调度、存储及网络状态。

验收：关闭结果后处理后仍能保持工作量守恒、baseline 趋势一致，主要论文点误差
不超过 15%。

## 2. 独立留出验证

当前共享参数由论文点辅助选择，最坏交叉验证误差为 22.16%。增加未参与拟合的
序列长度、batch、block size 和硬件配置，并冻结参数后再运行。

验收：参数不重新拟合，所有留出点方向正确，最大误差尽量进入 15%，无法达到时
明确报告适用范围和失效机制。

## 3. 自动模型前端

当前统一入口从人工编写的模型图 YAML 开始。补充 PyTorch FX/ONNX 导入、shape
propagation、算子合法化、CDC 划分、tag 分配、PE placement、寄存器分配及
SPM/DMA 规划。

验收：真实模型子图可通过单条命令生成合法的 MLX overlay 或 KernelProfile，且
所有 lowering 决策均可追踪到原始模型节点。

## 4. 工具链覆盖扩展

当前工具链覆盖 3 个图、14 个节点和 12 个执行单元。将其扩展到 Fig.19/20/23
全部形状、完整 Transformer block，以及 Llama2/FABNet 的多层组合。

验收：全部目标负载从同一入口完成 lowering、执行和 replay；不得再依赖图号专用
的人工拼接步骤。

## 5. 同输入数值等价验证

为 lowering 后的每个算子和完整 block 建立 PyTorch/NumPy golden reference，
比较中间张量、最终输出、事件边界及不同 mesh/SIMD 配置下的结果。

验收：相同输入下，所有中间边界和最终输出满足注册的 FP16/FP32 误差阈值；性能
比较仅在功能等价、工作量一致的执行之间进行。

## 推荐顺序

1. 先完成同输入数值等价框架；
2. 实现自动模型前端和完整 lowering；
3. 将 latency service 下沉为周期级机制；
4. 扩展完整模型与形状覆盖；
5. 最后冻结参数并进行独立留出验证。

现有依据：[数值收敛证书](../artifacts/results/numerical-convergence-goal-run193.json)、
[统一工具链说明](workload-lowering-toolchain.md)。
