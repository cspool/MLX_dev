# MLX 模拟器性能与功能验证目标

## 总目标

基于当前 C++/DSAGEN/Accel-Sim 模拟器，验证 MLX 核心架构在功能正确的
前提下，相对同工作量 baseline 呈现与论文一致且明显的性能提升趋势，并
尽量接近论文报告的提升比例。

## 范围

- 实现与改造周期级模拟器，不开发 RTL。
- 不验证面积、综合资源和功耗。
- 覆盖 tagged CDC、多上下文调度、可编程 PE/FU/RF、skip-hop NoC、
  SIMD/mesh 扩展、SPM/DMA 双缓冲和多端口数据供给。
- 覆盖 BSMM、FFT-CMP、Attention、SWA、elementwise 和完整 Transformer
  block。

## 第一验收标准：性能趋势

每项实验必须满足：

1. baseline 与 MLX 路径的算子、输入、工作量和流量一致；
2. 比较方向与论文一致；
3. MLX 提升至少为 1.2x；
4. 记录与论文提升比例的误差，但不强制要求 10%；
5. 不允许使用目标值拟合比例因子、周期偏置或工作负载选择。

优先验证以下核心收益：

- tagged block / 多上下文延迟隐藏；
- 4-lane 到 16-PE 全阵列执行；
- SIMD8 到 SIMD32、4x4 到 8x8 mesh；
- skip-hop、流水重叠、双缓冲和多端口的独立消融；
- BSMM、FFT-CMP、SWA 和完整 block 的端到端提升。

## 功能验收标准

- 同一输入下，BSMM、FFT-CMP、Attention、SWA、elementwise 和完整 block
  的输出与 PyTorch/NumPy golden 一致；
- FU 次数、pipeline 次数、NoC 传输和内存请求守恒；
- backpressure、bank conflict、tag/event、active-window 和队列满场景正确；
- 仿真确定性、无死锁，重复运行结果一致；
- 所有功能和性能结果均由自动化测试覆盖。

## 实验优先级

1. 统一数值功能执行与周期模拟，建立同输入差分测试；
2. 完成 16-PE 当前语义的组件矩阵和 Figure 21 端到端趋势；
3. 修正 Figure 22 利用率计数并验证延迟隐藏；
4. 完成 skip-hop、context、overlap、双缓冲、多端口消融；
5. 完成 Figure 25 roofline 趋势；
6. 最后改进 Figure 20/24/19 的外部基线比例。

## 停滞与跨实验转向规则

当某个实验的提升方向已经正确，但数值比例难以继续靠近论文时：

1. 冻结当前结果、残差和原因，不继续针对目标调比例；
2. 选择能隔离疑似缺失机制的其他实验，例如 context、NoC、SPM、DMA、
   multiport、SIMD 或 mesh 消融；
3. 新实现必须由架构语义或源码依据驱动，并先做 target-free 验证；
4. 修改后回归全部既有功能和同工作性能测试；
5. 最后返回原实验，将其作为 held-out 检验实现是否自然改善；
6. 若其他实验也否定该机制，则保留负结果并转向下一机制，不做残差拟合。

## 完成条件

- 所有核心功能测试通过；
- 所有核心架构比较均有合格同工作 baseline；
- 核心性能 claim 全部方向一致且提升不低于 1.2x；
- 论文提升比例、模拟结果和证据来源均有版本化 JSON 报告；
- 全量静态检查与 pytest 无失败。

当前里程碑：核心架构趋势 claim 已通过 5/5；定时完成路径已支持可验证
数值负载，五类独立算子（BSMM、FFT-CMP、Attention、SWA、elementwise）
同输入功能覆盖已通过，算子级覆盖为 5/6。下一阶段完成单次仿真的完整
block 动态组合；数值靠近停滞时
执行上述跨实验转向规则，而不是目标值拟合、RTL 或功耗实验。
