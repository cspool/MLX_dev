# 四后端的整图就绪执行入口

`mlx-ready-graph`是独立分支`mlx-shared-array-v1`中的新C++入口，消费同一个完整`mlx_tensor_semantics_v1`编译程序。它不调用Python推进周期，也不以功能算子或BLAS填补缺少的后端。原`mlx-physical-model`串行入口保留，主工作树中的两个完整运行没有更换版本。

## 执行与资源约束

- 根据实际SSA输入建立依赖图与就绪队列。每个源节点都必须有唯一的matrix/vector/memory/control lowering，缺失引用、循环/前向引用、重复来源和功能回退会拒绝。
- 就绪节点进入有界的主机侧活动队列；`max_active_nodes`不是额外PE上下文。矩阵和向量实际使用同一RF/SPM/ROM及发射/写回/FU资源域。控制叶执行器最多一个，搬运执行器最多一个；合法视图单独验证并消除。
- 结果在后端完成、请求返回和输出写入检查通过后，才于下一边沿发布。批广播矩阵的全部batch窗口结束之前，消费者不能读取其输出。
- 输入值和pin保留到实际消费者完成，引用计数由图依赖重新计算，不直接照搬面向串行执行的`release`提示。视图继续持有原storage，缓冲可以在安全释放后复用。
- 为保持现有物理存储协议，新增/回收绑定只在物理事务排空时进行；这个条件不暂停已有前端的计算。所有模型Tensor仍没有本地数值指针，读取/写入由真正的物理端口响应驱动。

## CPU、搬运和PE的共同响应分发

新增`model_io/physical_mux`，在同一个物理事务槽上建立独立响应通道。多个前端使用各自的AddressSpacePort，但共享全局请求身份；非所有者看不到别人的响应，错误/过期token和响应保持违规仍被诊断。响应消费之后的槽位到下一边沿才能再次使用，多个同周期`advance`调用不会多推进物理内存。

通道在仍持有请求时被销毁会使mux拒绝继续使用，而不是把资源交给新来源。此接口验证的是C++模型内部的响应归属，不是新的CPU或cache实现。控制目前仍为scheduled RV64叶执行器，不是实际Rocket取指执行。

## 已完成证据

[ready-graph-001](../artifacts/tagged/ready-graph-001/report.json)记录187项回归与8个图执行用例的ASan/UBSan/LSan重放；除了输出文件位置，完整事件、周期、资源和数值结果与release一致。响应mux的独立契约也经过检查。

实际执行包括四类后端共同参与的三步生成/cache图、权重扰动后重新生成token、由依赖计数管理的缓冲复用、8个批广播矩阵窗口，以及独立矩阵/向量分支并发后的真实数值汇合。测试核对全部源节点、窗口和请求/响应，不把第一个batch或一个子图通过当作整个输入程序完成。

完整公开Llama2的6181节点程序另做了**1周期上限的启动检查**：初始资产/图依赖被新入口接受，随后按预设周期限制退出1，没有完整推理结果或通过报告。这不是缩减模型推理，也不能证明全部节点的后端运行已经完成。正式报告明确保留`full_inference_result=false`。

开发期间保留了构建告警和mux测试中JSON无符号计数比较写法错误的历史记录；修正的是代码格式及测试取值类型，未放宽通道回收条件。

## 运行

```bash
cmake -S simulator_ext/model_system -B build/mlx-ready-graph -DCMAKE_BUILD_TYPE=Release
cmake --build build/mlx-ready-graph --target mlx-ready-graph -j4
build/mlx-ready-graph/mlx-ready-graph program.json options.json output-NEW
```

配置包括`base`、`bytes`、`memory`、`max_cycles`、`max_active_nodes`、`overlap`及`operator_progress`。关闭`overlap`仍经过相同图/端口协议，只限制同时活动的源节点。只读输入预装载不计为CPU/DMA冷启动时间；程序没有提供参考激活或参考token注入接口。

## 不能扩大的结论

默认依赖仍是整个源算子的完成边界。后续增加了显式选择的[有界双算子块事件模式](mlx-bounded-block-pipeline.md)，可以在登记闭合对内由部分tile结果唤醒消费者，但不是一般CDC。完整模型在此入口的端到端结果、实际CPU/cache系统集成、模板装载时序、其他模型与输入范围均未完成；`complete_cdc_verified`、`mlx_system_verified`和`inference_performance_eligible`仍为false，RTL仍后置。
