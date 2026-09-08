# 浮点向量与归约微程序：模型数值一致性实施记录

2026-09-07。[矩阵微程序](mlx-matrix-microcode.md)之后的非矩阵计算增量。目标仍是 C++ 模拟器先行，再接系统，最后 RTL。

## 已实现范围

`--vector-backend microcode` 将浮点 `add/mul/pow(x,2)/neg/rsqrt/silu/cos/sin/mean/softmax` 编译到有界微程序。源为实际 ATen 调用；输入、标量、shape、dtype、广播和 source_operator_id 保留。整数 add、比较/索引、布局与复制等不因此自动获得目标指令路径。

- 编译器：[model_vector_program.py](../src/mlxsim/model_vector_program.py)；dtype 修正：[model_dtype_lowering.py](../src/mlxsim/model_dtype_lowering.py)。
- C++ 执行器：[vector_program.cc](../simulator_ext/vector_model/vector_program.cc)。实际数据经 SPM 和 RF 执行程序字，不从参考激活回填，不在控制循环中直接调用整行归约代替微指令。
- 数值参考：[model_numeric_reference.py](../src/mlxsim/model_numeric_reference.py)。图执行、矩阵 K 顺序和归约控制独立；`expf/sqrtf/cosf/sinf` 明确共享系统 libm 原子实现，记录路径与 SHA。它不证明作者硬件数学单元逐位等同 libm。

本页描述**功能微程序路径**。后续已经新增向量多上下文周期状态及可替换内存端口，并接入同一张量入口，见[向量周期组件](mlx-vector-window-simulation.md)。其网络、完整系统地址/访存与 Chipyard 仍待接入，不能把组件周期覆盖直接套到完整模型。

## 资源与执行模式

| 项目 | 约束 |
|---|---|
| RF | 16 × 64-byte 容量；本模板使用 8 个向量，不扩大物理 RF |
| SIMD | FP32 每向量 16 lane；FP16 输入显式转换到 FP32 运算 |
| 超越/除法 | 每条微指令至多作用于 4 lane；完整向量由四个 lane-group 指令完成 |
| SPM | 全局容量 8 KiB；该工作集使用 320 bytes：两份 64-byte 输入区域、128-byte 部分和栈、64-byte 输出区域 |
| ROM | 单一去重 ROM，所有阶段引用同一组程序字；不得超过 32 words，完整 Llama2 运行中最大为 28 |
| 归约宽度 | 当前单个最后维，非空且不超过 2³¹；栈固定 32 个 FP32 标量，不按整行长度分配片上存储 |

FP32 MUL/ADD/SUB/DIV 分别 RNE，禁止 contraction、重结合与 FTZ/DAZ。RSQRT 明确为 `1 / sqrt(x)` 的两步 FP32 计算；SiLU 明确为 NEG → EXP → ADD(1) → DIV。具体 opcode、宽度切分和精度规则是本项目重建模式，不冒充论文公开的完整 ISA 编码或实测延迟。

归约采用 16-lane 分块、XOR shuffle 与逐级 ADD；跨块部分和通过有界 SPM 栈进行二叉合并。补齐到二次幂的虚拟输入使用显式零，保持与相邻配对归约相同的顺序，包括奇数宽度和符号零。MAX 的虚拟输入为负无穷；shuffle 禁止读取非活跃尾部 lane。

Softmax 分三遍流式执行：

```text
输入 → MAX 归约
输入 → SUB(max) → EXP → SUM 归约
输入 → SUB(max) → EXP → DIV(sum) → 输出
```

第三遍重算 EXP，避免把整行概率/指数临时数组放进 SPM。所有浮点计算仍由微指令完成，控制层只管理形状确定的循环、地址和部分和层级。共享 libm 原子函数并不意味着使用 Python/GPU 回退；目标进程只有 C++。

## dtype 边界修正

`softmax(dtype=FP16)` 的语义是先转换输入，再做运算，而不是仅转换输出。见 [PyTorch softmax 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.softmax.html)。输入 `[1000.0, 1000.2]` 的 FP32 softmax 再转 FP16 约为 `[0.4502, 0.5498]`，而先转 FP16 再 softmax 为 `[0.5, 0.5]`，argmax 也可能不同。

编译 pass 已在 MAX、EXP/SUM 和输出三个阶段的输入装载后插入 FP32→FP16→FP32 转换；RF 中反复转换保持所需输入量化，不新增整行中间数组。普通 C++ 功能参考及显式数值参考同步修正。当前 Llama2 使用 FP16→FP32 softmax，修正前后生成的完整程序逐字段相同。

FP32 输入显式 `mean(dtype=FP16)` 尚无单独登记的精度合约，编译阶段明确拒绝，不能将其悄悄等同为末尾转换。当前模型的 mean 是 FP32→FP32，不受该限制影响。

## 完整模型结果与证据边界

[`llama2-vector-001`](../artifacts/tagged/model-e2e/llama2-vector-001/numeric-conformance-002.json) 实际加载并消费公开 dense Llama2-7B 的全部 291 个参数张量，完整执行 32 层、8-token prefill 和两次 1-token decode：

- 6,181 个源调用全部执行；870 个矩阵调用和 2,622 个浮点向量调用走微程序，BLAS=0、Python/GPU tensor fallback=0。
- 矩阵完成 65,175,028,352 次有效 lane MUL 和同量 ADD；向量部分执行 14,869,023 条微指令，最大归约栈层级为 8。
- 对明确数值合约参考，三个 logits 向量共 96,000 个 FP16 值逐位一致；token 为 `393/372/338`，两次 decode 均由前一次实际 argmax 经形状视图驱动。
- 输出与此前矩阵微程序＋普通 C++ 浮点路径也逐位相同：新 lowering 没有改变此前的数值结果。原 GPU 参考门槛仍未通过，历史误差和失败文件保持不变。

[verify_mlx_model_numeric.py](../scripts/verify_mlx_model_numeric.py) 核对权重/程序/二进制摘要、全部参数的实际引用、逐调用路由、层与 cache 状态、数值参考覆盖及 token 数据流。源与参考程序除 15 个 arange 和 3 个设备转换的 CPU/CUDA 放置元数据外完全相同；该归一化仅服务数值比较，不证明内存/时序等价。验证器还直接检查 logits 有限性并独立重算 argmax。

**通过的是该公开基座、该输入、该次执行及其数值合约的一致性，不是整体系统验收。** 数值参考共享原子 libm，因此单独对四类原子函数做了与双精度 math 的采样检查，固定一 FP32 ULP 误差界；这仍不是硬件实现验证，也不是全部 FP32 输入的穷举证明。

剩余 2,689 个源调用仍由 C++ 功能语义处理，包含索引、mask/select、布局、转换/复制与别名。它们需要相应的目标侧或主机 ISA/memory 路径及消除合法性证据，不能仅凭数学输出一致就认定硬件 lowering 全覆盖。还需其他要求模型、更多输入用例、跨算子调度、完整系统 DMA/同步验证。报告继续保持 `mlx_system_verified=False`、`mlx_hardware_mapping_complete=False` 和 `inference_performance_eligible=False`。

## 回归与运行

[114 项组件/集成回归](../artifacts/tagged/model-e2e/vector-integrated-tests-005.xml)已通过，覆盖有界归约、1～8192 宽度、奇数/尾部/符号零、广播与混合精度、非法 ROM/分组/未就绪 lane、dtype 前置转换、独立原子误差检查、归一化图编译，以及数值通过不能越过系统性能门禁。ASan/UBSan 另外执行 mean(8192)、softmax(8192)、FP16→FP32 softmax(17)，结果位于 `artifacts/tagged/vector-asan-001/`。

```bash
.venv/bin/python -m scripts.run_mlx_tensor_semantics \
  --inventory artifacts/tagged/model-e2e/llama2-reference-005/inventory.json \
  --matrix-backend microcode --vector-backend microcode \
  --output artifacts/tagged/model-e2e/llama2-vector-new

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
.venv/bin/python -m scripts.capture_mlx_model_reference \
  --device cpu --capture-bindings --matrix-reference kasc --float-reference explicit \
  --output artifacts/tagged/model-e2e/llama2-reference-numeric-new
```

显式参考模式改变的是参考的数值合约，不是透明插桩。按照本次使用的 `trace-patch-target-discovery` 技能，透明追踪等价检查分别在该模式内执行；沿用 request/forward/layer/operator 连接键和有限旁路观察，不注入参考中间值，也不为取得通过修改原 GPU 误差门槛。
