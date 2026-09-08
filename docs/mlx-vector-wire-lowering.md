# 向量与归约算子的设备 wire 路径

2026-09-08。新增 [vector_lowering.py](../system_sim/physical_device/vector_lowering.py) 和 C++ vector_wire 解码器，把既有向量微程序带入CPU/设备接口，不在接口后仅按算子名称另算答案。完整模型的冻结运行未被修改。

## 描述符承载什么

版本1固定4288 bytes，包含两个Tensor/标量操作数、输出布局、至多32个ROM机器字、16个常量槽、12个有界阶段索引表，以及width/keepdim/softmax输入窄化标志。标量和常量用binary64描述符位模式保留编译输入，再由已登记执行模式转为FP32；这不表示PE新增了FP64计算单元。

保留add、mul、pow2、rsqrt、silu、cos、sin、neg、mean和softmax的实际机器字与阶段顺序。FP32输入的FP16 softmax必须保留归约前的CVT_DOWN/CVT_UP，不能只在输出窄化。非法phase索引、非零保留字段、未使用槽、dtype/shape、广播和读写区域冲突均拒绝。

阶段表属于控制描述符，不算额外PE指令ROM：它用于准确搬运当前C++程序语义，最终硬件需要以受约束的控制存储或状态机实现其顺序。RF仍为每PE16个向量、每上下文使用8个；SPM仍为全局8KiB、每上下文320 bytes；不能把描述符表的存在当作物理控制成本已解决。

## 完整模型编译与解码

[vector-wire-lowering-002](../artifacts/tagged/vector-wire-lowering-002/report.json)将公开Llama2全部2622个向量调用编码为11243136 bytes的wire。C++解码后的ROM、阶段表、常量、dtype/shape、stride、offset与物理绑定均和编译器一致，并实际构造原向量后端验证条件。

这是完整编译/解码核对，不是这些模型数据已在新接口执行。45项新增组件测试覆盖10类算子、两种精度、归约宽度、窄化softmax及损坏描述符；包括现有矩阵/主机/链路回归共[77项通过](../artifacts/tagged/vector-wire-tests-001.xml)。ASan/UBSan对全部2622个描述符解码的JSON与release逐字节相同，见 `artifacts/tagged/vector-wire-asan-001/decoded.json`。

## 实际 CPU—矩阵—向量—argmax 链

Spike插件现在按wire magic选择矩阵或向量后端，二者共用同一映射内存、请求身份与完成检查。CPU从编译器生成并链接到ELF的二进制镜像提交任务：

```text
CPU写输入 → 矩阵计算 → 向量缩放或softmax → 实际RV64 argmax
  → CPU用计算所得token选择下一次输入
```

[spike-vector-chain-002](../artifacts/tagged/spike-vector-chain-002/report.json)通过17个ELF用例，覆盖FP16/FP32、权重扰动、非法输入及恢复。加入向量后的三步链有6个成功设备窗口，token及全部输出位模式与参考一致；普通token为1/2/3，扰动后为2/3/0。softmax参考的归约控制独立，但原子libm明确共享并记录身份，只证明编译与数据流，不是独立数学单元或作者电路精度验证。

[ASan/UBSan联合版本](../artifacts/tagged/spike-vector-chain-asan-001/report.json)也通过17例，全部device.json与release逐字节一致。仍只插桩插件及C++依赖，第三方Spike的LeakSanitizer保持关闭，不能声称全Spike无泄漏。

```bash
.venv/bin/python -m scripts.verify_mlx_vector_lowering \
  --program artifacts/tagged/model-e2e/llama2-physical-full-001/program.json \
  --lifetimes artifacts/tagged/model-storage-002/model-lifetimes.json \
  --output artifacts/tagged/vector-wire-lowering-NEW
.venv/bin/python -m scripts.verify_mlx_spike_matrix_chain \
  --output artifacts/tagged/spike-vector-chain-NEW
```

输出目录必须选新路径。脚本保留旧名称，但现在包含矩阵/向量联合用例。

## 仍未完成的系统工作

后续[内存wire与完整源路由核对](mlx-memory-wire-lowering.md)已覆盖2659个内存计划，并在联合桥实际执行embedding/where/cat后进入矩阵与向量。四类编译报告现已逐节点合并核对，但完整CPU/设备模型编排、真实系统接口和全模型数据执行仍待完成。

主机/设备全模型编排、真实装载与Rocket/RoCC/HellaCache仍未完成。插件继续使用64-KiB映射测试内存，以每次status load推进一个设备tick；不能将其计数视为CPU耦合的系统性能。

当前完整Llama2运行仍属于冻结的共享原生执行器，不因本接口增量而改标为已在Spike插件执行。要求模型与输入集合的端到端正确性通过之前，不开放推理性能或RTL扩展。
