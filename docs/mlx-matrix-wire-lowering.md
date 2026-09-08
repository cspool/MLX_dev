# 模型矩阵调用到二进制设备描述符

2026-09-08。将[矩阵联合桥](mlx-spike-host-matrix-bridge.md)从手写测试描述符推进到模型节点的实际编译路径。实现位于 [physical_device/lowering.py](../system_sim/physical_device/lowering.py)，不修改正在运行的完整模型来源，也不改PE RTL。

## 路由与约束

```text
linear/matmul源节点 + 检查过的布局/物理绑定
  → M/N/K、batch广播与transpose-B推导
  → 每batch一个1088-byte matrix wire
  → CPU从链接进ELF的二进制镜像提交
  → C++解码/后端构造校验 → 已有矩阵调度与微指令执行
```

linear将前缀维度折为M、绑定转置权重；matmul分别计算A/B广播batch索引。非连续B仍保留实际stride，不能先当成contiguous再传错地址。bias有独立只读区域；输出范围不能覆盖输入存储，输出shape/dtype必须与源节点及布局一致。

编码器只接受登记的完整矩阵微程序及资源合约，机器字、精度、bias或资源字段被修改时不能忽略变化重新生成另一个程序。共享Tensor编码器检查64-bit地址、容量、元素offset、rank/stride、读写权限及视图范围。ABI仍为固定1088 bytes；header/未用维度/未用bias/未用ROM槽必须保持规范形式。

## 完整模型编译核对

[matrix-wire-lowering-002](../artifacts/tagged/matrix-wire-lowering-002/report.json)覆盖完整公开Llama2的6181节点程序，其中870个矩阵源调用展开为6822个batch窗口，共7422336 bytes的描述符。

每个描述符都实际交给 [wire_dump.cc](../system_sim/physical_device/wire_dump.cc)解码，并构造已有矩阵后端及地址适配器来检查资源、ROM、精度、batch与绑定。解码后的可执行字段和所有Tensor布局/地址逐字段匹配编译器。`numeric_contract`及`target_status`是编译端说明，不塞进wire；其余执行字段不能丢失。

该阶段的内存端口禁止任何数据请求，**只证明完整编译/解码，不证明6822个窗口已执行模型数据**。物理地址来自检查过的生命期重放，也不表示64-KiB插件测试内存可以容纳完整模型。

## 实际 CPU 链与回归

[spike-matrix-chain-006](../artifacts/tagged/spike-matrix-chain-006/report.json)的固件现在链接Python编译器生成的 `matrix_wire.bin`，由RV64 CPU复制和提交；不再手写矩阵描述符初始化。CPU仍只根据实际argmax修改下一次输入offset，不注入参考token。

FP16/FP32普通、扰动、拒绝和恢复等9个ELF用例通过；[32项相关回归](../artifacts/tagged/matrix-wire-tests-002.xml)另覆盖batch广播、非连续布局、bias、只读/重叠、修改微程序以及损坏二进制拒绝。

ASan/UBSan对全部6822个描述符解码的JSON与release逐字节一致，文件为 `artifacts/tagged/matrix-wire-asan-001/decoded.json`。生成式wire的9个实际ELF用例也通过，`spike-matrix-chain-asan-002` 的设备事件与计数与release一致；仍按联合桥的范围关闭第三方Spike LeakSanitizer，不据此声称整个Spike无泄漏。

```bash
.venv/bin/python -m scripts.verify_mlx_matrix_lowering \
  --program artifacts/tagged/model-e2e/llama2-physical-full-001/program.json \
  --lifetimes artifacts/tagged/model-storage-002/model-lifetimes.json \
  --output artifacts/tagged/matrix-wire-lowering-NEW
.venv/bin/python -m pytest -q tests/test_matrix_wire_lowering.py
```

下一步补向量/搬运wire与联合主机编排，再对接真实Rocket/RoCC/HellaCache。当前完整Llama2仍由冻结的共享原生执行器运行，不能将上述编译核对或小链执行改称它已使用新CPU/设备ABI；模型级正确性、性能和RTL门槛保持原要求。
