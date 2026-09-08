# 实际 RISC-V 主机控制代码与命令 ABI

2026-09-08。补齐“控制叶指令语义正确”到“真实主机 load/store 与循环执行”之间的缺口。实现位于 [system_sim/physical_host](../system_sim/physical_host/control_runtime.c)，独立于正在冻结运行的共享物理模拟器。它不修改PE/FU RTL，也尚未替换完整模型runner中的C++控制绑定。

## 已实现的主机执行路径

`mlx_host_control_execute` 是可用GNU RISC-V工具链编译的freestanding C函数，实际执行arange、整数add/mul、le和最后一维argmax。Tensor地址、shape、stride和offset都参与CPU地址计算；数据用对齐的LD/LWU/LHU/LBU读入，以SD/SB写出。循环、分支、MUL、FLE.S/FLT.S和fence进入真实RV64 ELF，不再把这些动作仅记为描述符绑定计数。

整数加乘按64-bit模运算，比较保留完整int64精度；argmax保留第一个最大值和第一个NaN。FP16通过整数位处理精确提升为FP32，NaN按登记主机合约转换为带符号的规范quiet NaN，不要求CPU具备Zfh。浮点比较在FP32域执行，保留fflags行为；此路径不是任意ATen弱标量promotion的通用实现。

启动代码设置栈、清零BSS、启用FPU并绑定trap出口。函数要求调用者稳定持有描述符和输入；无并发修改协议。输出不能与输入的存储范围或命令自身重叠，避免边执行边覆盖尚需读取的元数据。

## 主机内存 ABI，不是新增PE opcode

[control_runtime.h](../system_sim/physical_host/control_runtime.h)定义版本1格式，所有字段为little-endian 64-bit字：

| 结构 | 字节数 | 内容 |
|---|---:|---|
| Tensor | 176 | base、bytes、元素offset、rank、dtype、读写标志，以及各8项shape/stride |
| Operand | 208 | Tensor或显式类型的标量位模式；未使用字段必须为0 |
| Command | 640 | magic/version/主机操作/flags/extent/reserved、两个Operand、输出Tensor |

主机操作编号只选择C例程，不是MLX PE指令，也不是现有tagged native-v2 RoCC编码。rank最多8；输入输出地址按元素宽度自然对齐，检查乘积/地址溢出、view范围、广播结果、输出布局、权限、保留位、argmax轴和FP舍入模式。非法静态描述符在任何输出store之前返回错误。

这些检查不能证明实际RAM映射、PMP/MMU权限或accelerator缓存一致性。未映射地址仍可能触发CPU异常；fence也不是非一致性缓存刷新协议。真实Rocket/HellaCache连接和错误排空仍需单独验证。

## 编译器如何路由到它

[lowering.py](../system_sim/physical_host/lowering.py)先核对源节点的 `control_program` 与登记叶程序逐字段相同，再生成640-byte命令。如果原叶程序机器字被修改，不能忽略它并换成C内建算法。Tensor使用检查过的布局及显式物理绑定；整数标量不经float窄化，浮点常量按已登记FP32比较域编码。

[host-lowering-001](../artifacts/tagged/host-lowering-001/report.json)对完整公开Llama2的6181节点程序重新核对布局及SSA生命期，使用已检查的地址生命期重放结果，为全部30个控制调用生成命令，共19200 bytes。该证据是编译/绑定核对，没有在这些地址执行完整模型数据。

## 实际 ELF 证据

[physical-host-003](../artifacts/tagged/physical-host-003/report.json)通过40个控制/拒绝用例和全部65536种FP16位模式。测试数据位于独立的 `0x100000000` 段，CPU通过64-bit描述符取地址；因此不能用低32位地址恰好可访问的测试掩盖截断。

输出与独立NumPy期望逐字节核对，同时检查输入和前后guard不变、错误状态及fflags。覆盖整数回绕/极值、广播与非连续视图、空输出、NaN/tie、FP模式拒绝、无效rank/shape/stride、只读输出、越界/溢出、输入及命令重叠。

22个成功命令还由Python lowering重新编码，与链接后ELF里的C结构逐字节比较，避免只验证C内部自洽而未验证前端ABI。反汇编检查所需指令族；实际Spike进程执行并返回成功，不能仅靠反汇编代替执行。另有[9项主机编译路由/执行回归](../artifacts/tagged/physical-host-tests-002.xml)通过。

```bash
.venv/bin/python -m scripts.verify_mlx_physical_host \
  --output artifacts/tagged/physical-host-NEW
.venv/bin/python -m scripts.verify_mlx_host_lowering \
  --program artifacts/tagged/model-e2e/llama2-physical-full-001/program.json \
  --lifetimes artifacts/tagged/model-storage-002/model-lifetimes.json \
  --output artifacts/tagged/host-lowering-NEW
```

证据目录必须是未存在的新目录。当前使用本地GNU RISC-V GCC 10.2与既有Spike构建；不新增下载的CPU模拟器或修改其源码。

## 剩余集成

新[Spike主机/矩阵联合桥](mlx-spike-host-matrix-bridge.md)已让实际argmax结果驱动下一次矩阵输入，FP16/FP32及扰动/拒绝/恢复用例通过。它使用插件映射测试内存和status轮询推进后端，不是Rocket/HellaCache或完整模型验收。

1. 将这些主机命令与矩阵/向量/搬运的设备命令联合编排，使CPU实际计算出的token、mask和位置数据进入后续设备任务。
2. 明确新设备描述符与RoCC ABI，配置真实系统地址空间、装载、完成事件与缓存/错误语义；不能把旧native-v2小负载证书直接移用。
3. 在Rocket/Chipyard上执行相同主机代码，连接完整模型的真实访存与输出；对照源码、编译器、命令及全部算子覆盖。
4. 正在运行的 `llama2-physical-full-001` 仍使用冻结的C++控制组件，不能改称已经使用本C主机实现。新组件在冻结集合之外开发，完整模型正确性与性能门槛保持关闭，RTL继续后置。
