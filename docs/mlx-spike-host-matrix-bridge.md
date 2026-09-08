# 实际 RV64 主机与矩阵后端的联合执行桥

2026-09-08。新增 [physical_device](../system_sim/physical_device/spike_matrix.cc)，把真实RV64主机控制代码与已有C++矩阵周期后端连接起来。它是Spike插件中的设备映射测试内存，不是完整模型、Chipyard缓存或RTL性能验收。

## 已运行的数据流

```text
RV64 CPU写入数据/描述符 → 提交矩阵任务 → C++矩阵后端真实访存与微指令
  → CPU等待完成 → RV64 argmax写出token → CPU用该token选择下一次矩阵输入
```

CPU通过实际store初始化输入与描述符，不从插件侧预装载golden输出。矩阵读写使用同一设备映射内存，Tensor只保留布局/容量；输出每个字节必须由本次launch产生，不能把上次结果或初始化哨兵当作完成。主机argmax来自[实际C/RV64控制代码](mlx-rv64-host-control.md)，其结果实际参与下一次输入地址计算。参考token和logits只在执行后比较，不用于下一步赋值。

本地Spike版本提供[MMIO插件回调](../build/chipyard-native/toolchains/riscv-tools/riscv-isa-sim/riscv/mmio_plugin.h)，但构造参数不提供simulator RAM指针。本桥因此显式使用插件拥有的测试内存，CPU与加速器均访问它；历史链测试配置为64KiB。后续改为[C++稀疏宿主存储](mlx-sparse-mapped-memory.md)，允许大逻辑窗口及高偏移读写，但仍未直接接入Spike DRAM或HellaCache。

## 矩阵 wire ABI

[matrix_wire.h](../system_sim/physical_device/matrix_wire.h)定义1088-byte、little-endian版本1描述符：

| 区域 | 字节数 | 内容 |
|---|---:|---|
| 头部 | 128 | magic/version/transpose-B与bias标志，M/N/K、三个batch索引、阶段长度、保留位 |
| 四个Tensor | 704 | A/B/bias/output的176-byte布局和物理区域描述 |
| 微指令 | 256 | 至多32个32-bit机器字，装在64-bit传输槽中，上半部及未使用槽必须为0 |

解码器绑定已登记的 `mlx-matrix-f32-kasc-v1` 模式及固定M2/N16/K64、16个RF向量、8-KiB SPM、32-word ROM；实际机器字继续交给原矩阵后端验证和执行。FP16/FP32输入与输出、转置、batch和bias不是凭名字映射，均有显式字段。这是本项目编码，不是论文公开编码，也不兼容旧tagged native-v2 RoCC镜像。

设备寄存器偏移为ID=0、descriptor=8、launch=16、status=24、cycles=32、error=40；payload从0x1000开始。测试设备基址为0x100000000，主机实际使用64-bit地址。输入/输出不能与描述符重叠，writable区域不能覆盖输入；忙时拒绝改写配置或payload。

每个CPU status load推进一个后端tick，后端请求/响应延迟与资源仲裁仍实际发生，但这个时钟与CPU周期没有建立频率/时序对应。因此cycles只描述轮询驱动的组件执行，不能作为系统延迟、TTFT或吞吐。主机使用I/O fence连接提交、等待和数据消费。

## 当前证据

[spike-matrix-chain-005](../artifacts/tagged/spike-matrix-chain-005/report.json)完成9个实际ELF用例：

- FP16与FP32的三步普通链，token为1/2/3；修改真实权重后变为2/3/0，全部logits位模式也匹配独立参考。
- 非法magic、保留位与高位机器字在执行前拒绝，矩阵数据读写为0。
- 未初始化权重读取返回错误；此前4个合法A读取保留计数，没有输出写入，错误响应最终排空。
- 非法头部后重新提交合法任务，4次launch中3次正常完成，验证基本恢复路径。

每个正常三步链有60个设备读取、12个设备写入；CPU实际写入参数与描述符并读回结果。报告分别保存CPU payload访问与设备访问，不把它们混成DRAM事务数。

[ASan/UBSan版本](../artifacts/tagged/spike-matrix-chain-asan-001/report.json)也通过相同9个ELF用例，所有设备JSON事件和计数与release逐字节一致。只对插件及其C++依赖启用sanitizer；为避免未插桩第三方Spike自身分配干扰，禁用了LeakSanitizer，不能据此声称整个Spike无内存泄漏。

重放命令（输出必须选新目录）：

```bash
.venv/bin/python -m scripts.verify_mlx_spike_matrix_chain \
  --output artifacts/tagged/spike-matrix-chain-NEW
.venv/bin/python -m scripts.verify_mlx_spike_matrix_chain --asan \
  --output artifacts/tagged/spike-matrix-chain-asan-NEW
```

## 仍需完成

最新[搬运wire](mlx-memory-wire-lowering.md)也已连接同一插件，embedding、where和cat与矩阵/向量/真实CPU控制形成29个ELF用例；完整6181源调用已有唯一编译路由核对。这不等于完整模型已改走该接口。

后续[向量wire与联合执行](mlx-vector-wire-lowering.md)已接入同一插件，覆盖完整模型2622个向量描述符解码，并实际跑通矩阵→缩放/softmax→RV64 argmax的17个ELF用例。仍是映射测试内存和轮询驱动时钟，不是完整模型或真实SoC通过。

后续[模型矩阵wire编译](mlx-matrix-wire-lowering.md)已覆盖870个源调用/6822个batch窗口，并经C++解码和构造检查；9个ELF链现在使用编译器生成的二进制镜像。这里新增的是矩阵编译与接口证据，不是完整模型已在该插件桥执行。

1. 矩阵、向量、搬运wire及[通用主机编排](mlx-generic-host-dispatch.md)已接模型节点/绑定路径。当前只有登记组合链和45节点生成图实际执行，不是6181节点模型已改走此桥；完整资产流入仍待实现。
2. 继续验证多窗口、更多布局、忙时访问、运行期错误恢复等系统边界；现有结果不能外推为任意并发或reset协议通过。
3. 建立真正的CPU/设备时钟、Rocket/RoCC/HellaCache接口、真实外存与装载范围；本测试映射内存不能替代它们。
4. 正在运行的 `llama2-physical-full-001` 保持冻结来源，本桥在其来源集合之外开发，不能重新标注该run为已使用真实CPU。完整模型及要求输入集合通过前，性能门禁与RTL扩展继续关闭。
