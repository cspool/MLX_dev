# Rocket/RoCC 与 C++ 张量后端的系统连接

2026-09-08。新增 [clocked_rocc](../system_sim/clocked_rocc/adapter.cc)、[仿真DPI桥](../system_sim/clocked_rocc/MLXClockedRoCC.sv) 和 [MLXClockedRocketConfig](../system_sim/chipyard/MLXClockedRoCC.scala)。Rocket、cache和系统互连继续使用Chipyard模型；加速器的描述符读取、调度和数值执行仍在C++中，没有新增PE/FU功能RTL。

## 指令接口与数据通路

这是本项目的主机提交ABI，不是论文公开的PE指令编码，也不兼容旧native-v2配置镜像。使用RISC-V custom0槽，ID为`0x4d4c58434c4b0001`。

| funct | 主机参数 | 行为 |
|---|---|---|
| 1 | rs1=描述符地址，rs2=wire字节数 | 在设备排空后接受一个矩阵/向量/搬运任务；当前只允许machine mode |
| 2 | 无 | WAIT在设备完成或错误终态后返回状态；等待期间外部时钟仍推进C++ |
| 3 | rs1=查询号 | 读取状态、阶段计数或ID；查询本身不额外推进设备周期 |

状态位为busy=1、done=2、error=4。查询号14返回ABI ID，15返回错误标志；未登记查询返回UINT64_MAX。CPU返回数据和rd在反压期间保持，前端不覆盖尚未被消费的结果。

任务及张量数据经过同一个请求通路：

```text
实际Rocket CPU复制资产和wire → RoCC提交
  → 外部时钟C++控制器 → SimpleHellaCacheIF → DCache/系统内存
  ← 带身份的真实响应 ← 缓存/互连
  → C++后端完成并排空 → CPU同步、控制算术与最终结果检查
```

内部64-bit请求ID在接受时与有限cache tag绑定，响应必须匹配该绑定。tag只使用`coreParams.dcacheReqTagBits`，不占用HellaCacheArbiter另行追加的请求端口位。当前单事务槽在响应返回前不得复用；这不是任意延迟/重复响应下的无限代数tag协议。

本地SimpleHellaCacheIF已经负责s2_nack重放与s1 store数据寄存，新C++桥不重复重发cache已经保管的事务。byte/half/word/doubleword请求保留地址、size、数据和mask；非法响应tag以及在途reset明确拒绝。cache异常目前仍受本地SimpleHellaCacheIF的assert约束，不宣称支持可恢复页故障。

窄写入还必须由桥按Rocket的`StoreGen.data`规则展开。例如FP16 `0x3c00`写到一个64-bit字的byte 2位置时，数据总线为`0x3c003c003c003c00`，掩码选中byte 2–3。掩码不会自动把低位数据移到该位置。此转换只属于HellaCache总线表示，不修改后端逻辑写值或FP舍入；逻辑C++内存端口仍使用低位有效数据。

`windows[].source_id`暂为提交序号，不伪装成模型源节点ID。通用图执行时，按plan中kind=DEVICE任务的顺序关联该序号；模型源序号、operator_id和batch身份保留在编译任务表及CPU完成标记中。

## 同一通用主机运行时

[graph_runtime.c](../system_sim/physical_host/graph_runtime.c)在完整资产装载尝试终结后新增`MLX_GRAPH_CLOCKED_ROCC`编译分支，仅切换ID/状态/提交传输；资产复制、四类任务路由、batch完成检查和源完成表共用原实现。默认分支仍访问原Spike MMIO插件。正在运行的完整物理模型没有使用或被替换为该新路径。

新 [runner](../scripts/run_mlx_clocked_chipyard.py)将图缓冲绑定到系统RAM中的`0x81000000`测试窗口，编译真实RV64IMAFD ELF；参考输出只进入完成后的检查器。ELF/栈必须位于测试缓冲之前，不把未知地址当作已配置内存。

本节旧`MLXClockedRocketConfig`的外存是`0x80000000`起256MiB，图窗口1MiB。后续独立[16GiB配置与受检初始化](mlx-wide-system-memory.md)已通过高地址生成图；不将其来源/时钟与旧配置混用，也不把容量扩大计为完整Llama2系统执行。

## 验证状态

RoCC直接协议检查覆盖1/2/6-bit tag复用、命令和CPU返回反压、busy reset拒绝、非法函数/权限/对齐、错误响应tag及三后端数据链。最新 [clocked-rocc-002](../artifacts/tagged/clocked-rocc-002/report.json) 的5项检查使用正确的HellaCache字节通道选择，并额外验证高位byte store；ASan/UBSan/泄漏检查结果一致。[159项相关回归](../artifacts/tagged/clocked-rocc-tests-001.xml)通过。

先前clocked-rocc-001的测试端点把低位数据直接写到目标地址，没有覆盖真实cache字节通道语义，不能再作为该接口正确性的充分证据。更新端点后，旧实现的三个数值用例均失败，仅首个输出保持正确，其余为零；失败证据保留于`clocked-rocc-store-regression-001/`。

`clocked-chipyard-001`在编译DPI时失败：该Verilator版本将GNU++14追加在用户C++17选项之后。已用本配置的`CFG_CXXFLAGS_STD_NEWEST=-std=gnu++17`修正，没有修改共享Verilator安装。

`clocked-chipyard-002`实际执行完正常图的21个设备任务、23361个请求/响应，但最终字节检查退出码13，**不是通过**。源码核对和端点复现定位到上述窄store未展开问题；修正只作用于RoCC桥。

修正后的 [clocked-chipyard-003](../artifacts/tagged/clocked-chipyard-003/report.json) 已全部结束并通过：正常及权重扰动两张45源节点图，各执行15个真实CPU控制调用、21个设备任务并消除9个合法视图；全部源完成标记及最终logits/token字节在真实Rocket ELF中检查。每例23361个请求/响应排空，无前端错误或未消费返回。

[clocked-chipyard-wait-001](../artifacts/tagged/clocked-chipyard-wait-001/report.json) 的3个实际阻塞WAIT用例也已通过：FP16、FP32普通链token为1/2/3，FP16真实权重扰动链为2/3/0。每例9个设备任务完成，完整logits和token在CPU中比较；反馈输入使用实际argmax结果，不回填参考token。

这些是新张量后端在真实Rocket/cache通路中的登记图证据。模型仍是小型验证图，不是完整Llama2、作者MLX hybrid或所有输入范围通过；完整模型镜像/资产组合与完整系统执行仍待继续。历史失败保留，不能只因所有请求完成便跳过数值检查。

```bash
.venv/bin/python -m scripts.run_mlx_clocked_chipyard \
  --cases artifacts/tagged/clocked-chipyard-cases-001.json \
  --output artifacts/tagged/clocked-chipyard-NEW
```

使用新目录；只有构建来源和二进制仍匹配时才可用`--phase run`复用构建。报告必须包含实际ELF完成标记、源完成/输出检查、所有设备任务及请求排空。完整公开Llama2的Spike资产装载成功不能移用为本系统的完整模型证书。推理性能、作者MLX hybrid及RTL扩展仍需后续门槛。
