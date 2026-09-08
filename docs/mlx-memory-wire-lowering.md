# 搬运 wire 与完整源调用路由核对

2026-09-08。新增 [memory_lowering.py](../system_sim/physical_device/memory_lowering.py) 与 C++ memory_wire 解码器，将embedding、where、cat、cast、contiguous及检查过的视图接到设备接口。模型的长时冻结运行未改用这些新接口，仍按原来源执行。

## 视图与搬运不能混淆

版本1固定15872 bytes，保存操作/模式/selector/flags、有界参数、至多64个操作数槽、256项参数索引、输出布局和至多4条搬运模板指令。Tensor槽保留存储root身份，重复SSA由参数索引复用；同root的不同视图在C++中共享Storage，不能只复制shape后丢失别名。

- 视图仍重新推导布局并检查共享存储，无数据请求，也不要求本launch产生新字节。它们可在最终主机编排中折叠为地址/布局更新；验证用wire不等于必须发出的设备命令。
- transfer保留LOAD_INDEX/LOAD_PREDICATE/LOAD/CONVERT/STORE，写入新的独立输出。索引与predicate从实际端口响应获得，标量保留显式类型/位模式。
- reshape不能把严格view偷偷物化；cast的dtype参数、强制copy、memory format与设备同一性标志必须一致。不同逻辑device仅保留“需要物化”的语义，物理地址绑定仍由系统负责。
- 当前wire限制最多63个Tensor region、256个参数索引；超出时明确拒绝，不截断cat列表。描述符是控制/地址数据，不扩大4个64-bit寄存器或128-byte搬运暂存区。

## 完整模型编译核对

[memory-wire-lowering-002](../artifacts/tagged/memory-wire-lowering-002/report.json)覆盖全部2659个内存调用：1739个检查过的视图、920个transfer。全部wire经C++解码和原memory_model构造验证，布局、root关系、模式与模板机器字一致。描述符总计42203648 bytes；它不是实际DMA量或推理性能。

连同矩阵、向量和主机控制，[joint-wire-lowering-001](../artifacts/tagged/joint-wire-lowering-001/report.json)逐节点确认6181个源调用恰好一条路由：

| 家族 | 源调用 | 编译路径 |
|---|---:|---|
| matrix | 870 | 6822个batch wire窗口 |
| vector | 2622 | 向量/归约wire |
| memory | 2659 | 920个搬运wire，1739个检查过的布局消除 |
| control | 30 | 真实RV64主机C命令 ABI |

共验证12133个描述符，去掉视图验证描述符后有10394个需执行的主机/设备描述符。此报告核对来源、命令摘要、batch覆盖和唯一去向，**不证明完整模型已在新CPU/设备接口运行**。

## 实际联合链路

插件现在在同一映射内存中执行矩阵、向量和搬运后端。真实RV64主机的arange/argmax产生索引，embedding读取该索引并搬运选中行，再交给矩阵、softmax和下一次argmax。

[spike-memory-chain-003](../artifacts/tagged/spike-memory-chain-003/report.json)通过29个ELF用例。新增的三步链每次包括搬运、矩阵、向量共9个设备窗口：

- embedding使用实际CPU token，普通序列1/2/3，扰动后2/3/0。
- where使用真实CPU比较产生的predicate，选择张量或标量；序列为1/2/0，扰动后2/0/2。
- cat拼接同一存储root的两个不同切片，不能丢失offset或把它们当独立输入内容。

各用例同时比较全部logits位模式；不只比较argmax。softmax的原子libm共享仍显式记录。原有非法头部、机器字、未初始化读取和恢复用例继续执行。

[109项相关回归](../artifacts/tagged/memory-wire-tests-002.xml)通过。全部2659个wire的ASan/UBSan解码结果与release逐字节一致，文件在 `artifacts/tagged/memory-wire-asan-001/decoded.json`；联合链路的29份device.json也与sanitizer版本一致。仍不把关闭第三方Spike LeakSanitizer的运行称为全Spike泄漏检查。

## 边界与下一步

后续[通用主机编排](mlx-generic-host-dispatch.md)已从图自动生成任务表，并在真实RV64上执行正常/扰动的45节点生成图；完整6181节点Llama2也生成了任务计划，但受当前测试内存窗口限制，未执行完整模型ELF。

这块插件测试内存仍只有64KiB，设备时钟由status读取推进；不是完整外存、真实Rocket/HellaCache或CPU周期耦合。下一步需要完整主机/设备编排、实际装载、系统地址空间和错误/完成协议，再做要求模型及输入集合的端到端结果核对。原GPU误差门槛、系统验证与推理性能门禁均不因编译覆盖或组合链通过而解除，RTL继续后置。
