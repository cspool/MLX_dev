# C++ Boolean mask 与有运行时守卫的分支特化

本增量依据[完整 BERT QA 参考清单](mlx-bert-qa-reference.md)补齐 `ge`、Boolean `bitwise_and`、全量 `all` 和 Boolean scalar guard 的实际 C++ 路径。先验证独立模拟器，不改四个冻结的 Llama2 运行，不实现 RTL，也不把控制组件通过当作 BERT/系统端到端通过。

## 编译路径与指令依据

这些计算属于 RISC-V 主机控制域，不是新造的 PE 指令或 GPU warp 语义。新增 `mlx-controller-rv64-leaf-v2` 描述符，保留现有 v1 编码/行为；v2 明确校验标准 RV64 指令序列，不能用返回常量的模板替换 guard。没有增加寄存器、ROM、在途指令或访存容量。

| 实际来源 | C++ 路由 | 实际计算与约束 |
| --- | --- | --- |
| `aten.ge.Scalar` | `ge` | 有符号 I64 的 SLT/XORI，或交换操作数的 FLE.S；F16 输入按原接口提升到 F32；保留 NaN/fflags 规则 |
| `aten.__and__.Tensor`、`aten.bitwise_and.Tensor` | `bitwise_and` | 读取两个 Boolean 张量，广播后执行 RV64 AND；不声称已支持任意整数 bitwise overload |
| `aten.all.default` | `all` | Boolean 全张量归约：初始化 true，按实际布局逐元素读取，执行 SLTU/AND；空张量返回 true，不提前短路漏读元素 |
| `aten.is_nonzero.default`、Boolean `aten._local_scalar_dense.default` | `guard` | 实际读取恰好一个 Boolean 元素，SLTU 规范化，XOR 对照编译分支的期望值；不匹配时失败，不能继续回放该分支 |

`control_program.py` 生成真实指令字；`control_model` 执行同步叶程序；`control_schedule` 按外部响应和逐指令延迟推进。单指令在途、单事务在途、32-word 上限及 512-byte RV64 寄存器状态保持不变。周期入口读取实际端口数据，不从虚拟张量的宿主 backing 取值；请求/响应回压和错误归属规则继续适用。

guard 的期望 Boolean 是**图特化条件**，不是模拟器的计算结果。结果来自输入读取和实际 RV64 运算。宿主模型检查 XOR 结果；不匹配时输出不写入，不产生成功报告。它不是完整 Rocket 分支执行或异常模型，也没有自动选择/编译另一条分支；后续系统编排还需实现经过检查的变体选择或完整控制流。

## 分支条件不能在图调度中丢失

编译器将 scalar guard 物化为独立 Boolean SSA 结果，并给后续源调用附加 `control_dependencies`。依赖和普通数据依赖一样参与 ready DAG 的准入、生命期和完成可见性检查，不能让一个只依赖初始输入的分支节点绕过 guard 提前启动。

原生串行、张量执行入口和 ready DAG 都检查完整守卫依赖集合；删除、伪造或重复依赖被拒绝。配对编译及完成证据检查也保留这些边。当前采用保守的“全部既有 guard 约束所有后续源”规则，未实现一般分支重汇合优化。

Boolean scalar 只在输入为单元素 Boolean、实际捕获结果确为 Boolean 时接受；浮点 scalar、其他 dtype、多元素和缺少明确 RV64 后端均拒绝。新增路径不提供框架/通用功能算子的回退。

## 已完成的验证与边界

[444 项回归](../artifacts/tagged/mask-control-tests-002/regression.xml)通过，覆盖旧控制指令、张量语义、物理内存、ready DAG、配对与原生模型验收工具，以及本次 24 项 mask/guard 测试。其中：

- 实际物理输入与宿主占位值不同，核对 I64 边界、F16/F32 的 NaN/无穷/正负零、Boolean 广播、28/64/66 元素尾部和空归约。
- 正常/padding 特化组件图逐源编译，在串行与 ready DAG 中实际执行。guard 完成可见前，后续分支节点不能启动；改动实际 mask 后同一编译分支被拒绝。
- 错误 guard 的专用证据确认读取了 1 个实际 Boolean、写入为 0，并拒绝成功报告。
- [31 次 sanitizer 重放及 28 组独立 Spike 对照](../artifacts/tagged/mask-control-safety-001/report.json)通过。重放有 19 个成功结果和 12 个预期拒绝；成功结果的输出、事件、周期和资源记录均与 release 相同。Spike 验证真实 RV64 指令及 fflags，不等于 Rocket 系统集成。
- [旧完整 Llama2 编译兼容检查](../artifacts/tagged/mask-control-safety-001/legacy-compilation.json)重编译全部 6,181 个节点，与正在运行的冻结程序完全相同；这是编译一致性，不是额外一次完整推理。

开发记录保留：早期 canonical 指令检查误用了 JsonCpp 数值存储类型的整体相等；改为先校验 UInt32 范围再逐字比较。浮点特殊值测试改用真实二进制资产，避免非标准 JSON Infinity；测试目录 001 曾因父目录未建立发生 fixture 错误，最终 002 全部通过。这些记录未覆盖，也未计为通过证据。

本次仅为原 BERT 清单中四种控制入口提供实际路由。完整 BERT 仍缺 LayerNorm/GELU、索引/生成、多输出/布局、单文件权重与 QA 输出/后处理等完整路径；参考报告中的旧覆盖数字不被重标为模型已执行。v2 控制描述符尚未接入真实主机 C/RV64 系统 ABI，当前系统 lowering 会拒绝这些未登记 kind。所有所需模型的 ME0–ME3、推理性能和 RTL 门槛仍未通过。

复现安全与 ISA 检查（先按测试入口构建 release 及指定的 sanitizer 二进制）：

```bash
PYTHONPATH="$PWD/src:$PWD" .venv/bin/python -m scripts.verify_mlx_mask_control \
  --tests artifacts/tagged/mask-control-tests-002 \
  --asan-port build/mask-port-asan/control-external-memory \
  --asan-graph build/mask-control-asan/mlx-ready-graph \
  --asan-physical build/mask-control-asan/mlx-physical-model \
  --spike /path/to/spike --output artifacts/tagged/mask-control-safety-NEW
```
