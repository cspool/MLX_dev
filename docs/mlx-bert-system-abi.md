# BERT 系统入口增量：布尔控制与 memory v2

本次补齐完整 BERT 进入系统编译路径的一部分接口。**不是完整 BERT、配对整模或 Chipyard 整模通过。** 现有完整运行没有重启或替换二进制，RTL 未修改。

后续更新：下文列出的 source-group、split 生命周期与 QA 系统输出已在[现代系统图接口](mlx-modern-system-graph.md)中接入并通过实际 Rocket 小图；完整 BERT 配对 Rocket 尝试已启动，仍待完整终态及整模审计。下文缺口清单保留为此原始增量的历史状态。

## 已实现并验证

### RV64 主机控制 v2

`physical_host/control_runtime.c` 增加 `ge`、Boolean `bitwise_and`、`all`、`guard`，编译端同步生成版本 2 的 640-byte 主机命令；旧操作仍使用版本 1 和原有编码。它们是 CPU 执行入口，不是新增 MLX PE opcode。

`all` 从 CPU 可寻址的实际 Boolean 数据逐元素读取，支持空输入恒真、非连续/广播视图和非规范非零字节归一化。`guard` 读取实际单元素条件，与编译时声明的分支特化条件比较；不符时返回 `MLX_HOST_GUARD_FAILED`，不写入成功输出，也不让图调度器继续执行后续设备任务。浮点 `ge` 保留 NaN/舍入模式检查，整数比较不先转换到浮点。

配对图的 CPU 依赖表和生命周期计算现在包含 `control_dependencies`，即使后续节点读取的是独立资产，也不能漏掉前面的运行时 guard。

### memory wire v2

增加 `advanced_index`、`new_ones`、`squeeze` 的完整编码/解码，接到现有 C++ 响应驱动搬运后端、外部时钟控制器和 Spike 系统插件。主机任务运行时识别新描述符长度。

- v1 仍为 15,872 bytes；v2 为 15,936 bytes。公共字段/operand 偏移保留，扩展的是有界控制模板存储，不是 PE RF、SPM 或搬运资源。
- v2 最多容纳 12 个模板 word，可表达 8 维索引所需的 8 次索引读取加 load/convert/store；实际模板仍严格检查，不能漏读索引或用表项数代替实际次数。
- `advanced_index` 的索引和值来自真实总线响应，保留负索引、广播、SSA 重用和类型；越界时排空但不发布成功。
- `new_ones` 保留显式/继承 dtype，实际写入输出，不读取原型 Tensor 的数值。测试故意不初始化原型数据，防止隐藏读取。
- `squeeze` 是经过形状/存储身份核对的视图，不能被计为发生数据搬运。

## 验证范围

| 证据 | 已完成内容 |
| --- | --- |
| `host-mask-control-002/report.json` | 实际 RV64 ELF 执行 71 个控制场景，含 31 个 v2 场景、两次 guard 不匹配；36 个正常命令与 Python 编码逐字节对应；65,536 个 FP16 转换位模式检查保留 |
| `system-bert-abi-regression-002.xml` | 99 项主机 ABI、memory wire、外部时钟控制器及新增语义回归通过 |
| `system-bert-abi-graph-regression-001.xml` | 29 项配对图、实际 RV64 生成图及系统 profile 回归通过 |
| `system-memory-v2-safety-002/report.json` | 34 个总线作业和 26 个解码作业经 ASan/UBSan/LSan 重放，与 release 的报告、终态及预期拒绝一致 |
| `system-bert-primitive-chain-002/regression.xml` | 三个实际 RV64 组合链测试通过：正常、数据扰动、分支不匹配 |

组合链包含 8 个源节点：CPU `all→guard→arange`、设备 `new_ones→squeeze→advanced_index→add`、CPU `argmax`。正常与扰动分别完成 4 次 CPU 控制和 3 个设备窗口，输出在 ELF 内逐字节比较；分支不匹配时后续设备启动数为 0。它是覆盖 BERT 所需语义的测试链，**不是缩小版 BERT 结果，更不是完整 BERT 证书**。

这些组合链使用 Spike CPU 与 C++ 设备插件；外部时钟测试使用受检总线驱动。当前尚未为这些新增接口完成独立 Rocket/Chipyard 重构建和整模运行，不能将两个测试环境的结果拼成 Chipyard 通过。

## 尚未关闭的系统缺口

1. `source_groups`：1,141 个原始源调用与 3,046 个降级步骤须保留完整分组、任务表与独立完成核对。
2. `split`：多输出别名、物理生命周期和下游绑定须进入系统地址规划，不能把内部 tuple owner 当普通输出 Tensor。
3. QA：双 logits、输入 mask/UTF-8 offsets、真实 CPU span 后处理、系统输出检查与镜像审计还需接入，不能继续使用 Llama2 的 logits/token 固定格式。
4. 以上内容完整后，构建独立 Chipyard 系统版本，在完整参数/模型/输入上实际执行并验收。现有 `compile_graph` 对 v4/source-group/split 的拒绝仍保留。

用户给出的[DFU/simict 模拟器卡片](mlx-dfu-simulator-reference.md)用于核对“编码—执行—状态—后继消费”的完整修改关系及运行产物组织；没有复制其内部实现或资源常数。优先级仍为 P0 主线 → GPU → RTL/PPA。
