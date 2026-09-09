# BERT双输出与C答案选择的端到端执行

新协议`mlx_tensor_semantics_v4 / mlx-qa-result-v1`显式绑定`start_logits`和`end_logits`，不借用生成模型的`token`字段。两者必须是同一forward真实计算出的F32 `[1,sequence]`张量。可与source_groups和split视图合约组合，旧Llama2数值程序不变。

Tokenizer的context mask与UTF8字节offset作为输入资产绑定，不能来自参考答案或参考logits。编译器核对真实输入binding、padding和segment；C++再次验证shape、offset顺序和UTF8边界。两个结果与metadata保留到实际读取完成，串行/配对物理入口通过同一响应驱动端口读取并计数：每个长度n的用例为5n笔、25n字节。

`simulator_ext/control_model/qa_span.c`使用实际双logits，枚举同一连续context内、不超过30token的跨度，以F64计算两个F32分数之和，严格大于时更新，保持字典序首个平局结果。拒绝NaN/Inf（包括被mask的位置），依据实际读取的offset从输入context取答案。没有Python目标计算或golden中间值。该C函数当前在原生宿主执行，CPU周期未建模；尚不声称在真实Rocket运行。系统ABI明确拒绝v4，后续需连接真实CPU软件与新算子ABI，RTL顺序不提前。

## 完整BERT首次端到端结果

[验收归档](../artifacts/tagged/qa-publication-001/manifest.json)包含完整程序、执行源码快照、实际双logits、参考与比较、回归和安全证据，不含权重、宿主大二进制或运行中的周期结果。

`artifacts/tagged/model-e2e/bert-qa-native-full-001`已退出0。严格加载本地dense QA baseline的199个F32张量、108,893,186参数，三组用例各完整执行12层，原始1,141个调用降级为3,046步。每个实际源调用均有对应后端，BLAS/功能回退/Python或GPU目标计算均为0。

| 用例 | 序列长度 | 实际C选择答案 | 参考答案 | 双logits最大绝对误差 |
| --- | ---: | --- | --- | ---: |
| normal | 28 | Zurich | Zurich | 7.153e-6 |
| padding | 64 | Zurich | Zurich | 7.153e-6 |
| changed-context | 64 | Oslo | Oslo | 9.537e-6 |

运行前固定条件为`abs(error) <= 0.005 + 0.005 * abs(reference)`，全部312个F32 logits通过；不要求框架逐位相同。原始参考是CPU PyTorch eager F32，而非GPU或另一个硬件模拟器。三组答案均由模拟器实际输出驱动。宿主运行352.14秒仅记录为复现信息，不是MLX性能。

23项QA组件检查以及与split/LayerNorm/旧原生与周期入口、性能计数合计138项回归通过，覆盖中文UTF8、padding、真实输入扰动、回压重试、结果篡改与错误协议拒绝；另有19次ASan/UBSan/LSan重放通过（7个成功、12个预期拒绝），实际输出、事件、周期及资源报告与release一致。这不是作者hybrid或真实Rocket系统通过。完整串行/配对周期运行已分别启动，用上述实际微程序结果检查周期调度不改变数值，随后生成[性能对照](mlx-native-inference-performance.md)。
