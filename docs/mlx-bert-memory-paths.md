# BERT 所需的 C++ 索引、常量与布局路径

本增量补齐[完整 BERT QA 清单](mlx-bert-qa-reference.md)中的 `index.Tensor`、`new_ones.default`、`squeeze.dim` 和 `to.dtype_layout`。新增路径使用 C++ 同步或响应驱动的内存执行器；Python 只生成/检查计划，不推进周期或代算张量。四个冻结的 Llama2 尝试、权重、精度和 RTL 未修改。

## 语义、资源与指令边界

新增 kind 使用 `mlx-memory-plan-v2`；旧 kind 的 v1 计划和行为保持不变。它是本项目重建的**搬运控制器合约**，不是论文公开的 PE opcode 或 RISC-V 指令编码。本次不添加新的浮点功能单元。

| 来源入口 | C++ 路径 | 实际行为与限制 |
| --- | --- | --- |
| `aten.index.Tensor` | `advanced_index / indexed_nd` | 对 source 的每个轴提供一个 I64 索引张量，广播后逐个读取实际坐标；支持负坐标、标量/空索引和非连续 source，拒绝越界、Boolean index、缺省/部分轴及不合法广播 |
| `aten.new_ones.default` | `new_ones / constant_one` | 新分配输出；每元素执行描述符常量 LOAD(1)、类型转换、实际 STORE。不读取模板张量 payload，也不把参考中间值当输入 |
| `aten.squeeze.dim` | 经检查的 affine view | 只移除指定的 size-1 轴；非 size-1 轴保持不变，scalar 仅允许 axis 0/-1。保留 offset、stride 和存储所有权，零搬运不是无条件复制消除 |
| `aten.to.dtype_layout` | 规范化到已有 `cast_device` | 检查 dtype/layout/device/copy/non_blocking/memory_format/pin_memory；只有计划证明可共享存储时才消除，否则由 C++ 执行转换或复制 |

索引范围为 source rank 1–8，必须完整索引全部轴；这覆盖实际 BERT mask 的两个 I64 索引，但不声称实现任意 ATen advanced indexing。输出形状由索引张量广播得到，dtype 与 source 相同，输出新分配为连续存储。运行时每个坐标经 I64 负索引规范化及范围检查，地址累积有溢出检查，最终通过 source 的真实 stride/offset 读取数据。

四个 64-bit 数据寄存器分别保存输入、转换结果、当前索引/谓词和地址累积，总量仍为 32 bytes；staging 仍为 128 bytes、chunk 64 bytes，单事务及单转换在途。索引计划最多为 8 个 `LOAD_INDEX` 加 `LOAD/CONVERT/STORE`，共 11 个描述符词，不扩大 PE RF/SPM/ROM，也不把它们混称为 PE 指令。

`new_ones` 的常量来自算子定义，并在目标模型中生成与写入；即使源模板内容改变，也不应改变 ones 输出。输出不允许与输入共享 allocation。空输出仍验证完整计划，但没有元素可读写。pinned-memory、未登记 dtype/layout 和不一致形状明确拒绝。

`to.dtype_layout(copy=True)` 不能作为别名；dtype 或参考设备变化、contiguous 要求等继续由已有计划证明。标志必须符合类型约定，不因名字熟悉而忽略未知属性。索引/生成/squeeze 必须选择 `planned` 或 `scheduled`，不能回退到通用功能算子。

## 实际验证

[345 项回归](../artifacts/tagged/bert-memory-tests-002/regression.xml)通过，其中新增 53 项覆盖这些入口及其 C++ 路径：

- 物理端口中的索引值与宿主占位值不同；核对 I64 大整数不经 FP32 截断、负坐标、广播、rank 1/8、非法 rank 9、越界及空/标量输出。
- 非连续 source 在原始 SSA 名称释放后仍可正确索引；squeeze 的别名也在原值释放后保持有效，强制复制有真实读写。
- FP16/FP32/I64/Boolean 的 ones 使用新存储，实际读模板 payload 为 0，写入字节覆盖完整输出；错误 dtype/shape/pin/模板/别名计划拒绝。
- 从实际 inference-mode ATen 调用采集的组件图逐源编译，分别在原生串行和共享资源 ready DAG 完整执行，比较真实输出及 argmax；没有 BLAS、Python/GPU 或通用张量回退。该图不是缩小 BERT 的模型验收替代物。

[46 次 ASan/UBSan/泄漏检查重放](../artifacts/tagged/bert-memory-safety-001/report.json)通过：35 次成功输出与完整报告（含事件、周期和资源）一致，11 次预期拒绝不产生成功报告。源码、程序/选项、输入、二进制与动态库在执行前后绑定摘要。同一检查还重新编译旧 Llama2 全部 6,181 个节点，与冻结程序逐字段相同；这只是编译兼容性，不是新一次完整模型运行。

开发记录未覆盖：第一批新测试在非 inference-mode 下未观察到期待的高层 `to.dtype_layout` overload；改为与真实 BERT 参考一致的 inference-mode 后通过。`bert-memory-tests-001` 的337项检查先通过，随后补充 rank 容量、非法生成描述符与 pin 类型边界，最终002为345项通过。发布证据采用002。

## 尚未完成的模型与系统任务

对原 1,141 次 BERT 调用重新进行名称/属性预检，剩余显式拒绝为 LayerNorm 75 次、GELU 36 次、split 3 次。其余 1,027 次仅预检接受，**不代表已将这些调用全部编译/执行**；完整 BERT 仍需单文件 checkpoint/参数名映射、split 多输出 SSA、QA 输出/后处理协议，以及完整 C++ 和系统运行。

三个新 v2 memory kind 尚未登记到真实系统 wire/ABI，系统 lowering 会拒绝它们。`to.dtype_layout` 已规范化到原 kind，但也不能据此宣称新的 BERT 系统结果已验证。作者模型/其他模型身份、完整数值门槛、推理性能和 RTL 仍未验收。

```bash
PYTHONPATH="$PWD/src:$PWD" .venv/bin/python -m scripts.verify_mlx_bert_memory \
  --tests artifacts/tagged/bert-memory-tests-002 \
  --asan-memory build/bert-memory-port-asan/memory-external-memory \
  --asan-graph build/bert-memory-graph-asan/mlx-ready-graph \
  --asan-physical build/bert-memory-graph-asan/mlx-physical-model \
  --legacy-program /path/to/frozen-llama2-program.json \
  --legacy-inventory /path/to/frozen-llama2-inventory.json \
  --output artifacts/tagged/bert-memory-safety-NEW
```
