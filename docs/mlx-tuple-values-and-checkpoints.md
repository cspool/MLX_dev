# split 多输出与单文件 checkpoint 编译路径

本增量继续补齐完整 BERT 的前端和原生 C++ 路径，不修改三个仍在运行的完整尝试，也不把已归档的旧 Llama2 串行结果转用于当前实现。新增代码未修改 RTL、模型权重或精度。

## 一个源调用，多个真实结果值

`aten.split.Tensor` 现在由一个 `split` 源节点路由到 C++ 内存模型。每个返回分块有独立的值ID、形状、stride、offset和最后使用位置；它们共享源存储，不把split展开成重复计数的源调用，也不只保留第一个结果。

含split的程序使用 `mlx_tensor_semantics_v2` / `tuple_view_outputs_v1`，split内存计划为 `mlx-memory-plan-v3`。普通程序继续使用v1；旧完整6,181节点Llama2重新编译与冻结程序一致。这里的格式版本不是实际Rocket host ABI版本，也不是新增PE opcode。

节点主ID仅是内部存储句柄，`split_outputs`中的 `vN:0`、`vN:1` 等才是下游可用的结果。编译器和C++均拒绝把内部句柄作为Tensor操作数或导出结果。C++根据实际源布局重新推导全部分块，核对数目、规范ID、dtype/shape、offset、stride、存储extent与root；缺少第二个输出、错误偏移或伪造所有权不能通过。

原生张量入口、逐算子串行物理执行器和ready DAG会在同一个源完成时发布全部结果。ready DAG把每个结果映射回同一生产者，正确处理同一消费者同时使用两个分块；未使用结果及时回收，仍有消费者的别名继续持有原allocation。配对编译、物理/ready证据检查也保留这些值定义和依赖，不把子结果当成无生产者的初始输入。

split属于经过检查的view操作，没有额外数据搬运或PE计算；不扩大RF/SPM/ROM。支持整数split size、合法正/负axis、尾部分块、非连续源、单分块及空轴；size=0只允许空轴。普通单结果观察接口暂不支持直接选择tuple成员，会明确拒绝观察split源，避免把内部句柄误报为原算子输出；实际消费者及最终输出仍能通过C++端口读取、比较。

## 完整参数绑定不能靠文件名猜测

编译器同时支持原分片 `model.safetensors.index.json` 和单文件 `model.safetensors`，使用真实头部的dtype、shape、数据偏移及字节数生成mapped-file资产。索引分片必须存在并位于模型目录内，引用缺失Tensor、错误dtype/shape/extent或未绑定指纹会拒绝；不执行模型目录中的Python代码。

参数名迁移必须由 `checkpoint_name_by_model_parameter` 显式给出完整双射。例如BERT已验证的LayerNorm `gamma/beta → weight/bias`：IR记录原checkpoint参数名和模型参数名，但不改Tensor内容或精度。重复映射、缺项、未知目标和冲突文件指纹被拒绝，没有非严格加载或模糊名称匹配。

真实BERT baseline的单文件SHA已再次匹配参考清单。用其完整清单编译，目前能够完成索引/名称绑定并推进到 `aten.layer_norm.default`，随后明确拒绝；不再因缺少分片索引文件而停止。这是编译前置进展，不是完整BERT执行。

## 验证结果和仍待完成的工作

[361项回归](../artifacts/tagged/split-values-tests-001/regression.xml)通过，其中新增24项split和9项单文件参数测试。覆盖全部返回值、重排消费、非连续布局、空/单分块、原名释放后的生命期、未使用分块回收、正常串行/ready/配对执行、损坏描述符/版本/内部句柄拒绝，以及真实小型F32测试参数经mapped-file进入C++计算。

[32次ASan/UBSan/泄漏检查重放](../artifacts/tagged/split-values-safety-001/report.json)通过：20次成功输出和完整报告（包括事件/周期/资源）一致，12次预期拒绝没有成功报告。旧完整Llama2编译兼容性和实际BERT checkpoint身份也在该检查中绑定。开发期间单文件测试有一次将 `st_size` 属性误写为函数的fixture错误，修正后最终361项通过；旧开发记录未覆盖或计作通过证据。

对BERT参考的1,141次调用做名称/属性预检，显式未登记的入口剩LayerNorm 75次、GELU 36次；其他1,030次**不是已完成的模型编译/执行覆盖数**。完整BERT仍需这两个数值算子、QA输出/后处理协议和真正端到端执行。实际系统host ABI尚未登记tuple结果，当前会在任务生成前拒绝新程序，不能静默消除split后丢失第二个输出。

原GPU数值门槛、三个仍在运行的完整尝试、其他模型/作者变体以及推理性能/RTL门槛继续独立保留。

```bash
PYTHONPATH="$PWD/src:$PWD" .venv/bin/python -m scripts.verify_mlx_tuple_values \
  --tests artifacts/tagged/split-values-tests-001 \
  --asan-memory build/split-memory-asan/memory-external-memory \
  --asan-graph build/split-graph-asan/mlx-ready-graph \
  --asan-physical build/split-graph-asan/mlx-physical-model \
  --asan-tensor build/split-graph-asan/model-storage/tensor-model/mlx-tensor-semantics \
  --bert-inventory artifacts/tagged/bert-qa-reference-003/inventory.json \
  --legacy-program /path/to/frozen-llama2-program.json \
  --legacy-inventory /path/to/frozen-llama2-inventory.json \
  --output artifacts/tagged/split-values-safety-NEW
```
