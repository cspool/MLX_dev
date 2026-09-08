# 完整 BERT QA 前端参考与实际编译缺口

`scripts/capture_mlx_bert_qa_reference.py` 为本地 dense BERT QA baseline 建立专属输入、输出和算子采集入口。这是模型到编译器的前置诊断，不是 C++ MLX 执行，不替代作者 structured/hybrid 模型身份，也不修改正在运行的四个 Llama2 实现、数值参考或退出门槛。模拟周期和目标计算仍须由 C/C++ 实现，之后接入系统，最后才进入 RTL。

## 完整身份和参考执行

使用本地 `bert-squad-baseline-run014/model.safetensors` 的全部 199 个 F32 参数张量、108,893,186 个参数元素：12 层、hidden 768、intermediate 3072、12 heads，完整 embedding 和 QA head。没有随机参数替代、缩层、缩宽、精度转换或读取训练参数 pickle。使用已安装 Transformers 的显式 BERT 类，不执行 checkpoint 目录中的自定义模型代码。

唯一迁移是 LayerNorm 参数名 `gamma/beta → weight/bias`。迁移必须一一对应；额外键、缺键、名称冲突、错误形状、非 F32 和非有限值均拒绝。随后严格加载，并以 FP32 位模式核对每个实际加载的参数；checkpoint 文件不被重写。结构化 `.factors` 不能通过此入口。

[输入定义](../configs/models/bert-qa-reference-cases.json)包含正常、padding、上下文扰动三组问题/上下文；不做输入截断。CPU FP32、eager attention、确定性算法、2 个宿主线程均显式记录，模型、tokenizer、输入、采集源码及实际模块来源文件前后核对摘要。这不是 GPU 数值对照或数据集准确率评估。

[最终参考证据](../artifacts/tagged/bert-qa-reference-003/summary.json)中，每组都分别执行无插桩和有插桩的完整前向。两个完整 start/end 输出以 FP32 位模式比较，包含正负零的区别；实际输出另存二进制和摘要。每组均核对所有 12 个 encoder 层各执行一次，以及 199 个参数张量均被实际调用消费。

| 用例 | 序列长度 | 实际源调用 | 参考答案 |
| --- | ---: | ---: | --- |
| normal | 28 | 353 | Zurich |
| padding | 64 | 394 | Zurich |
| changed-context | 64 | 394 | Oslo |

三组 start/end 的插桩等价检查全部通过。QA span 后处理限定 context token、start ≤ end、最多 30 tokens，以 start/end logits 的 FP64 和选最大值，同分按最早的 `(start,end)`；它在参考端执行，`postprocessing_compiled=false`。后续必须让实际目标输出进入对应的主机/设备后处理，不能把参考答案注入模拟器。

## 从真实调用导出的下一步目标

完整清单有 1,141 次调用、30 种 ATen overload。采集时的入口仅对 1,001 次接受名称/属性预检，**尚未将这 1,001 次编译并执行**；其余 140 次、11 种 overload 明确拒绝。全部 `compiled_and_executed_calls` 仍为 0。不能将预检成功数当作后端覆盖数。

后续[原生 C++ mask/guard 增量](mlx-mask-control-guards.md)已为其中四种 Boolean/比较/scalar 控制入口提供实际 RV64 叶程序和响应驱动执行，并以显式依赖约束后续图节点；错误特化条件不能继续回放。组件测试与 ISA 对照不构成 BERT 整图执行，下面保留参考采集时的完整缺口表及仍需满足的目标。

再补齐[原生 C++ 索引、常量与布局路径](mlx-bert-memory-paths.md)后，当前名称/属性预检的显式拒绝为 LayerNorm、GELU、split 三种，共114次；其余1,027次不是已编译执行的覆盖数。345项相关回归与46次安全重放只验证登记组件及兼容性，不重标本参考记录为MLX模型通过。

后续[split多输出与单文件checkpoint](mlx-tuple-values-and-checkpoints.md)又补齐了独立结果值及其C++发布/回收，并连接显式参数名双射。361项回归、32次安全重放通过；实际完整BERT编译在完成权重索引/名称绑定后停于LayerNorm，名称/属性预检仅剩LayerNorm和GELU共111次拒绝。仍未完成完整BERT执行或QA系统输出验收。

| 实际缺失入口 | 三组调用数 | 必须补齐的目标 |
| --- | ---: | --- |
| `layer_norm.default` | 75 | 完整最后轴统计、epsilon、仿射参数、数值顺序及有界归约/广播微程序；先定义并验证精度合约 |
| `gelu.default` | 36 | 本清单为默认 `approximate="none"`；按架构约束实现对应复合算子或有依据的模式，不能静默换为 tanh 近似 |
| `__and__.Tensor`、`ge.Scalar`、`all.default`、`is_nonzero.default` | 12 | Boolean/整数控制路径、实际归约值与主机分支；不是用记录中的 Python Boolean 常量替代目标执行 |
| `index.Tensor`、`new_ones.default` | 4 | 实际索引与常量生成、shape/dtype/范围检查和目标访存 |
| `split.Tensor`、`squeeze.dim` | 9 | 多输出 SSA/别名/生命周期、QA 双输出绑定；不能丢弃第二个输出 |
| `to.dtype_layout` | 4 | 明确类型、布局、设备、copy 属性及内存行为；禁止未校验属性消除 |

尤其是 `is_nonzero`：normal 的 source 10 返回 true，而 padding/changed-context 的 source 363/757 返回 false；两条分支的源调用数也不同。后续可使用有运行时守卫的特化图，或实现完整主机控制流，但守卫必须读取实际输入并检查条件。不能把参考采集到的条件当常量，从而让不同 mask 无条件重放同一路径。

此外，现有模型编译器按分片 safetensors 索引及 Llama2 单 logits/token 记录工作。BERT 需要单文件权重偏移与显式参数名映射、独立 QA 输出协议及后处理路由。已有矩阵/向量入口也须逐一完成这些真实 shape、F32、bias、mask 和尾部的全路径验证，不能只因操作名称已有就直接放行。

## 验证与复现

[38 项测试](../artifacts/tagged/bert-qa-reference-tests-002/regression.xml)覆盖原有清单/材料检查和新增严格加载、位模式比较、完整层计数、span 顺序与边界、非法输入以及拒绝覆盖历史目录。003 为加入正负零位模式检查后的最终参考；001/002 是开发期间记录，没有被覆盖，也未作为本批最终证据发布。

```bash
PYTHONPATH="$PWD/src:$PWD" .venv/bin/python -m scripts.capture_mlx_bert_qa_reference \
  --model /path/to/bert-squad-baseline-run014 \
  --cases configs/models/bert-qa-reference-cases.json \
  --device cpu --threads 2 \
  --output artifacts/tagged/bert-qa-reference-NEW
```

后续仍需 ME1 的全部实际编译路径、ME2 的完整 C++ 执行、ME3 的真实系统输出对照，以及作者模型/其他模型身份确认。`full_reference_model_executed=true` 与 `full_model_execution_verified=false` 同时保留；本证据不打开 ME4 推理性能或 RTL 门槛。
