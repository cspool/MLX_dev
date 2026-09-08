# 后续模型材料前置审计

本批包含[模型材料/入口缺口说明](../../../docs/mlx-next-model-materials.md)、只读检查器、[13项测试](../model-material-tests-002/regression.xml)及三份来源绑定记录：

- [InternLM2](../model-materials-002/internlm2.json)：227个真实BF16张量、7737708544个元素；当前MLX编译dtype表无BF16，未静默转换。
- [BERT QA baseline](../model-materials-002/bert-qa.json)：199个F32张量；需要专属输入、QA输出和算子采集入口。
- [BERT structured k1](../model-materials-002/bert-structured-k1.json)：199个F32张量，末层Q/K/V为factors参数；不能依赖相似config做非严格vanilla加载。

[manifest.json](manifest.json)绑定本批9个文件、218180字节（不含本索引和manifest）。不发布模型权重或训练参数pickle；检查不执行自定义模型源码、不加载模型推理、不采集forward。001是来源/元数据检查完善期间的开发记录，002是最新绑定结果。

这不是作者hybrid身份、算子覆盖、模型执行或系统验收。精度/结构选择仍需明确，全部模型与输入、正确性、性能及RTL门槛未关闭。四个既有Llama2尝试继续保持冻结版本。

