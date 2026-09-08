# BERT 所需原生 C++ 内存路径发布

发布[索引、fresh ones、squeeze 与 dtype/layout 转换](../../../docs/mlx-bert-memory-paths.md)的编译与 C++ 计划执行、测试和真实输出。没有修改四个冻结的 Llama2 尝试，不发布模型权重、宿主大二进制、构建或活动完整模型结果。

- [345 项回归](../bert-memory-tests-002/regression.xml)通过，其中53项为本增量测试。
- [46 次 ASan/UBSan/泄漏检查重放](../bert-memory-safety-001/report.json)通过：35成功/11预期拒绝；成功输出、事件、周期和资源记录一致，失败不生成成功报告。
- 旧完整6,181节点Llama2编译产物保持一致；仅为兼容检查，不是额外完整推理。

[manifest.json](manifest.json)绑定另255个文件/4140114字节，不含本索引和manifest自身。包括新组件的实际输入、输出和预期拒绝记录，不把开发失败或较早范围的测试记录算作本批最终验收。

完整BERT仍缺LayerNorm/GELU、split多输出和资产/QA输出协议；新的v2 memory kind尚未接入实际系统ABI。模型正确性、系统、推理性能及RTL门槛未通过。
