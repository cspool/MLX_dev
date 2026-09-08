# LayerNorm C++原语序列与源分组发布

本批发布[LayerNorm重建数值模式及源分组](../../../docs/mlx-layernorm-source-groups.md)的编译、C++执行记录、输入/输出和拒绝证据；包含合成测试参数，不包含真实模型checkpoint、宿主大二进制、环境或三个活动完整结果。

- [368项回归](../layernorm-tests-005/regression.xml)通过，包含24项新的LayerNorm/源分组测试。
- [19次ASan/UBSan/泄漏检查重放](../layernorm-safety-002/report.json)通过：8成功/11预期拒绝。成功输出、实际argmax、完整事件/周期/资源记录一致。
- 原始源和实算步骤分开计数，所有中间数据在C++中产生；旧完整6181节点Llama2编译不变，BERT完整编译推进到GELU后仍明确拒绝。

必须保留数值边界：大偏移测试相对FP64参考误差约2.09e-7，但相对框架差约0.3466604；未据此覆盖框架门槛。完整BERT、QA协议、系统source-group ABI、原GPU/性能/RTL门槛仍未通过。

[manifest.json](manifest.json)绑定另154文件/70618080字节，不含本索引和manifest本身；原始生成文件按SHA保留，不将较早开发记录计作最终通过证据。
