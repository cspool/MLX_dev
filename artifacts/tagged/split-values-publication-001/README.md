# split 多输出与单文件参数绑定发布

发布[编译/C++多结果值及参数绑定](../../../docs/mlx-tuple-values-and-checkpoints.md)的代码、真实组件输入/输出和拒绝记录。包含仅6个F32参数的合成测试文件，不包含真实BERT/Llama2/InternLM2 checkpoint、宿主大二进制、环境或三个活动完整运行结果。

- [361项回归](../split-values-tests-001/regression.xml)通过，新增24项split及9项参数绑定测试。
- [32次ASan/UBSan/泄漏检查重放](../split-values-safety-001/report.json)通过：20成功/12预期拒绝，成功输出与完整事件/周期/资源报告一致。
- 旧完整6181源Llama2编译产物保持不变；真实BERT权重SHA与清单匹配，完整编译完成索引/名称绑定后在LayerNorm处明确拒绝。

[manifest.json](manifest.json)绑定另212文件/6818889字节，不含本索引与manifest自身。原生成报告保留原始格式以维持SHA，不把开发失败记录计为通过证据。

名称/属性预检仍拒绝LayerNorm75次和GELU36次；其他1030次不等于已经编译并执行。完整BERT/QA协议、系统tuple ABI、其他所需模型以及原GPU/性能/RTL门槛仍未通过。
