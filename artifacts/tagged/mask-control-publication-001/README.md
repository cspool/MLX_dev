# 原生 C++ mask/guard 控制路径发布

本批发布[Boolean 比较、AND、ALL 与分支守卫](../../../docs/mlx-mask-control-guards.md)的编译、RV64 叶程序、响应驱动 C++ 实现及图依赖约束。不修改四个冻结的 Llama2 运行，不包含权重、宿主大二进制、构建目录或活动完整模型输出。

- [444 项回归](../mask-control-tests-002/regression.xml)通过，包含真实物理值、回压、空/尾部归约、正常/padding 守卫图、错误 mask 和依赖破坏拒绝。
- [31 次 ASan/UBSan/泄漏检查重放及 28 组独立 Spike 对照](../mask-control-safety-001/report.json)通过；19 次成功的输出/事件/周期相同，12 次预期拒绝没有成功报告。错误 guard 实际读 1 byte、写 0 byte。
- [旧完整 6,181 节点 Llama2 程序](../mask-control-safety-001/legacy-compilation.json)重编译一致；这不是新一次完整推理。

[manifest.json](manifest.json)绑定本批 255 个文件、4301198 字节（不含本索引和 manifest 自身）。包括新组件的真实输入、结果、预期拒绝记录以及 ISA ELF，不把开发失败目录计作通过证据。

此处的 v2 是控制叶程序合约，尚未接入实际 Rocket 主机 ABI。完整 BERT/其他模型、系统输出、推理性能及 RTL 仍未验收；图 guard 不匹配会停止，不会静默回退或自动执行另一分支。
