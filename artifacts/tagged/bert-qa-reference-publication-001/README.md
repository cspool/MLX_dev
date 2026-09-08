# 完整 BERT QA 前端参考发布

本批发布[完整 BERT QA baseline 参考入口](../../../docs/mlx-bert-qa-reference.md)、输入与测试，以及[实际完整前向清单/双输出](../bert-qa-reference-003/inventory.json)。不发布权重、环境、构建目录或四个仍在运行的 Llama2 输出。

- 完整 12 层、199 个 F32 参数张量、108,893,186 个参数元素；严格名称迁移/加载，参数位模式不变。
- 正常、padding、上下文扰动三组均执行完整参考前向，插桩前后 start/end 位模式一致；参考答案 Zurich/Zurich/Oslo。
- [38 项回归](../bert-qa-reference-tests-002/regression.xml)通过，覆盖严格加载、位模式比较、输入/层/span 合约和禁止覆盖历史目录。
- 实际 1,141 次调用、30 种 overload；11 种、140 次被当前入口拒绝。其余只是名称/属性预检接受，尚未编译或执行。mask 的运行时 Boolean 分支与 QA 后处理也必须进入实际目标路径。

[manifest.json](manifest.json)绑定本批 15 个文件、2740834 字节（不含本索引及 manifest 自身）。本批 `compiled_and_executed_calls=0`，不是 MLX 模型、系统或作者 hybrid 验收，不开放推理性能或 RTL。003 是最终位模式检查版本；开发期间 001/002 未覆盖、未计入本批正式证据。
