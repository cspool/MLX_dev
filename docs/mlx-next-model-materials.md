# 后续模型材料与编译入口缺口

本页补充模型级 ME0/ME1 的实际前置条件，避免把“目录里有权重”当作“已能在 MLX 执行”。本轮只读本地 checkpoint 元数据，不实例化模型、不执行自定义模型代码、不采集前向算子，不替换作者模型身份，也不改变四个正在运行的 Llama2 尝试。

## 已核对的本地材料

`scripts/audit_mlx_model_materials.py` 对 PyTorch分片使用 `weights_only=True, map_location='meta', mmap=True`，逐一检查键集合及索引数据字节数；对 safetensors 只读头部，检查形状/类型/区间及重叠。本次读取的config、checkpoint索引/分片、tokenizer文件和目录中的自定义Python源码均绑定摘要，前后核对不变；未读取训练参数pickle来执行模型构造。

| 材料 | 张量数 / 元素数 | 实际精度 | 直接接入障碍 |
| --- | --- | --- | --- |
| [InternLM2 本地 checkpoint](../artifacts/tagged/model-materials-002/internlm2.json) | 227 / 7737708544 | 全部 BF16，15475417088 数据字节 | 当前 dtype 表没有 BF16；现有参考入口只登记 Llama2 |
| [BERT QA baseline](../artifacts/tagged/model-materials-002/bert-qa.json) | 199 / 108893186 | F32 | 需独立 QA 输入/输出及算子采集入口，不是 Llama2 的 logits/token 合约 |
| [BERT structured-distilled h38/k1](../artifacts/tagged/model-materials-002/bert-structured-k1.json) | 199 / 107676674 | F32 | 需恢复与 checkpoint 对应的结构化构造，再严格加载和采集 |

这些是 checkpoint 张量元数据，不是已加载模型的 `named_parameters()` 核验，更不是推理结果。InternLM2 两个文件分别约9.97GB和5.51GB，实际包含149/78个索引张量，不是 LFS 指针。配置为32层、hidden4096、intermediate14336、32个query heads/8个KV heads、vocab92544；融合QKV权重实际为 `[6144,4096]`，不能套用Llama2的三个独立投影或其模型签名。

InternLM2 自定义源码顶层直接引用的已安装 torch/transformers/einops 符号检查没有发现缺失，但这只验证导入符号，不证明模型构造、cache API 或 forward 兼容。源码没有被执行；可选 flash-attention 路径未验证。

BERT 两份 config 都写 `BertForQuestionAnswering`、12层、hidden768，但权重键并不相同：baseline 有 LayerNorm `gamma/beta` 命名，结构化版本为 `weight/bias`；结构化 k1 还把第12层 query/key/value 的 `.weight` 替换为 `.factors`。不能仅凭相同 config 建立vanilla模型并忽略缺失/额外键。命名迁移和结构恢复必须分别绑定并验证，不能混为一次非严格加载。

## 弥补目标及顺序

1. **身份和精度先固定。** 作者 hybrid、公开基座、chat、本地重建及QA任务分别登记。InternLM2 的 BF16 不能按 FP16 位模式读取，也不能静默转换后继续宣称验证了原 BF16 模型。需依据 MLX 架构/数值约束建立受支持的精度路径，或明确选择并验证转换合约；本轮未替用户作该选择。
2. **建立专属参考入口。** 当前 `capture_mlx_model_reference.py` 明确限制完整 Llama2-7B签名。InternLM2 需绑定其自定义代码、融合QKV/GQA布局、tokenizer、cache和生成策略；BERT QA需绑定问题/上下文、padding/mask和start/end输出，而不是套用单token生成检查器。
3. **采集后才认定算子覆盖。** 静态 config 中的GELU/LayerNorm等提示了BERT所需路径，但本页没有实际forward清单，不能宣称算子已齐全。每个实际调用和多输出都要路由到已验证MLX后端或底层指令，缺少的结构、精度或语义必须明确拒绝，禁止框架回退后计作MLX计算。
4. **完整执行再验收。** 保持完整模型、权重和输入语义，在C++、系统中分别完成端到端结果及状态检查；所有所需模型/输入通过后才评估推理性能，最后进入RTL。

ViT/FABNet的完整拓扑、权重、预处理及任务头仍未绑定；不能用随机/单层模板替代。作者模型与本地变体的选择仍沿用此前已提出的待确认项，本检查不构成替代授权。

[13项检查器测试](../artifacts/tagged/model-material-tests-002/regression.xml)覆盖BF16元数据保持、禁止执行自定义代码、LFS指针/缺分片/错误索引和数据长度拒绝、层覆盖、结构化因素识别及非法safetensors范围。报告中的 `model_loaded_for_inference`、`operator_trace_captured`、`author_hybrid_identity_verified`、`full_model_execution_verified` 和 `inference_performance_eligible` 均为false。
