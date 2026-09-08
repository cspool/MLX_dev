# LayerNorm：显式 C++ 原语序列与原始源分组

本增量为 `aten.layer_norm.default` 提供编译到现有C++向量/搬运路径的计算序列。Python只做降级和元数据检查，不计算中间Tensor或推进模拟周期；所有中间值均由C++实际计算、存储和读取。没有修改模型权重/精度、三个冻结运行或RTL。

## 数值模式与硬件约束

模式 `mlx-layernorm-shifted-fp32-v1` 支持FP16/FP32输入、一个非空的最后归一化轴、可选gamma/beta和非负有限epsilon。FP16输入和需要的仿射参数先显式转换到FP32，最后再转换回输出dtype。BF16、多归一化轴、非法shape/精度/属性明确拒绝。

每行先取实际首元素作为anchor，执行下列FP32顺序；每次算术、归约和转换都按已登记原语舍入：

```text
shifted = x - x[..., 0:1]
centered = shifted - mean(shifted)
variance = mean(centered * centered)
normalized = centered * (1 / sqrt(variance + epsilon))
output = normalized * gamma + beta   # 缺省的仿射项省略
```

减去anchor在实数算术下不改变LayerNorm函数；这里明确采用这一FP32计算顺序以降低大偏移输入的消减误差。mean沿用16-lane、相邻配对与有界carry栈的既有合约；rsqrt仍是SQRT后DIV。新登记的减法入口使用已有SUB opcode，未增添浮点功能单元。

这是**本项目重建数值模式**，不是论文公开的LayerNorm电路或GPU内核顺序。各个阶段分别使用现有有界模板，均保持32-word ROM、既有RF/SPM/端口约束。当前是未融合的合法降级：临时Tensor位于模型物理内存，分配、读写和目标周期均实际发生；不能把这些成本隐藏为免费局部寄存器，也不声称具有融合LayerNorm的效率。

## 原始算子数与实算步骤分开

含复合降级的程序使用 `mlx_tensor_semantics_v3` / `source_groups_v1`。普通v1、tuple-view v2程序保持原格式。FP32且带gamma/beta的一个LayerNorm源对应12个步骤，包含anchor view、shift/mean/center、variance、epsilon/rsqrt和仿射；FP16还会有显式转换步骤。

`source_groups`记录原始operator及ID、原始精度/形状/epsilon、全部降级ID、阶段名和最终结果绑定。C++核对每个阶段恰好归属一个源，拒绝缺失/重复/改名、错误阶段kind、错误原始精度/shape/epsilon、伪造仿射精度或用direct标签绕过LayerNorm。

v3的低层事件保留兼容字段 `source_operator_id`，其域为**降级ID**，并明确增加 `lowered_operator_id`、`origin_source_operator_id`、`lowering_stage` 和阶段名。`executed_lowered_calls`报告实算步骤，`executed_source_calls`报告原始源；原始源完成记录由全部实际阶段事件汇总，不用标签或少数步骤代替完成。带时序的源区间取阶段实际最早start和最晚complete，不将重叠阶段周期相加。

v3中的低层窗口/观察ID同样按降级ID使用，不能直接把原始source ID当作最终算子输出的观察点；应通过分组查到对应阶段。tuple-view仍遵守其显式结果约定，内部owner不能作为Tensor输出。当前系统host ABI会拒绝v3分组程序，尚未完成真实主机分组调度。

## 当前验证与数值差异

[368项最终回归](../artifacts/tagged/layernorm-tests-005/regression.xml)通过，新增24项LayerNorm/源分组测试；覆盖FP16/FP32、仿射有/无、17/33/768宽度，以及实际BERT形状 `[1,28,768]` 的完整LayerNorm操作；另覆盖普通张量、串行物理、共享配对、LayerNorm后split联合图和元数据破坏拒绝。它们是完整算子/组件图测试，不是缩小模型代替BERT验收。

[19次ASan/UBSan/泄漏检查重放](../artifacts/tagged/layernorm-safety-002/report.json)通过：8次成功输出、实际argmax及完整事件/周期/资源报告一致；11次预期拒绝无成功报告。源码、输入、独占测试二进制和运行库前后核对摘要。早期记录保留：新增sub曾被旧测试按单输入构造，另有一项拒绝测试预期的错误子串不符；最终测试005已修正并扩大到源精度/shape/epsilon、仿射精度及源标签检查，较早测试/安全记录不作为本批最终验收。

目标输出与独立控制的shifted-FP32参考逐位一致；普通数值范围下也满足测试预先指定的框架误差条件。需要明确保留一个反例：大偏移输入专项测试中，当前结果相对FP64数学参考的最大误差约 `2.09e-7`，但相对该框架运行的结果最大差约 `0.3466604`。这说明浮点顺序不等价，**不是覆盖框架门槛的理由**；记录保留在安全检查报告中，未以“更接近FP64”宣布完整模型通过。

旧Llama2的完整6,181节点程序仍要求重新编译一致。实际BERT清单的75个LayerNorm可展开；1,141个原始源变为1,966个降级节点，初始bindings不变，没有注入中间参考值。完整BERT编译目前在 `aten.gelu.default` 明确拒绝，尚未产生完整可运行BERT程序。名称/属性预检仅剩36次GELU显式拒绝，不代表其余源已完成模型级执行。

后续仍需GELU的有依据数值路径、QA结果/后处理协议、完整BERT实际运行，以及真实系统分组ABI。旧公开Llama2的原GPU门槛失败、三个活动完整尝试、其他模型/作者变体与性能/RTL门槛继续独立保留。
