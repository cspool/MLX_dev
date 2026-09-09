# 完整 BERT 系统图：source-group、split 与真实 CPU QA 输出

2026-09-09。现代系统图接口已接入，并通过实际 Rocket 小图和独立审计；完整 BERT 配对 Rocket 尝试已启动，**尚未终态，不是整模通过**。原有 Llama2/BERT 运行不被替换或重标。

## 实现

- C++ 地址生命周期规划支持 v2/v3/v4 程序。split 的每个实际输出分别持有同一存储分配；内部 tuple 容器仅作为短生命期编译标识，不被当作可读 Tensor 或长期存储所有者。全部 QA 输出和输入元数据保留到 CPU 读取。
- graph ABI v3 将降级步骤槽和原始算子组分开。组表覆盖全部步骤，每完成一个步骤更新所属组计数；全部步骤结束才计为该原始算子完成。配对执行可跨步骤重排，但数据依赖和 guard 依赖均保留。
- 补齐 LayerNorm/GELU 所需 `sub/div/maximum/exp` 系统向量描述符，使用 wire v2；长度仍为 4,288 bytes，复用已有微指令与有限 RF/SPM/ROM，不增加硬件资源。
- QA 结果由实际 RV64 CPU 从系统内存读取双 logits、mask 和 UTF-8 offsets，再执行 C span 选择。参考结果只参与后续检查，不进入计算输入。
- CPU 输出实际 logits 位模式、span 和周期边界；Python 独立重算 span、核对原始输出及预绑定参考。`rdcycle` 的图执行和后处理区间不含后续参考检查/输出打印，也不含启动前预装载；Spike 计数不能当作 Rocket 时序。

旧 graph v1/v2、control v1、memory/vector v1 编码保留。v3 的兼容字段 `source_calls/source_count` 仍指降级步骤槽，报告显式增加 `source_count_basis=lowered_stage_slots`、`lowered_calls` 和 `original_source_calls`。进度记录中的阶段 source ID 须结合程序的 `origin_source_operator_id` 解释，不能把高编号当作已完成层数。

## 验证边界

| 证据 | 结果及范围 |
| --- | --- |
| `modern-system-regression-002.xml` | 227 项回归通过；含现代图、tuple/QA、原有系统图、向量 wire 与系统审计 |
| `modern-system-qa-tests-002.xml` | 28 项检查通过；实际 RV64 QA 读回/后处理及每个测试参数下 14 个 CPU 组状态场景 |
| `bert-system-audit-tests-001.xml` | 7 项整模审计比较器检查通过，区分容忍误差、错误答案、非有限值和指纹变化 |
| `modern-chipyard-002` | 两个实际 Rocket 配对 QA 小图通过，共 45 个降级步骤、15 个原始源调用、23 个设备窗口；实际输出已导出 |
| `modern-chipyard-audit-002` | 两例均完成 ELF/任务/输入重新编译一致性、实际输出、源组和访存排空审计 |
| `bert-system-audit-reject-small-001` | 用完整原生 BERT 参考替换小图参考的尝试被拒绝，不能拼接成完整系统通过 |

上述小图分别是 tuple QA 和含 GELU 源组的 QA，不是缩小模型外推。完整模型采用全部原始算子与真实参数。

## 完整 BERT 镜像及运行

`prepare_mlx_bert_system_image.py` 从原始完整 inventory 重新编译四类 scheduled 后端，与已绑定的完整配对程序逐项相同；数值图与已完成原生微程序参考一致，但不转移其执行结论。

| 完整 BERT 配对系统项 | 实际准备值 |
| --- | ---: |
| 参数 / 参数张量 | 108,893,186 / 199，FP32 |
| 输入 / 层数 | 28、64、64 token 三组；每组 12 层 |
| 原始调用 / 降级步骤 | 1,141 / 3,046 |
| 闭合配对 / CPU 任务 / 设备窗口 | 693 / 2,749 / 2,094 |
| 命令字节 | 10,755,520 |
| 完整初始化资产字节 | 435,583,244 |
| 地址计划需要字节 | 441,633,344 |
| 阵列 / 上下文 / 系统 RAM | 4×4 / 每 PE 2 / 16 GiB |

`artifacts/tagged/model-e2e/bert-paired-rocket-full-001/bert-paired-resident` 已启动真实 Rocket/RoCC/HellaCache 完整尝试。程序、来源、二进制、profile 与输入保持冻结。该目录的运行中状态与只读进度不是验收证书。

终态后使用 `scripts/verify_mlx_bert_system_model.py` 独立验收：要求完整实际系统终态与 CPU 输出、原始 inventory 重编译、全部参数/层/输入覆盖、原生数值图一致、实际 logits/span 独立重算及框架容差通过。未终态、仅原生通过或仅小图通过均不能放行。

整模系统运行成本仍很高，之前宿主 `-O3` 试验未改善这一点；不根据小图估计整模推理性能，不用合并/跳过未验证的 CPU 或缓存边沿冒充原系统结果。P0 未关闭，GPU 新实验和 RTL/PPA 继续后置。
