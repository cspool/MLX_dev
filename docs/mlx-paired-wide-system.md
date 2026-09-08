# v2 配对图的大地址空间与完整镜像验证

本阶段复用已有 C++ 配对调度器和 16 GiB 系统封装，没有修改 PE、调度器或其他加速器 RTL。执行仍由实际 Rocket/RoCC/DCache 驱动 C++ 后端，顺序保持“模拟器 → 系统集成 → 完整模型正确性 → 推理性能 → RTL”。

## 实际高地址系统用例

[pair-wide-rocket-001](../artifacts/tagged/pair-wide-rocket-001/report.json) 的五个登记用例全部退出 0，使用 4×4、每 PE 两个上下文及 `native-tensor-4x4-reference-v1` 参数。实际 RAM 为 `[0x80000000, 0x480000000)`，图窗口从 `0x181000000` 开始，高于 4 GiB。资产和 ELF 在时钟前按受检清单初始化；每个用例的 CPU 资产复制字节数明确为 0，不视作冷启动 DMA。

| 用例 | 源调用 | 设备窗口 | 独立系统审计 |
| --- | ---: | ---: | --- |
| 非相邻算子对 | 5 | 3 | [separated-pair](../artifacts/tagged/pair-wide-model-audit-separated-pair-001/report.json) |
| 8-batch 广播矩阵 | 4 | 1 | [batch-pair](../artifacts/tagged/pair-wide-model-audit-batch-001/report.json) |
| 正常三步生成，观察关闭 | 45 | 18 | [generation-0](../artifacts/tagged/pair-wide-model-audit-generation-0-001/report.json) |
| 相同正常生成，观察开启 | 45 | 18 | [generation-observed](../artifacts/tagged/pair-wide-model-audit-generation-observed-001/report.json) |
| 权重扰动三步生成 | 45 | 18 | [generation-1](../artifacts/tagged/pair-wide-model-audit-generation-1-001/report.json) |

每项审计核对完整 CPU ELF 重建、源与 batch 的实际工作量、物理请求/响应及排空、所有 ELF/资产初始化区段和输出字节检查。正常生成的观察开/关两次运行使用固定种子，除观察器自身记录外的设备结果完全相同。所有结果仍是登记图验证，不是完整 Llama2 或全部 MLX 模型通过。

参考 argmax 检查现在按最后一维逐行计算，覆盖广播矩阵的全部输出行，而不是把多行 logits 当成一个向量；并拒绝空形状、非法维度和非有限值。完整 Llama2 的额外身份、层数、参数与生成条件未放宽。[151 项相关检查](../artifacts/tagged/pair-wide-audit-tests-001/regression.xml)已通过。

## 完整 Llama2 v2 镜像：仅准备与初始化

[完整镜像](../artifacts/tagged/model-e2e/llama2-pair-system-image-001/report.json)及[完整镜像审计](../artifacts/tagged/model-e2e/llama2-pair-system-image-audit-001/report.json)保留公开 dense Llama2 的全部 6181 源、873 对、8284 任务和 360 个资产（13476831558 字节），没有缩减模型或权重。

镜像审计不再默认回到 v1：它按 v2 参数重编译完整任务/地址计划，并重新生成 CPU 检查器、ELF、描述符和 launch map，要求与准备的镜像逐字节一致。随后把完整 ELF 和全部资产装入实际 C++ 内存对象，核对 361 个初始化段、目标摘要及无重叠区间。模型时钟、AXI 读写计数均为 0，`actual_cpu_execution=false`；这一结果不能被标为推理执行通过。

完整镜像图基址为 `0x88000000`，给低地址代码/栈保留 128 MiB。完整 CPU ELF 为 33115096 字节，命令为 31808512 字节；这些是实际生成尺寸，不是模型算量或性能估计。原 GPU 数值误差门槛仍保留，显式数值参考绑定不能覆盖其失败。

后续需以独立目录和独占二进制实际执行此完整配对模型，并在终态后审计全部输出、源/批次、事件、初始化和数值参考。旧版完整运行的结果不得重标为使用 v2。一般 CDC/NoC 直传、其他所需模型、全部输入范围、推理性能和 RTL 仍未完成。
