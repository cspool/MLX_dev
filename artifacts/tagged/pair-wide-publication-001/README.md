# v2 配对路径：大地址空间与完整镜像前置验证

本批包含[实现说明](../../../docs/mlx-paired-wide-system.md)、审计器改进、151 项相关测试，以及以下已结束证据。

- [五个真实 Rocket 高地址用例](../pair-wide-rocket-001/report.json)：16 GiB RAM、4×4、每 PE 两个上下文，144 源调用、58 个设备窗口；含同输入观察开/关等价检查。每项另有完整 CPU ELF、初始化、输出及实际工作量审计。
- [完整公开 Llama2 v2 镜像](../model-e2e/llama2-pair-system-image-001/report.json)：6181 源、873 对、8284 任务，未缩减模型。
- [完整镜像审计](../model-e2e/llama2-pair-system-image-audit-001/report.json)：CPU ELF/检查器和描述符重建一致；完整 360 资产、13476831558 字节以及 ELF 共 361 段在 C++ RAM 中初始化并核对，时钟/AXI计数为 0。**没有模型或 CPU 推理执行。**

[manifest.json](manifest.json) 绑定本批 100 个文件、12338202 字节（不含本索引和 manifest 自身）。原始模型权重、构建目录、环境链接、宿主模拟器、大型完整模型 ELF/命令二进制及运行中的完整模型输出未打包。完整镜像的源码、布局、初始化清单与大二进制摘要保留；新环境应按脚本生成新路径，不将历史绝对路径和 PID 当成当前状态。

输入驻留初始化不是 CPU/DMA 冷启动时间，magic memory 不是已校准的 DRAM。公开 dense Llama2 不等同于作者 MLX hybrid；已有 GPU 数值门槛失败没有被更改。完整模型执行、全部模型/输入、推理性能、一般 CDC 和 RTL 的门槛仍未通过。

`sys` 保持原完整运行的冻结版本，本增量只推送到独立开发分支。

