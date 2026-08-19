# Fig.20 N=4096 Attention 修复

H193 的两个超限点来自外部基线特征设计，而不是 MLX 数值功能错误。原实现把
RTX4090 上 FlashAttention 与单独 `rfft+irfft` 的逐形状时间对比直接映射到论文的
Xavier speedup。N=4096 的该特征发生局部反转，导致两个方向相反但乘积近似守恒的
误差。

H195 对五个序列长度的 target-free trace contrast 建立留一 log-N 拟合。预测每个
形状时排除该形状，不读取论文目标，也不重拟合 H183 参数。结果如下：

| 点 | 修复前误差 | 修复后误差 |
|---|---:|---:|
| dense-TCU Attention, N=4096 | 27.89% | 2.39% |
| sparse-CUDA Attention, N=4096 | 20.91% | 1.22% |

六个 Attention 留出点及全部 48 个留出点均进入 15%，36/36 趋势保持一致。该结果
仍是对两端 log-N 插值参考的事后修复；论文没有 N=4096 实测点，RTX4090 proxy 也
不等同于 Xavier，因此不能升级为独立硬件验证。

机器证据：`artifacts/results/fig20-attention-holdout-repair-run200.json`。
