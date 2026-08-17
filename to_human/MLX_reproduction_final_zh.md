# MLX 论文来源与复现最终报告

## 最终结论（截至 2026-08-17）

本仓库已经完成可公开核验范围内的部署、目标提取、原生/替代实验和逐项审计，但**没有实现“论文全部实验误差均不超过 10%”**。最终机器证书覆盖 18 个数值实验条目：1 项完整通过，7 项充分尝试后被拒绝，3 项仅属于目标导向回放，7 项因公开输入、作者实现、原始 trace、RTL 或所需硬件缺失而阻塞。

这不是把“做不到”当成通过：证书字段
`all_paper_experiments_reproduced_within_10pct` 明确为 `false`，并且
`exact_mlx_author_artifact_used` 为 `false`。

## 来源推断

证据支持的最强表述是：

- MLX 明确在 256-GOp/s 缩减设计模拟器处引用了 **SimICT**，但只能证明“引用层面的框架关系”，不能证明复用了 SimICT 源码。
- MLX 与 **DFU-E、M2-DFU、DFGAS** 以及 2023 ICCD transfer 工作处于同一 ICT/CAS 论文谱系，作者和机构高度重叠；这些信息不能单独证明架构派生。
- **DSAGEN** 是最接近的公开空间架构编译/模拟替代底座；**Assassyn** 是有价值的异步周期语义参照。二者都不是已证明的 MLX 原始项目。
- 精确流片父芯片仍未解析；没有公开证据支持任何候选项目的代码或 RTL 是 MLX 的直接来源。

H33-H36 已穷尽当前可用渠道：精确论文 artifact 搜索、谱系主记录、记录导出的 IEEE/ACM/DOI/UCAS 表示，以及一次冻结的 Fig. 14 原图文字检查。Fig. 14 中没有可可靠转写的芯片名或硬件数值。

## 已部署内容

仓库提供一个可检查的 MLX 专用 CPU 离散事件替代模拟器，建模闭合依赖组件、有限跳数 skip link、tag 调度以及 load/compute/transfer 解耦流水。环境位于项目 `.venv`；DSAGEN、Accel-Sim、Timeloop、FABNet 等作为公开参照或可选后端，不冒充作者实现。

常用入口：

```bash
make test
make lint
make reproduce-architecture
make audit-completion
```

`make audit-completion` 是只读复核，不会覆盖 formal 结果。

## 18 项证书

| 类别 | 数量 | 论文条目 | 含义 |
|---|---:|---|---|
| 10% 内完整通过 | 1 | Fig. 22 | 开放替代模拟器的两条完整利用率曲线通过；最大误差 7.10% |
| 充分尝试后拒绝 | 7 | Fig. 5、Fig. 6、Fig. 15(b)、Fig. 18、Table V/Fig. 19、Fig. 20、Fig. 21 | 已执行预注册的充分替代/原生尝试，但完整逐点门限失败 |
| 仅校准回放 | 3 | Fig. 23、Fig. 24、Fig. 25 | 数值可接近或精确匹配，但目标已参与校准/拟合，不能算验证 |
| 公开证据阻塞 | 7 | Fig. 2、Fig. 3、Table II/Fig. 14、Fig. 15(a)、Fig. 15(c,d)、Fig. 16、Fig. 17 | 缺作者实现、数据划分、训练配方、trace、RTL、合成流或目标硬件测量 |

重要的局部成功仍被保留：

- Fig. 22：BSMM/FFT 最大相对误差分别为 3.33%/7.10%。
- Fig. 23：三组 scaling 数值最大误差为 1.07%-3.20%，但属于已消耗目标的校准证据。
- Fig. 15/16：公开公式生成的全部 MLX compute bars 均在 10% 内，但这不复现 accuracy/perplexity 训练结果。
- InternLM2 和 Llama2 的 WikiText-2 原始 PPL 分别以 3.91% 和 7.91% 误差通过；BERT/SQuAD 原始基线也通过。
- Llama2 WinoGrande 的一个预注册 dense LoRA 替代基线误差为 2.50%，但它不是作者公开的原始配方，也不能证明压缩模型。

主要拒绝证据包括：Fig. 6 最大误差 96.83%，BERT 深层替换最大误差 81.47%，Fig. 19 最终 fused-FFT 假设最大误差 77.49%，Fig. 20 speed 最大误差 79.03%/122.72%，Fig. 21 speed 最大误差 75.51%。Fig. 21 dense memory 在新目标下通过，但 sparse memory 最大误差 13.97%，因此整项仍失败。

## 不可变证据

- 协议源码提交：`f8be0f5aa47fb08893428731e68b2b1ce997d267`
- fresh suite：`artifacts/results/full-paper-lightweight-suite-run042.json`
  - 449,350 bytes
  - SHA-256 `07d47946a391c5b1192eba4fa914a23ba36f2a0ccce33424a3d7d6f229a3877e`
- 完成证书：`artifacts/results/full-paper-completion-certificate-run042.json`
  - 70,922 bytes
  - SHA-256 `7182c5dd1612680f85912c6e96bf14e78b871a4d560f0f3610a6c0dc7de6b071`
- 全量验证：164 tests passed，17 个已知 warning；全仓 Ruff diagnostics 通过。

## 重新开启条件

只有新增的独立一手证据才足以重新开启已停止路线，例如作者发布 MLX simulator/RTL/compiler/mapping、完整压缩模型与训练 manifest、FGSCR-42 精确划分、原始 GPU/FPGA trace、目标硬件访问，或明确的 MLX↔DFU-E/M2-DFU 派生声明。继续从已观察残差选择参数、URL、OCR 或训练配方，只会形成后验拟合，不能满足论文复现定义。
