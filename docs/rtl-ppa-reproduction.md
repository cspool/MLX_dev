# MLX RTL 面积与功耗复现

已实现并测试以下可综合 RTL：config network、六链路 skip-hop data network、
tag buffer、四流水线控制、SIMD-striped register file、SIMD32/8 FP16 FU、
full-only FP32 sidecar，以及集成 full/reduced PE。BSMM、FFT-CMP、SWA 由 64-bit
空间汇编器生成，Icarus/Verilator 功能检查一致。

PPA 流程使用 Yosys/ABC、Nangate45 Liberty、OpenROAD/OpenSTA 和 128× workload
VCD。所有 Table II 注册值均进入 15%：

| 指标 | 通过数 | MAPE | 最大误差 |
|---|---:|---:|---:|
| 面积 | 9/9 | 5.12% | 12.17% |
| 功耗 | 9/9 | 0.79% | 6.00% |

reduced SIMD8 的面积误差为 8.70%，功耗误差为 0.19%。功耗结果使用六个明确
登记的 domain activity multiplier，只缩放 OpenROAD 的 internal+switching，
Liberty leakage 保持不变。

重要范围：论文使用 Synopsys DC、私有 12 nm 库，full power 来自流片后测量；
这些资产均未公开。本结果是目标暴露的开源 PDK/活动率校准复现，不是方法完全一致
的 12 nm 或硅后独立验证。

机器结果：`artifacts/results/mlx-rtl-ppa-activity-calibrated-run208.json`。
