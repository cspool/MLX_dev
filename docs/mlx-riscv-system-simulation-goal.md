# MLX + RISC-V 系统协同仿真目标

更新日期：2026-09-02

## 1. 总目标

构建一个无需实体芯片的流片前风格验证环境：RISC-V 在 Chipyard/Verilator
中运行 bare-metal ELF，通过精简命令接口配置并启动 MLX，MLX 周期模型或
4×4 PE 阵列 RTL 执行空间程序，行为级存储系统提供 DMA/SPM/DRAM 时序。

目标是验证 MLX 的功能、关键指令周期和端到端性能收益趋势，并由 RTL 活动
轨迹评估 MLX 关键模块的面积与功耗。CPU、缓存和系统内存不计入 MLX PPA。

## 2. 论文证据边界

论文明确公开了周期精确 MLX 模拟器、Verilog RTL、RISC-V host、空间汇编器
和流片硬件测量，但没有公开其 CPU 型号、系统互连、缓存/DRAM 参数或完整
系统验证平台。因此本项目是“论文约束下的 MLX + RISC-V 系统重建”，不能
声称复刻了作者未公开的全系统实现。

## 3. 目标结构

```text
模型/算子
   -> MLX lowering 与空间汇编
   -> C header、数据镜像与 bare-metal ELF
   -> Chipyard RISC-V
   -> RoCC/TileLink 命令与内存接口
   -> MLX 系统控制器
   -> 周期模型（完整负载）或 4×4 PE RTL（关键路径）
   -> 行为级 DMA/SPM/DRAM
```

两个 MLX 后端共享相同的命令、程序、数据和统计格式：

1. 快速周期模型用于完整工作负载、参数扫描和论文性能趋势复现。
2. Verilator RTL 用于关键指令、小型内核、逐周期一致性和 VCD 活动采集。

## 4. 实施优先级

### P0：系统执行闭环

1. 定义 `config / launch / wait / status` 的 RISC-V 可见接口。
2. 在 Chipyard 中接入 MLX 周期模型，运行真实 bare-metal ELF。
3. 实现行为级 DMA/SPM/DRAM 时序和地址空间。
4. 打通空间二进制、C header、输入数据、输出校验和运行清单。

### P1：可执行 MLX RTL

1. 实现自主取指、Tag 调度、RF、FU、load/store、xfer 和完成反馈闭环。
2. 实现真实 4×4 PE 阵列、坐标、邻接/skip-hop 网络、流控和阵列控制器。
3. 将 RTL 后端接入与周期模型相同的 Chipyard 接口。

### P2：功能与性能实验

1. 运行 BSMM、FFT-CMP、SWA 以及至少一个组合 Transformer block。
2. 对齐软件参考、周期模型和 RTL 的输出。
3. 测量 host、配置、DMA、load/store、compute、xfer、同步和总周期。
4. 测量 FMA、ADD、MUL、MAX、EXP、DIV、SHUFFLE、XFER、LOAD、STORE 的
   latency、initiation interval、依赖停顿和资源冲突。
5. 复现论文关键架构创新相对 baseline 的明显同向收益；数值接近是次级目标。

### P3：RTL PPA

1. 对真实 4×4 MLX 顶层综合、STA 和布局布线，而不是单 PE 乘以 16。
2. 使用系统/RTL 工作负载生成的 VCD 估算动态功耗。
3. 分别报告未经目标拟合的原始结果和任何显式校准结果。
4. 报告 1 GHz slack；若无法闭合，则报告实际 Fmax 并按频率解释性能。

## 5. 完成条件

- RISC-V ELF 能配置、启动、等待 MLX，并正确读回结果。
- BSMM、FFT-CMP、SWA 在周期模型后端通过端到端功能验证。
- 关键指令和小型内核在 RTL 后端通过功能及逐周期验证。
- 周期模型与 RTL 对相同小型负载的指令数、事件顺序和周期差异有明确解释。
- 系统总周期包含 host/config/DMA/同步开销，且与加速器内核周期分开报告。
- 论文核心 MLX 性能实验呈现相同的收益趋势。
- 真实 4×4 阵列获得面积、功耗、时序和关键路径报告。
- 所有结果标明来源：测量、RTL、系统仿真、架构模拟、推断或目标校准。

## 6. 非目标

- 不实现或流片完整 SoC。
- 不要求运行 Linux；bare-metal 系统验证足够。
- 不要求复刻作者未公开的 RISC-V 核、私有 12 nm 库或 Synopsys 流程。
- 不把 CPU、缓存、DRAM PHY 或行为级内存模型计入 MLX 面积与功耗。
- 不以训练、模型质量或 GPU 原生实验阻塞 MLX 系统和架构验证。

## 7. 实施状态与审计入口

- lowering、空间汇编、周期模型及自主执行的 4×4 RTL 已实现，四个工作负载共享
  同一套程序、输入、golden、reference 和 lineage 清单。
- 默认 `mlx_array_4x4` 已切换为 16 个自主 `mlx_array_pe_tile` 构成的分布式顶层；
  旧集中式实现仅保留为诊断基线。单 tile 已完成零 DRC/零 pin-access 物理签核，
  分布式顶层由 `scripts/run_mlx_distributed_top_ppa.py` 重放综合至 DRT/STA/power。
- Chipyard 使用 `/root/chipyard` 固定 checkout，commit
  `b5d013190d637e634113cb5179f8c8885df1945a`；兼容补丁与幂等安装脚本保存在本仓库。
- cycle/RTL 两个 Rocket 配置均运行真实 bare-metal ELF，并通过相同 RoCC 命令接口
  完成 config、launch、wait/status、行为 DMA/SPM/DRAM 和输出回读。
- H205/run210 与 H207/run212 分别保存 standalone 后端和 Chipyard 的机器可读结果；
  H206/run211 保存真实 4×4 分层物理实现，H208/run213 汇总最终逐项审计。
- 早期单 PE 外推和目标校准结果不作为本目标证据；最终 PPA 使用未经目标拟合的
  Nangate45 综合、布局布线、STA 与 workload VCD 结果，并明确报告证据边界。

## 8. 2026-09-02 关键进展快照（P3 真实 4×4 PPA）

- P0–P2：真实 Chipyard + bare-metal ELF、周期模型、RTL、四个负载功能闭环已稳定打通；
  共享接口与数据清单在两个后端一致，并已形成可复用验证链路。
- 真实 4×4 物理流：`run210`（standalone）与 `run212`（Chipyard）已通过对应检查；
  当前真实顶层签核仍卡在 `run211`，尚未生成最终结果 JSON 与 `run213` 审计。
- `run211` 当前执行到 `scripts.run_mlx_distributed_top_ppa --stage local-repair` 的
  `repair5`，`repair5` 进入 detail 路由第一轮并写到
  `artifacts/environment/h206/distributed_4x4/mlx-array-4x4-distributed-u70-iter5-clean-retry1-local-repair5-droute.log`；
  最新日志仅有 `Completing 10%/20%/30%/40%/50%`（2/9/9/9/14），未进入 `MLX_ARRAY_DROUTE_COMPLETE`。
- `run211` 仍缺失受验收入口要求的签核产物：`artifacts/environment/h206/distributed_4x4/*local-repair5-routed.{def,odb,drc,spef}`
  以及 `artifacts/results/mlx-array-ppa-run211.json`、`artifacts/results/mlx-riscv-system-goal-run213.json`。
- 运行证据门禁仍是：
  1) GRT completion marker 与全部网 routed 且 `GRT-0026=0`
  2) 同步通过 DRT completion、`stdCellPinNoAp=0`、`macroNoAp=0`、`DRT-0073=0`
  3) 同时保留 `repair5` 的输入、输出与日志审计链条。
- 与 1 GHz 目标的关系：当前目标优先是完成验收版签核链路（零 DRC/可复现验收产物）；
  后续再按 run213 结论决定是否在顶层层级补齐 1 GHz 时序闭合报告。

### 2026-09-02 20:05 UTC 追加核验

- `run211` 观察窗口显示：`openroad` 进程 PID 2905117 仍在运行（CPU 约 1119%，RSS 约 69.5 GB），
  但日志 `.../local-repair5-droute.log` 自 19:45:55 起长度与时间戳未变化（`25359` 字节）。
- 仅看到 `Completing 10%/20%/30%/40%/50%` 的进度；尚未出现 `MLX_ARRAY_DROUTE_COMPLETE`、
  `DRT completion`、`*.routed.{def,odb,drc,spef}`。
- 这意味着当前卡点是可复现的验证对象本身（repair5 细节布线第一轮），不是 DRT/STA 之后阶段；
  目标下一步是等待该迭代结束或中止并改造该轮次配置（不在当前目标文档内作策略细节裁剪）。

## 9. 变更对齐规则（用于目标审计）

- 本文档以 `docs/mlx-riscv-system-simulation.md` 的最新运行日志与产物为主线，任何
  "可运行但未写入受验收路径" 只能作为诊断，不得替代 `run211/run213` 合法通过。
- `run211` 的受验收条件未满足前，不得将 P3 标注为完成；不以 `GRT` 低 overflow
  或 "所有网已route" 绕过拥塞/DRT 门禁。
