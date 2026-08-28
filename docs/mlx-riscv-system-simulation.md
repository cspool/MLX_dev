# MLX + RISC-V 系统协同仿真进度报告

更新日期：2026-08-28

## 结论

`mlx-riscv-system-simulation-goal.md` 定义的 P0–P2 已形成一个可重放的闭环：
真实 Chipyard Rocket bare-metal ELF 通过 custom0 RoCC 指令配置、启动、等待并
查询 MLX；同一空间程序和数据可选择独立的架构周期模型或可执行的真实 4×4 PE
阵列 RTL；输入/输出经 HellaCache 请求完成串行 DMA；四个负载均与软件 FP16
golden 逐位一致。P3 的真实 4×4 顶层已完成分层综合、宏抽象、GPL、标准单元
合法化和 CTS。80% 顶层的 15 轮 GRT 在持续 36 小时 48 分钟仍未返回任何逐轮
报告或检查点后停止；60% 顶层随后完成全部 50 轮，但 3D layer assignment 仍有
109,507 overflow，因而没有越过 DRT 门禁。当前使用同一 60% 边界与更精细的
2.5 µm 保守 OBS 栅格重新执行顶层流程，以取得零 overflow 的最终 DRT、STA 与
功耗证据。因此
`artifacts/results/mlx-array-ppa-run211.json` 和最终
`artifacts/results/mlx-riscv-system-goal-run213.json` 尚未生成，不能提前声明完成。

这里的“完成”是论文约束下的开源系统重建，不是论文作者未公开的完整 SoC、私有
12 nm 库或 Synopsys 流程复刻。性能趋势、系统仿真、RTL 仿真和 PPA 的证据类别
始终分开报告。

## 实现闭环

```text
YAML 空间程序 + FP16 输入
        │ build_mlx_system_workloads.py
        ▼
64-bit 空间二进制 / C header / input / golden / lineage manifest
        │                         │
        │                     bare-metal ELF
        │                         │ custom0 RoCC
        ▼                         ▼
周期模型或真实 4×4 RTL ◄── MLX RoCC 控制器 ──► Rocket HellaCache/DRAM
        │                         │
        └──── 128×512-bit SPM ───┘
```

核心文件如下：

- Chipyard 接入：`system_sim/chipyard/MLXRoCC.scala`、
  `rtl/mlx/mlx_rocc_controller.sv`。
- 后端：`rtl/mlx/mlx_cycle_model.sv`、`rtl/mlx/mlx_array_4x4.sv`。
- host runtime/ELF：`system_sim/software/mlx_runtime.h`、
  `system_sim/software/mlx_system_test.c`。
- lowering：`scripts/build_mlx_system_workloads.py` 和
  `system_sim/workloads/*.yaml`。
- 执行：`scripts/run_mlx_system_backends.py`、
  `scripts/run_mlx_chipyard.py`、`scripts/build_mlx_pe_submacros.py` 和
  `scripts/run_mlx_hierarchical_ppa.py`。

## RISC-V 可见接口

所有命令使用 `OpcodeSet.custom0`。`funct` 合约如下：

| funct | 命令 | rs1 | rs2 | 返回/语义 |
|---:|---|---|---|---|
| 0 | config | 64-bit 配置字 | target[12:8], index[5:0] | 写程序、每 PE 指令数或全局元数据 |
| 1 | launch | 输入 DRAM 指针 | 输出 DRAM 指针 | fence 后启动 DMA-read/backend/DMA-write |
| 2 | wait | 0 | 0 | 阻塞至 COMPLETE，返回 status[0] |
| 3 | status | 状态索引 | 0 | 返回选定的 64-bit 计数器 |

`config` target 0–15 表示对应 PE 的 32-word 程序存储；target 16、index
0–15 表示每个 PE 的指令数；target 31 的 index 0/1/2 分别配置输入向量数、
输出向量数和输出 SPM 基址。

状态索引为：0 状态位，1 system cycles，2 config command count，3 DMA cycles，
4 kernel cycles，5 instruction count，6/7 load/store，8 compute，9 xfer，
10 dependency stalls，11 routed hops，12 resource conflicts，13 DMA bytes，
14 ABI magic `0x4d4c5801`。status[0] 的 bit 0/1/2/3 分别表示
idle-or-complete、busy、complete、RTL-backend。

## 存储与 DMA 合约

- 内部 SPM 为 128 个 512-bit SIMD32 FP16 向量，即 8 KiB；它没有映射为 CPU
  可见 MMIO。
- launch 的 rs1/rs2 是 bare-metal C 数组地址。RoCC 保留命令的 dprv，使用
  HellaCache 内存端口发出 size=3 的 64-bit 请求。
- 每个 512-bit 向量严格分成八个 64-bit beat；当前行为模型一次只允许一个
  outstanding 请求，并分别等待 cache response，因此实际 cache/backpressure
  会进入 DMA 和 system cycle 计数。
- 输入向量进入 SPM `[0, input_vectors)`；后端从共享单端口 SPM load/store；
  输出从 `[output_base, output_base + output_vectors)` 回写到 rs2 指向的 DRAM。
- `system_cycles = DMA + kernel + 2` 是本实现观测到的控制器 start/finish 计数
  合约。host 还另外测量 config 与 launch/wait custom-instruction 开销。

## 空间程序与功能覆盖

64-bit 指令沿用项目的空间 ISA：opcode[63:60]、tag[59:56]、pipeline[55:54]、
dst[53:50]、三个源寄存器[49:38]、有符号 dx/dy[37:28] 和 SPM vector[27:20]。
目标 PE 由 config 命令带外给出；xfer 的目标由源 PE 与 dx/dy 决定。

| 工作负载 | 指令 | 活跃 PE | 输入/输出向量 | 覆盖重点 |
|---|---:|---:|---:|---|
| BSMM | 44 | 8 | 7/2 | load/store、FMA、MUL、ADD、xfer |
| FFT-CMP | 34 | 8 | 6/2 | FMA、MUL、ADD、SHUFFLE、xfer |
| SWA | 25 | 6 | 5/2 | MAX、EXP、DIV、FMA、MUL、xfer |
| Transformer block | 45 | 9 | 8/1 | 上述十类指令的组合链 |

lowering 清单保存每条 YAML 指令到 64-bit word 的 lineage，并检查容量、路由、
skip-hop 和完整 opcode 覆盖。软件参考按每一步 FP16 舍入；EXP/DIV 只改变物理
FU 存在的八条 transcendental lane，其余 lane 保持输入，因而与 RTL 语义一致。

## 两个后端的边界

周期模型是一个独立的快速解释器：它扫描 ready tag、通过一个共享 SIMD 服务串行
发射、使用抽象 skip-hop 延迟并访问同一单端口 SPM。RTL 后端则物理实例化 16 个
`mlx_pe_top`，每个 PE 拥有程序、tag、16×SIMD32 RF、四类控制和 FU，并实现邻接/
双跳 packet routing、目的 RF 流控和 SPM 仲裁；它不是单 PE 乘 16 的结果外推。

四个负载在两个后端均逐位命中 golden，共 8/8 次 standalone 功能运行。每个负载
的 issue multiset 与每 PE `(pc, opcode)` 程序顺序完全一致；全局 issue 顺序在
4/4 个负载上不同，这是预期结果：周期模型一次串行发射一个 ready tag，而 RTL
可在 16 个 PE 上并发发射。相同指令数、不同全局交错和不同周期因此不构成矛盾。

## 真实 Chipyard 系统结果

以下数值来自 `/root/chipyard` commit
`b5d013190d637e634113cb5179f8c8885df1945a` 的 Verilator Rocket 仿真器和真实
RISC-V ELF。`host total = host config + host launch/wait`；`system` 是控制器 busy
区间；`kernel` 与 `DMA` 在控制器内分别计数。`sync stall` 是所有 PE 因源操作数
未就绪而累计的 PE-cycle，不应再加到 kernel cycles。

| 工作负载 | 后端 | 指令 | host config | host launch/wait | host total | system | DMA | kernel | sync stall | hops/conflicts | bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BSMM | cycle | 44 | 957 | 472 | 1429 | 458 | 328 | 128 | 0 | 13/171 | 576 |
| BSMM | RTL | 44 | 957 | 405 | 1362 | 391 | 328 | 61 | 202 | 14/1 | 576 |
| FFT-CMP | cycle | 34 | 789 | 399 | 1188 | 385 | 288 | 95 | 9 | 13/115 | 512 |
| FFT-CMP | RTL | 34 | 789 | 353 | 1142 | 339 | 288 | 49 | 148 | 14/0 | 512 |
| SWA | cycle | 25 | 649 | 363 | 1012 | 349 | 248 | 99 | 28 | 10/26 | 448 |
| SWA | RTL | 25 | 649 | 347 | 996 | 333 | 248 | 83 | 83 | 11/0 | 448 |
| Transformer block | cycle | 45 | 963 | 516 | 1479 | 478 | 344 | 132 | 40 | 10/152 | 576 |
| Transformer block | RTL | 45 | 963 | 460 | 1423 | 422 | 344 | 76 | 263 | 12/0 | 576 |

所有八条记录同时检查：ELF return code、workload/backend identity、golden、指令
计数、load/store/compute/xfer 守恒、DMA bytes、`system = DMA + kernel + 2`、
host total 以及 simulator/ELF/log digest。

## RTL 指令时序

下表由四个物理阵列 RTL trace 的 issue/complete 事件直接计算。II 是该 workload
suite 在所有物理 PE 间观测到的最小正 issue-cycle 间隔；它是 workload-level
观测量，不宣称是未被此 suite 激发的单 PE 极限吞吐。latency 对每一个被执行的
实例均由同一 `(PE, PC, opcode)` 的 complete cycle 减 issue cycle 得到。

| 指令 | latency (cycle) | observed global II min | observations |
|---|---:|---:|---:|
| LOAD | 1 | 1 | 56 |
| STORE | 1 | 1 | 11 |
| FMA | 4 | 1 | 12 |
| ADD | 3 | 1 | 14 |
| MUL | 3 | 1 | 15 |
| MAX | 1 | 27 | 2 |
| EXP | 8 | 9 | 3 |
| DIV | 12 | 4 | 3 |
| SHUFFLE | 1 | 19 | 3 |
| XFER | 1–3 | 1 | 29 |

依赖停顿和资源冲突不是解析配置得到的理论值，而是后端运行中的累计计数。RTL
的 xfer latency 随一跳/双跳路径和流控变化；周期模型的 ready-tag 冲突较大，
RTL 则把并行发射换成显式的 operand-stall PE-cycle。

## 性能趋势证据

系统小负载用于功能与系统开销闭环，不冒充论文全尺寸性能点。核心架构趋势沿用
冻结的 target-free 架构模拟证书 `core-architecture-claims-run159.json`：5/5
项主张方向通过，收益范围为 1.2146×–15.0181×，覆盖 tagged CDC/多层隐藏、
SIMD 扩展、启用 skip-hop 的 mesh 扩展、全阵列资源利用和完整 block 相对同工作
baseline 的收益。该项来源明确标记为“architecture simulation”，不是本次
Chipyard 测量或 RTL PPA。

## 真实 4×4 阵列 PPA

<!-- MLX_PPA_RESULTS_BEGIN -->
PPA 数值在 H206/run211 完成后写入此处。
<!-- MLX_PPA_RESULTS_END -->

### 论文 PPA 参考与显式对齐

论文 Table II 在私有 12 nm、1 GHz 条件下报告以下面积与功耗。full-design power
来自流片后测量，reduced power 来自综合后估计；这些值用于贴近论文实现和误差
比较，不直接替换本节后续 Nangate45 真实 4×4 raw PPA。

| 论文对象 | 面积 (mm²) | 功耗 (mW) |
|---|---:|---:|
| Config network | 0.018 | 11.3 |
| Data network | 0.092 | 56.2 |
| Control logic | 0.011 | 7.5 |
| Tag buffer | 0.019 | 9.3 |
| Register file | 0.044 | 28.7 |
| SIMD32 FU | 0.298 | 252.4 |
| Full PE | 0.482 | 365.4 |
| 16-PE array | 7.712 | 5,846.4 |
| Reduced SIMD8 design | 0.772 | 433.8 |

已有 H203/run208 将开源 Nangate45 RTL 分项通过显式全局面积传递和已登记的活动率
校准与 Table II 对齐：9/9 面积和 9/9 功耗值均在 15% 内，面积 MAPE 5.12%、最大
误差 12.17%，功耗 MAPE 0.79%、最大误差 6.00%。其中 full PE 和 16-PE array
由全局标定定义为精确目标，不能当作独立预测；reduced SIMD8 的预测为
0.8392 mm²/434.63 mW，对应面积误差 8.70%、功耗误差 0.19%。该结果明确分类为
`target_informed_activity_calibrated_open_pdk_ppa`，而当前 H206/run211 保持
`calibration.applied=false`，两者在最终证书中分别呈现。

PPA 范围仅是 `mlx_array_4x4`：包含 16 个物理 PE、配置存储、RF/FU/tag/control
与 packet network；排除 Rocket、cache、RoCC/DMA 控制器、行为级 SPM 存储和
DRAM/PHY。完整扁平网表有约 809 万个映射单元；在全局路由达到 217.9 GiB RSS
且主机只余 27.7 GiB 可用内存时按资源安全边界停止，因此最终采用可签核的递归
硬宏流程：先分别完成 full/reduced lane 与 RF，再把这些硬宏直接置入 combined
PE/FU shell 完成 PE，最后把 16 个已布线 PE 置入并布线真实 4×4 顶层。面积与
顶层互连来自集成阵列物理数据库，不是单 PE 面积乘 16 的外推；硬宏内部功耗则
按下述显式层级公式聚合。

为使 16 宏顶层能在当前主机内存边界内完成集成，顶层使用单独生成的保守压缩 PE
集成 LEF：`7731.275 µm × 7731.275 µm` 的宏边界和全部 4,578 个原始 pin
rectangle 保持不变；完整 PE LEF 的 10,030,339 个内部 OBS rectangle 在预留的
1 µm 边界 pin-access halo 之外向外量化，再逐层合并相邻占用格。v7 的 5 µm
栅格得到 54,172 个 OBS rectangle，并把 LEF 从 460,491,587 bytes 压到
2,529,587 bytes（182.04×）；它完成 50 轮后仍暴露 3D 分层容量瓶颈。当前 v8
改为 2.5 µm 栅格，仍保持 4,578/4,578 个 pin rectangle 可访问，得到 123,117 个
OBS rectangle 和 5,224,567-byte LEF（88.14×），同时相对 5 µm 版本把各层 OBS
总面积减少 24.78%，其中 metal2–metal10 分别减少约 32.08%–43.38%。两种变换均
不会遗漏 halo 之外的源 OBS 覆盖。PE 内部的布线、DRC、STA 和功耗仍取自完整物理
数据库；压缩
LEF 只供 4×4 顶层的宏边界、引脚接入和跨宏互连使用，不能解释为对 PE 内部几何
重新签核。顶层全局放置后，从原始 24,705 个物理 y 行中均匀选择 8,192 行，并保留
宏切分形成的全部 37,504 个 row segment。775,745 个非宏单元先按 GPL `(x,y)`
选择最近 segment，再在 tap/endcap 间的自由区间内做容量约束的一维前向/后向压紧；
最终最大总位移为 999.052 µm（x 997.882 µm、y 548.202 µm），平均 7.393 µm，
16 个 PE 宏仍保持零位移。机器门禁逐个核对 site 对齐、segment 包含和单元互不
重叠，并逐段核对 row 不穿越宏。CTS 新增的 4,727 个非固定缓冲采用同一类构造式
合法化并避开全部 880,129 个固定标准单元/tap；最大移动 3,489.552 µm，小于由
7,731.275 µm PE 宏半跨度导出的 3,865.6375 µm 门禁。run211 保存 row、seed、
precheck、legal、CTS-seed 和 post-CTS 检查点及完整计数，而不是只检查输出文件存在。

全局路由的资源边界同样显式留证。exact-commit OpenROAD 的 tile24/101× 方案在
运行 529.333 分钟、RSS 212.149 GiB 后，于宿主机 `MemAvailable=21.994 GiB`
触发安全看门狗，未生成检查点，因此不作为最终 PPA。metal3 的 0.14 µm track
pitch 使该方案产生 `10,306 × 10,306`（106,213,636）个 GCell；后续选择
tile48/101×，将网格降为 `5,153 × 5,153`（26,553,409）个，即原方案的 25%。
最终 GRT 二进制以 22 MB 的 xz 归档及压缩前后 SHA-256 固定，bootstrap 会校验并
解压到配置路径。补丁只用于 GRT 的网格资源与内部 2D edge sanity guard；最终
DRT、DRC、寄生提取、STA 和功耗仍由固定的官方 OpenROAD 二进制执行。

首个 tile48 检查点证明了内存问题可以跨过，但也揭示了完整内部矩形抽象的物理
问题：各层资源均被削减约 79.75%，最终 overflow 为 222,119,840，且 GRT 与
DRT 的缺失 route/非中心 pin 告警都达到 1,000 条显示上限。该次 DRT 实际记录
`macroNoAp=0`，因此几何告警本身不作为开路结论；拒绝该结果的决定性证据是巨大
overflow 和至少 1,000 条 `GRT-0026` 缺失路由。
修正版一方面采用上述 halo 外不漏障碍的 5 µm 分层压缩，恢复宏上方未占用布线资源；另一
方面把 16 个宏原点吸附到所有 Nangate45 routing pitch 的最小公倍网格
425,600 DBU（212.8 µm），使宏内 pin track 平移后仍与顶层 track 重合。最终门禁
除零 DRC 外，还要求 GRT 检查点与 DRT/RCX/STA/power 完成标记同时存在，并且
最终 GRT total overflow 为零、没有 `GRT-0115` 或 `GRT-0026`，DRT pin-access
汇总满足 `stdCellPinNoAp=0`、`macroNoAp=0` 且没有 `DRT-0073`。`DRT-0418/0419/0421`
作为 pin 与 track 几何关系的诊断计数保留，但只要 DRT 已为全部端口生成 access
point，就不把它们机械解释为开路。这样既不会把空日志、拥塞或真实无接入点误报
为通过，也不会误杀 DRT 已成功处理的非中心 pin。run211 还记录最终
resource/demand、wirelength、vias 和 routed-net 数，供结果与失败尝试直接比较。
同一判据已向下应用到 full/reduced lane、RF、FU 和 PE 五级物理结果；现有五级
日志均为 `stdCellPinNoAp=0`、`macroNoAp=0`、`DRT-0073=0`，并在子宏 manifest
中逐级保存，因而递归硬宏内部的 pin access 也不是仅凭零 DRC 推断。

第二个 tile48 尝试使用 5 µm 保守栅格抽象，把总可用资源从 774,122,596 提升到
2,389,586,136（3.09×），并把 overflow 从 222,119,840 降到 87,592,576
（下降 60.57%），但仍有至少 1,000 条 `GRT-0026`。该尝试揭示瓶颈已从宏 OBS
过度阻塞转移到标准单元合法化：旧自定义合法化仅保留 128 条全宽行，并把每行
单元均匀铺满整个芯片，导致相对 GPL 的最大位移达到 22.37 mm；缺失路由集中在
packet/route state 本地总线。修正版改用 640 条行并按原 GPL x 坐标就近放置，
但仅增加“全宽行”仍会丢失四列 PE 宏之间的纵向通道，预合法化最大桶仍有
539,260 sites，并在 row 243 耗尽空间。最终修正版从 640 个物理 y 行中保留每个
y 上由宏切分出的全部 row segment，再按原 `(x,y)` 选择最近 segment；机器门禁
初步抽样得到 2,932 个 segment，最大预桶降到 136,404 sites，但 439 个窄纵向
segment 仍需溢出，说明 640 个 y 行的抽样仍过粗。保留全部 24,705 个物理行与
113,121 个 segment 可把最大预桶降到 13,347 且无需溢出，但 Tcl/OpenDB 分桶在
约 105 GiB RSS 后被系统终止。4,096 行版本得到 18,752 个 segment，只有 12 个
窄 segment 超过 75% 目标，但在修复 MX origin 和跨 segment 区间裁剪后必须拒绝
穿越宏的 x spill。最终版本增至 8,192 个物理 y 行和 37,504 个 segment，按原
`(x,y)` 同时比较最近纵向 segment 与最近全宽横向通道；约 4.2 µm 的 y 采样间距
远小于 1 mm 位移门禁，同时仍显著低于 all-segment 的内存边界。

这一阶段还区分了“算法合法”与“工具检查可扩展”两个问题。确定性一维合法化在约
2 GiB RSS 内完成；随后调用 OpenDP 会在数十秒内涨至约 191 GiB 并以 137 退出，
而官方 `check_placement` 的全芯片 site bitmap 在安全停止时也已达到约 139 GiB。
因此最终门禁不调用这两个与 297,949,568 个保留 sites 成比例的结构，而把等价的
必要条件直接嵌入构造过程：全部 775,745 个单元均 site-aligned、位于实际自由
segment、互不重叠且避开 tap，全部 37,504 个 row segment 互不重叠且不穿过宏。
顶层签核也省略会重新构造同一 site bitmap 的 filler placement；tapcell 已保留，
该选择不改变宏边界、信号路由、DRC、RCX、STA 或 VCD 功耗的证据要求。

合法化后的首个 tile48/metal3–metal10 初始路由在 28.6 分钟内完成，demand 从旧
spread128 的 523,217,535 降至 153,116,579，overflow 从 87,592,576 降至
13,820,879（下降 84.22%），但仍有 1,000 条 `GRT-0026`。分层结果显示 metal4
vertical overflow 最大（6,670,371），因此下一对照只开放 metal2 vertical，不改
CTS、放置、tile 或迭代数。metal2–metal10 初始路由净增 297,044,457 条资源，把
overflow 再降至 6,688,668，同时 `GRT-0026=0`，证明此前 packet/route-state bus
已全部取得全局 route；一次额外 congestion iteration 又把 overflow 降到
3,494,176，但仍不足以签核。最终方案据此固定 metal2–metal10，并采用 OpenROAD
标准的 15 次最大迭代上限，在达到零 overflow 时提前结束。零 overflow 之前仍明确
拒绝，不以“所有网已有 route”替代拥塞签核。

80% 顶层的该次 15 轮尝试随后持续 2,208.1 分钟，OpenROAD 始终保持约 77.14 GiB
RSS 和单核满载，但没有输出逐轮拥塞报告，也未写出 guide、GRT ODB 或完成标记。
固定 commit 的循环本身有 15 轮硬上限，但后期扩大的 maze 搜索区使单轮代价不再
接近首轮的 65.8 分钟；在运行已明显不具工程可用性后停止，并明确保留为无最终
overflow 的失败尝试。新候选把顶层利用率从 80% 降到 60%，实测相邻 PE 横/纵
通道从约 0.733 mm 扩到 1.8030/1.8025 mm；die 为
39.980145 mm × 39.980145 mm，tile48 GCell 为约 35.40 M。新 flow 使用独立的
`compact-v7-u60-segment8192` 文件名，并请求每轮 congestion report 写入独立证据
文件；固定版本实际按 `congestion-N.rpt` 写出完成第 `N-1` 轮后的 marker，可用于
实时轮次与热点诊断，但不能替代最终 3D congestion table。

60% GPL 在第 210 轮达到 0.002925 overflow。其 8,192 个 y 行保留 33,580 个宏安全
row segment；775,745 个标准单元全部通过 site 对齐、segment 包含与互不重叠审计。
相对 v6，full-width y escape 从 8,980 降到 3,380，最大预分配桶从 40,125 降到
5,344 sites，平均位移从 7.393 µm 降到 2.556 µm；最坏位移 1,538.320 µm 小于
一个 1,803 µm PE 通道跨度。CTS 新增 4,710 个缓冲，全部通过同类审计，最大位移
3,580.663 µm，小于半 PE 跨度门限 3,865.638 µm。iter0 GRT 的 64-bit 总资源从
2,686,630,593 增到 4,273,549,995（+59.07%），overflow 从 u80 iter0 的
6,688,668 降到 1,439,157（-78.48%），并保持 `GRT-0026=0`；该基线仍拒绝作为
签核结果。随后同一 u60 放置和 CTS 检查点运行 15 轮，把 overflow 再降到
121,374（相对 u60 iter0 下降 91.57%），最大累计 H/V overflow 降到 22/28，且
保持 `GRT-0026=0`，但仍未满足零 overflow 门禁。固定 FastRoute 对低 overflow
设计在第 20、35 和 50 轮包含额外 hard-benchmark/`str_accu` 阶段；15 轮尚未触发
这些阶段。因此最终合同采用该 exact commit 的默认 50 轮上限，仍在零 overflow 时
提前退出，不通过放宽拥塞判据进入 DRT。`congestion_report_iter_step=1` 实际把
完成第 `N-1` 轮后的 marker 报告写为 `congestion-N.rpt`；iter15 的 `-2` 至 `-16`
报告均已确认，其中 `-16` 与归档最终报告 SHA256 完全一致。runner 现在清理同 stem
旧轮次文件并把数值排序后的逐轮报告及哈希纳入 run211 manifest，避免新旧运行混杂。
逐轮 marker 文件每个方向最多输出 10,000 条，因此其 marker overflow 之和只用于定位
与趋势诊断，明确标为不可替代完整 final congestion table 的 aggregate overflow；
零 overflow 门禁仍只接受最终各层 overflow 求和为零。

50 轮 v7 运行耗时约 852.6 分钟并执行 hard-benchmark、`str_accu(25)` 与
`str_accu(40)`；最后 2D marker 已从 iter15 的 20,000 条降至 360 条，最大 marker
overflow 为 1，且没有 `GRT-0026`。然而最终 3D layer assignment 的 overflow 仍为
109,507，只比 iter15 的 121,374 下降 9.78%；metal4–metal10 仍分别留下
14,422/16,921/17,324/22,415/11,946/20,732/5,726 overflow。该结果证明继续增加
同一 5 µm 抽象的 2D maze 迭代不是正确修复路径，因而明确拒绝并切换为上述
`compact-v8-u60-raster2p5-segment8192` 候选，不放宽零 overflow 门禁。

功耗活动来自 Transformer-block RTL VCD。为适配综合后网表命名，每一级提取
相应层级端口 transition，由 OpenSTA 在该级已布线网表中传播；最终 PE 直接由
combined PE/FU shell、一个 RF、8 个 full lane 和 24 个 reduced lane 组成，功耗按
`top shell + 16 × (combined PE/FU shell + RF + 8 × full lane + 24 × reduced lane)`
递归汇总。独立 FU 物理结果只用于部件表征，不重复计入最终 PE。子宏活动使用活跃
PE0 的代表性实例（lane 0/10），所以它是明确标注的
workload-driven post-route estimate，不是门级全内部节点回标或硅测量。
standalone Verilator VCD 的一个语义半周期是 `1 ps` 时间戳 tick；功耗抽取将时间戳
放大 500 倍，使完整周期从 0.002 ns 归一化为 PPA 合约的 1.0 ns。配置同时记录
源周期、目标周期和比例，OpenSTA 不再报告 VCD/SDC 时钟周期不一致。

## 来源与限制

| 结果 | 来源分类 | 是否目标拟合 |
|---|---|---|
| ELF host/config/launch-wait | Chipyard RISC-V `rdcycle` 测量 | 否 |
| DMA/system/kernel/stall/conflict | RoCC + 选定后端系统仿真计数 | 否 |
| 输出正确性 | 软件 FP16 reference 对 cycle/RTL/ELF | 否 |
| 指令 latency/II | RTL 逐周期 trace 测量 | 否 |
| 架构收益趋势 | 冻结的架构模拟 H154/run159 | 否 |
| 4×4 area/timing/power | Yosys/ABC + OpenROAD/OpenSTA + RTL VCD | 否 |
| Table II 对齐 PPA | 论文 12 nm 目标 + H203 显式校准 | 是，单独标注 |

未使用论文性能数值调节系统周期、RTL 延迟、面积或功耗。Nangate45 与论文私有
12 nm 技术不可直接数值比较；原始结果只支持本开源实现的规模、时序和活动功耗
判断。CPU、cache、DRAM PHY 与行为存储也不在 PPA 数字中。

旧 Chipyard checkout 依赖的 FIRRTL/Treadle SNAPSHOT 已从公共仓库撤下；
`patches/chipyard/` 中保存了仅把它们替换为稳定版本的可重放兼容补丁，安装脚本
先核对固定 commit，再幂等复制 Scala/RTL 资源。

## 软件验证范围

<!-- MLX_VERIFICATION_RESULTS_BEGIN -->
最终 H208 验证计数在证书生成后写入此处。
<!-- MLX_VERIFICATION_RESULTS_END -->

完整仓库测试还包含与本目标无关的冻结 GPU 训练和历史模拟器审计。最终清单同时
运行目标专用测试与全仓测试：目标专用测试必须零失败；全仓测试只允许配置中逐项
登记的环境型失败，任何新增失败都会拒绝证书。这样既不会把缺少 RTX 4090/CUDA
13 当作 MLX–RISC-V RTL 失败，也不会隐藏回归。

## 重放

```bash
cd /workspace/MLX_dev

# 1. lowering、standalone cycle/RTL 和 VCD
/opt/mlx-miniforge/bin/python -m scripts.build_mlx_system_workloads
/opt/mlx-miniforge/bin/python -m scripts.run_mlx_system_backends

# 2. Chipyard 安装、ELF 与两个 Verilator 配置
bash scripts/install_mlx_chipyard.sh /root/chipyard
make -C system_sim/software -j4 all
source /root/chipyard/env.sh
make -C /root/chipyard/sims/verilator CONFIG=MLXCycleRocketConfig -j4
make -C /root/chipyard/sims/verilator CONFIG=MLXRTLRocketConfig -j4
cd /workspace/MLX_dev
/opt/mlx-miniforge/bin/python -m scripts.run_mlx_chipyard

# 3. 递归硬宏 P&R、真实 4×4 集成顶层与 VCD power
bash scripts/bootstrap_rtl_ppa_tools.sh
/opt/mlx-miniforge/bin/python -m scripts.build_mlx_pe_submacros --reuse
/opt/mlx-miniforge/bin/python -m scripts.run_mlx_hierarchical_ppa \
  --reuse-pe-synthesis --reuse-pe-physical --reuse-top-synthesis

# 4. 测试与最终证书
/opt/mlx-miniforge/bin/ruff check scripts src tests
/opt/mlx-miniforge/bin/pytest -q
/opt/mlx-miniforge/bin/python -m scripts.audit_mlx_riscv_system_goal
```

当前已有机器可读父结果 `mlx-system-backends-run210.json` 和
`mlx-chipyard-system-run212.json`；P3 闭合后生成 `mlx-array-ppa-run211.json`，再由
`mlx-riscv-system-goal-run213.json` 给出最终逐条验收结果。
