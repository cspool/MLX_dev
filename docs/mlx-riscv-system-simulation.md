# MLX + RISC-V 系统协同仿真完成报告

更新日期：2026-08-22

## 结论

`mlx-riscv-system-simulation-goal.md` 定义的 P0–P3 已形成一个可重放的闭环：
真实 Chipyard Rocket bare-metal ELF 通过 custom0 RoCC 指令配置、启动、等待并
查询 MLX；同一空间程序和数据可选择独立的架构周期模型或可执行的真实 4×4 PE
阵列 RTL；输入/输出经 HellaCache 请求完成串行 DMA；四个负载均与软件 FP16
golden 逐位一致。最终完成证书为
`artifacts/results/mlx-riscv-system-goal-run213.json`。

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

PPA 范围仅是 `mlx_array_4x4`：包含 16 个物理 PE、配置存储、RF/FU/tag/control
与 packet network；排除 Rocket、cache、RoCC/DMA 控制器、行为级 SPM 存储和
DRAM/PHY。完整扁平网表有约 809 万个映射单元；在全局路由达到 217.9 GiB RSS
且主机只余 27.7 GiB 可用内存时按资源安全边界停止，因此最终采用可签核的递归
硬宏流程：先分别完成 full/reduced lane 与 RF，再把这些硬宏直接置入 combined
PE/FU shell 完成 PE，最后把 16 个已布线 PE 置入并布线真实 4×4 顶层。面积与
顶层互连来自集成阵列物理数据库，不是单 PE 面积乘 16 的外推；硬宏内部功耗则
按下述显式层级公式聚合。

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

机器可读父结果为 `mlx-system-backends-run210.json`、
`mlx-array-ppa-run211.json`、`mlx-chipyard-system-run212.json`，最终逐条验收结果为
`mlx-riscv-system-goal-run213.json`。
