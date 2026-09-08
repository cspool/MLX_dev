# MLX 原生模型：系统接口与验收合约

本文件定义 M3 使用的项目接口，不是论文作者原始指令编码。执行模式和推测边界见 [ISA 设计依据](mlx-native-isa.md)。加速器执行由原生 C++ 模型完成，Rocket/缓存/内存由 Chipyard 驱动；SV/DPI 包装不是 PE 或调度器功能 RTL。

## 1. 程序经过的接口

受限源图先进入已登记的 MLIR dialect，再经空间 lowering 生成块、PE 映射、RF/SPM 分配和机器字。RISC-V GCC 将镜像与输入、期望输出封装进 bare-metal ELF；期望输出仅供主机完成后比较，不供设备执行。Rocket 实际发出的 RoCC 命令经 DPI 进入 `mlx::system::Device`。

设备独立解码并检查镜像；输入由缓存接口的真实返回值装入 SPM，随后逐周期调用第一阶段的 `mlx::tagged::Simulator`，最后将输出写回主机内存。独立模型与系统装载结果必须在指令、事件、数值、资源和 kernel 周期上保持一致。

## 2. RISC-V 命令 ABI

`custom0` opcode 为 `0x0b`。以下 funct 是 R 型指令的 7-bit funct 字段；`.insn r` 的第二个数是 `xd/xs1/xs2` 位组合，不是设备操作码。

| 命令 | funct | xd/xs1/xs2 | rs1 | rs2 | 结果/接受条件 |
|---|---:|---:|---|---|---|
| config | 0 | 011 | 64-bit 配置字 | 配置地址 | 非 busy 时接受；写地址 0 开始新的配置集合 |
| launch | 1 | 011 | 输入缓冲地址 | 输出缓冲地址 | 非 busy 时接受；先检查镜像与地址，再执行 DMA/kernel |
| wait | 2 | 100 | 不使用 | 不使用 | 完成或出错后接受，rd 返回状态位 |
| status | 3 | 110 | 状态索引 | 不使用 | 执行期间也可查询；rd 返回对应状态 |

响应缓冲满且主机未接收时，不接受新的命令；响应数据及 rd 在反压下保持稳定。配置不能覆盖正在运行的程序。wait 表示设备完成或错误可见，并不替代主机对普通访存的内存排序；当前 host 在 launch/wait 周围执行 `fence`。

程序镜像 magic 为 `0x4d4c580200000001`。系统 ABI 查询值为 `0x4d4c5802`，两者不是同一个字段。镜像头、四字块描述符和 PE 指令各自验证；保留位、pipeline 与 opcode 不符、错误路由、缺失/多余机器字及非法资源需求不能被默默忽略。

系统增加四个 I/O 配置字：`0x1e00/0x1e01` 为输入槽低/高 64-bit 掩码，`0x1e02/0x1e03` 为输出槽掩码。两种主机缓冲分别按 SPM 槽号递增、向量内 lane 递增紧密排列；每个 lane 为 FP16 bit pattern，每个 64-bit beat 包含四个 lane。

## 3. 执行、反压与错误恢复

当前采用输入 DMA → kernel → 输出 DMA 的串行控制阶段，一个 64-bit 请求在途。请求未握手时保持地址/数据；握手后等待对应返回，输入数据来自读响应，输出写入在收到确认后计为完成。请求 size 为 3、mask 为 `0xff`，设备侧 tag 固定为 0。缓存侧的仲裁与 nack 重放由 Chipyard 接口处理，不通过重复提交新的逻辑请求实现。

DMA 基址必须 8-byte 对齐，整个缓冲范围必须落在配置的地址位宽内（当前 Rocket 桥为 40-bit）。该验收运行的是 bare-metal、有效映射的主机地址，不包含虚拟内存故障恢复、进程隔离或多请求并发协议的保证。

每次 launch 尝试都先清空上一轮的执行计数和 kernel 快照。因此非法镜像或非法指针的拒绝应显示零 kernel 周期、零 DMA 字节/请求，而不是暴露上一次成功执行的计数。有效镜像可多次 launch，每次重新从主机内存取输入；更换程序时重新从配置地址 0 装载。

错误通过 status/wait 暴露。若出错时还有合法请求在途，设备保持 busy，禁止配置或重新启动，直到原请求返回并排空；错误的 tag 不能冒充该请求的完成。wait 可以先返回错误状态，不能据此认为未完成的内存请求已撤销。全系统 reset 清除镜像、执行状态和响应缓冲；不是一个可在任意外部请求在途时单独使用的取消命令。

## 4. 计数口径

| status 索引 | 内容 | 口径 |
|---:|---|---|
| 0 | bit0 idle、bit1 busy、bit2 complete、bit4 error、bit5 native backend | complete 包括错误终止；排空错误请求时 busy 与 complete 可以同时置位 |
| 1 | system_cycles | 本次启动的设备 busy 周期 |
| 2 | config_commands | 从最近一次配置地址 0 起接受的配置字数 |
| 3、4 | dma_cycles、kernel_cycles | 对应设备阶段的周期；当前 system = DMA + kernel |
| 5–9 | 总 issue、load/store/compute/xfer issue | 已完成 kernel 的动态指令快照 |
| 10–12 | dependency stall、routed links、route stall | 已完成 kernel 的原生资源统计；不是三种可直接相加的总周期 |
| 13 | dma_bytes | 已确认读/写请求的总字节数 |
| 14、15 | ABI magic、错误码 | 当前错误码 1 表示设备拒绝/执行错误，详细原因在模型报告 |
| 16 | overlap_pe_cycles | 同 PE 至少两个上下文在途的 PE-cycle 总数 |

Rocket `rdcycle` 测得的 host config、launch/wait 区间另行记录，包含主机与命令接口开销。kernel 的周期不能代填为系统测量；不同后端也不能混用旧控制器的 `system = DMA + kernel + 2` 公式。ELF 的初始化、结果比较和 HTIF 打印未包含在 launch/wait 区间内。

## 5. 构建身份与验收

`scripts/run_mlx_native_chipyard.py --phase build` 记录 native 源码、实际链接的 C++/FESVR 库、已安装桥、Chipyard/submodule 版本及 tracked diff、工具版本和构建命令。其摘要编译进 DPI 设备，退出报告返回 `build_identity`。运行前后检查 `native-model-build.json` 与当前输入、二进制摘要一致，实际设备报告还必须返回同一标识。缺失或过期的构建记录要求重新构建，不接受只给旧二进制补写当前源码摘要。

构建依赖包含该身份头文件；源码或链接库摘要变化会改变头文件并触发重新生成、编译和链接。安装相同桥文件时不改变其时间戳，避免无意义地重跑 Scala 生成。

```bash
.venv/bin/python -m scripts.verify_mlx_native_stage2
# 已完成对应构建时可跳过构建动作，但不能跳过身份校验：
.venv/bin/python -m scripts.verify_mlx_native_stage2 --skip-build
```

前提是已准备固定版本的 Chipyard checkout、Java/SBT、RISC-V GCC、FESVR 和 native 构建依赖；本脚本不负责重新安装工具链。验收先检查第一阶段证书是否仍对应当前源码，再使用新的运行目录完成：

- C++ 协议与 DPI 对照、输入内存变更、反压、非法镜像、连续启动、错误恢复和 reset 回归。
- 四个登记负载的真实 Rocket ELF 执行，以及两 PE 折叠 BSMM 的串行/重叠对照。
- 真实 ELF 对不兼容镜像版本、错误路由、非零保留位的执行前拒绝。
- 独立重放 DMA trace，检查每笔请求/响应配对、地址连续性、输入读值、输出写值及字节守恒。
- 系统装载后与独立 C++ 的逐事件/计数/数值对照，绑定 ELF、模拟器、源码和运行日志摘要。

连续启动及错误恢复目前由直接 C++ 设备回归检查；真实 ELF 场景各执行一次 launch，两类证据分别记录。当前证书位于 `artifacts/tagged/stage2-acceptance/certificate.json`，每次验收先标为 running，全部通过后才写入 passed，失败则记录 failed；源码变化会使已有 passed 证书过期。

M3 证书只允许进入最后的 RTL 实现阶段，不证明功能 RTL、PPA、完整论文 ISA 或论文规模性能已经完成。
