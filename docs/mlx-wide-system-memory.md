# 16GiB 系统地址空间与受检初始化

2026-09-08。新增 [wide_memory](../system_sim/wide_memory/memory.cc) 和 [MLXClockedLargeRocketConfig](../system_sim/chipyard/MLXWideMemory.scala)。这是独立仿真配置；PE、RF、SPM和算子微程序资源没有扩容。旧256MiB配置与历史证据保留。

## 为什么不能只改容量参数

本地TestChipIP的旧SimDRAM将AXI地址送入32-bit DPI参数，底层`mm_t`又使用地址取模。这在旧的小容量、固定基址配置下不能证明高位地址正确；直接扩大容量可能使不同物理地址互相别名。

新仿真封装在SV和DPI边界保留64-bit地址，显式检查`[base, base+size)`，再转换成相对内存偏移。后端继续复用原`mm_magic_t`：范围检查保证其内部取模对合法偏移不产生作用。没有通过复制高低地址的同一数据掩盖别名问题。

当前支持64-bit数据总线、64-byte cache line、一个AXI外存端口、INCR突发及单拍窄访问；多拍读使用8-byte beat，窄写保留字节选通。非法地址、尺寸、突发模式、写入last/选通和在途reset明确拒绝。此实现仍是magic memory，不是已校准的DRAMSim2/HBM时序或硬件容量/面积证明。

生成设备树已核对以下真实系统配置：

| 项目 | 范围/值 |
|---|---|
| RAM基址 | `0x80000000` |
| RAM容量 | `0x400000000` bytes，即16GiB |
| RAM末端（不含） | `0x480000000` |
| 高地址图窗口 | `0x181000000`起1MiB |
| 完整资产初始化计划的图基址 | `0x88000000`，为低地址命令镜像保留128MiB |

高地址图窗口用于验证地址通路，不是完整模型的默认布局。完整模型需显式选择足够大的图窗口，并保留程序/栈与数据之间的间隔。

## 初始化与执行分开记录

`--large-memory --preload-elf`使用受检的ELF64 little-endian RISC-V ET_EXEC初始化：检查程序头、文件区间、物理地址、可执行入口和段重叠，实际复制PT_LOAD文件字节并清零BSS尾部；复制后核对目标摘要。它在内存时钟开始之前进行，**不计为CPU或DMA装载时间**。

该配置的`+loadmem=ELF`是二进制ELF入口，不是旧SimDRAM的hex格式。runner通过`+permissive ... +permissive-off`把选项送到内存与TSI模型；同一ELF同时供FESVR取得入口/符号。不能只让TSI跳过加载，却没有初始化程序内存。非wide配置拒绝此入口。

可选文件片段初始化保留64-bit文件偏移、精确字节数、目标地址和SHA256；初始化后的目标数据再次核对，禁止覆盖已初始化段。它可提供完整模型的初始内存，但不是模型计算，不是CPU复制，也不是推理性能。初始RAM延续旧magic模型的零初始化语义；模型输出是否真正写满仍由设备控制器的本次写响应覆盖检查负责。

## 当前证据

- [wide-memory-002](../artifacts/tagged/wide-memory-002/report.json)：11项pytest、10个C++驱动作业，ASan/UBSan/泄漏检查结果一致。实际SV/DPI驱动写入并读回相差4GiB的不同值、上界附近数据和窄写入，验证地址没有被封装截断。
- [clocked-wide-chipyard-002](../artifacts/tagged/clocked-wide-chipyard-002/report.json)：正常和真实权重扰动的45节点图都在16GiB配置、高地址窗口通过；每例21设备任务和完整logits/token检查，内存报告确实收到高于4GiB的地址。正常例内存最高读地址为6459294208，非编译计划中的推测地址。
- [llama2-wide-assets-001](../artifacts/tagged/model-e2e/llama2-wide-assets-001/report.json)：完整公开Llama2的360资产、13476831558 bytes在该C++内存类中初始化并全部摘要匹配；完整6181源节点计划与33286912-byte命令数据同步生成。时钟、AXI请求和CPU执行均为0，不能将其计为完整模型推理或实际Chipyard资产加载。
- [170项相关回归](../artifacts/tagged/wide-memory-tests-001.xml)通过。

`clocked-wide-chipyard-001`因未使用permissive转发选项而在命令行解析时退出，没有执行CPU。002已修正并通过。[clocked-wide-tsi-001](../artifacts/tagged/clocked-wide-tsi-001/report.json)也已通过：同一高地址正常图使用常规TSI装载，初始化段清单为空，全部输出正确。这证明两种初始化路径在登记图上的功能结果一致，不表示冷启动时间等价。

```bash
.venv/bin/python -m scripts.run_mlx_clocked_chipyard \
  --large-memory --preload-elf \
  --cases artifacts/tagged/clocked-chipyard-cases-001.json \
  --output artifacts/tagged/clocked-wide-chipyard-NEW

.venv/bin/python -m scripts.verify_mlx_wide_assets \
  --program artifacts/tagged/model-e2e/llama2-physical-full-001/program.json \
  --lifetimes artifacts/tagged/model-storage-002/model-lifetimes.json \
  --output artifacts/tagged/model-e2e/llama2-wide-assets-NEW
```

输出必须选择新目录。后续[完整系统镜像及显式后端profile](mlx-complete-system-image.md)已完成组合和初始化审计，并保留单PE演示配置；完整模型系统执行仍未完成。既有完整CPU资产复制、C++初始化、小图系统执行以及独立完整推理不能拼成一个完整模型系统证书。性能和RTL扩展继续后置。
