# 实际 RV64 主机的文件资产装载

2026-09-08。新增 [asset_source.cc](../system_sim/physical_device/asset_source.cc) 和 [asset_source.py](../system_sim/physical_host/asset_source.py)，使输入权重不再必须嵌入CPU的小ELF。数值计算、设备执行和装载循环仍由C/C++与实际RV64指令完成，Python只生成输入绑定并审计身份。

## 资产如何进入设备

```text
checkpoint文件区间 / literal文件
  → C++只读输入地址区（基址256GiB）
  → RV64 CPU执行LD/LBU与SD/SB复制
  → 设备的C++稀疏内存
  → 装载摘要核对 → 原有矩阵/向量/搬运/控制任务
```

主机资产表仍使用原来的source/destination/bytes ABI，没有新增MLX PE opcode。编译器保留checkpoint文件名、64-bit文件偏移、dtype/shape对应的准确字节数与目标绑定；仅将literal打包到独立小文件，不复制完整checkpoint生成巨型payload或ELF。输入区与设备窗口不重叠，未映射间隙、跨区域访问、非法宽度/对齐和写操作均拒绝。

输入区使用一个64KiB宿主读缓存减少系统调用。它不是模拟硬件缓存，不计为RF/SPM、DRAM或存储设备的目标时序。文件大小及身份在打开、缓存重新填充和最终报告时检查，runner在执行前后核对完整文件SHA256；不能把运行中改变过的文件记为有效输入。

CPU读计数按资产记录，当前装载器必须顺序、恰好读取每个资产的全部字节。第一次计算launch之前，设备通过有界宿主诊断读回已经由CPU写入的字节，核对每个资产SHA256及初始化状态；只提供预期摘要，不向设备回填golden数据。摘要错误或未写满会在计算数据访问之前失败。该诊断读回不计为目标DMA或推理周期。

诊断缓冲使用堆内存。开发时64KiB局部缓冲曾超过Spike的64KiB协程栈，造成普通构建退出损坏；已修正并重新运行普通/插桩检查。不能因ASan在该协程环境下未报告同样错误而忽略普通构建失败。

## 两种运行范围

`scripts.run_mlx_spike_graph --asset-source files` 保留原有完整任务分派及执行后logits/token检查，只替换资产来源。命令仍嵌入小ELF并保留48MiB命令容量预检和链接后64MiB CPU内存/栈边界检查；数据资产不再占该ELF容量。

`--asset-source files --load-only` 则仅执行CPU资产复制和目标摘要核对：源算子执行数、设备计算窗口数均为0，不执行推理，也不声称检查了推理输出。它用于在启动新的完整计算前验证全部真实权重确实进入了对应设备地址，不能代替模型端到端验证。

runner为每次尝试复制独占插件二进制，记录 `execution.json` 中的PID、来源/输入/二进制身份、watchdog与终态。终态后还需通过资产完整性及结果审计才能生成 `report.json`；仅有进程退出或execution记录不能视为通过。宿主运行时间不解释为MLX性能。

## 当前证据

[file-asset-source-001](../artifacts/tagged/file-asset-source-001/report.json)完成14项回归：文件偏移超过4GiB、非页对齐文件切片、缓存边界、只读/越界/重复/截断拒绝、literal精确打包，以及正常与权重扰动生成图。45个源节点图在嵌入资产和文件资产两种方式下的全部设备报告相同，仅文件方式额外包含装载审计；全部logits/token仍在实际ELF中检查。

错误摘要和未写入目标地址均在计算访存前被拒绝。普通与ASan/UBSan图、load-only以及两个独立C++文件读取探针结果一致；独立探针开启泄漏检查，第三方Spike插件运行仍关闭LeakSanitizer。开发失败目录保留，不将其补标为通过。

[139项相关回归](../artifacts/tagged/file-asset-source-tests-001.xml)通过。完整公开Llama2的[只装载尝试已结束并通过](../artifacts/tagged/model-e2e/llama2-cpu-assets-001/report.json)：360个资产共13476831558 bytes，其中291个checkpoint参数资产共13476831232 bytes，另有真实初始输入和literal。实际RV64执行1684603950次源读取，逐资产恰好覆盖全部字节；设备CPU写入字节数相同，360个目标摘要全部匹配，失败读取/遗漏为0。设备计算launch、读写和源算子执行数均为0，结果明确不作为推理证据。

ELF的text/data/bss合计18012 bytes，不包含完整权重；目标实际分配3290244个数据页，设备写入代数页为0，符合只由CPU装载的范围。该证据绑定当时源码/独占二进制；此后为通用主机增加RoCC传输分支，不将旧证据重新标注为新传输或实际Chipyard装载通过。

```bash
.venv/bin/python -m scripts.run_mlx_spike_graph \
  --program artifacts/tagged/model-e2e/llama2-physical-full-001/program.json \
  --lifetimes artifacts/tagged/model-storage-002/model-lifetimes.json \
  --asset-source files --load-only --device-bytes 17179869184 \
  --timeout 172800 --output artifacts/tagged/model-e2e/llama2-cpu-assets-NEW
```

仅用于在没有同一活动尝试时重放，输出目录必须新建。活动运行状态需检查PID/会话，不能因观察超时重启；运行期间保持execution来源集合不变。这个装载测试与仍在进行的完整物理推理是两个独立范围，不复用对方的通过结论。

```bash
.venv/bin/python -m scripts.verify_mlx_asset_source \
  --output artifacts/tagged/file-asset-source-NEW
```

本路径仍是文件支持的测试输入区和轮询驱动的设备时钟，不是真实Rocket/HellaCache、磁盘/DMA装载时序或作者MLX hybrid模型验证。完整模型结果、真实系统时钟和其他要求模型仍需逐一验收；性能与RTL扩展门槛继续关闭。
