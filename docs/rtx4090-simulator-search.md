# RTX4090 / Ada / SM89 模拟器检索

检索日期2026-09-09。查询对象是建模RTX4090架构的模拟器，不是“使用4090运行的物理/机器人模拟器”。本轮使用GitHub仓库、issue/评论API及作者发布材料；没有可检索的独立知识库连接。尚未安装新模拟器或把社区结果计为本项目实验。

## 结论

**找到Accel-Sim/GPGPU-Sim上的RTX4090实验适配和实卡相关性记录，可作为候选起点；尚不能称为直接可用、完整校准的标准4090模型。** 上游标准配置未列4090只是一项事实，不等于不存在4090适配。正式GPU实验要保留SM/block/warp并发，并与本机4090实测校准。

| 候选／证据 | 已确认内容 | 使用边界 |
| --- | --- | --- |
| [Accel-Sim #518](https://github.com/accel-sim/accel-sim-framework/issues/518) | 公开RTX4090参数、配置修订及实卡统计；2026-01-01评论报告10个应用、193个kernel的GPC周期相关系数0.9735，工具输出Err=10.14% | 社区报告，不是本项目重现；同份记录的L2读命中、DRAM读计数仍有明显误差，不能只摘周期相关性当全模型准确性 |
| [Ada支持 #405](https://github.com/accel-sim/accel-sim-framework/issues/405)及其[指向的回复](https://github.com/accel-sim/accel-sim-framework/issues/380#issuecomment-2669830168) | 用户报告通过修改trace版本接受/复用Ampere opcode映射运行Ada trace | 只说明一部分trace可解析执行，不证明覆盖全部Ada指令、Tensor Core或精确时序 |
| [实验配置 #409](https://github.com/accel-sim/accel-sim-framework/issues/409) | 明确使用`experimental-cfgs/SM89_RTX4090`与tuner/白皮书生成配置 | 该issue含死锁；不能把名称存在当成验收通过 |
| [ARC作者论文](https://www.embarclab.com/static/media/arc.3f76b7ecf6e4d4fcda8f.pdf)§6/Table1 | 使用GPGPU-Sim的4090-Sim研究原子归约 | 发布artifact主要为ARC-SW/3DGS实卡实验；当前未确认完整4090硬件模拟源码公开，不能仅下载[软件仓库](https://github.com/Accelsnow/gaussian-splatting-distwar)就当作取得了该模拟器 |

周期统计的直接来源为[2026-01-01评论](https://github.com/accel-sim/accel-sim-framework/issues/518#issuecomment-3703487717)。它报告的Err保持原工具标签，不擅自改写为未经核对公式的MAE/MAPE；评论同时报告L2 Read Hits Err=13143.03%、DRAM Reads Err=90.64%。这些计数误差与周期误差不是同一指标。

## 适配风险与具体线索

[早期workaround](https://github.com/accel-sim/accel-sim-framework/issues/518#issuecomment-3677412698)包含复用RTX3070 ISA与trace配置，并改变内存分区以避开hash限制；不能原样视为4090真实硬件。[后续配置修订](https://github.com/accel-sim/accel-sim-framework/issues/518#issuecomment-3682278255)改到12个内存通道与总72MiB L2，并继续比较计数。应使用明确版本、实测延迟/带宽和完整容量核算，而不是为了让模拟器运行而任意改变资源。

维护者指出tuner的L2问题，并提供[framework修复PR512](https://github.com/accel-sim/accel-sim-framework/pull/512)与[微基准修复PR75](https://github.com/accel-sim/gpu-app-collection/pull/75)。应用修复后应重采对应硬件统计，不能混用旧统计名和新配置。实际采纳时需锁定合并状态、commit、配置和NVBit/驱动版本，单独保留本机595.84驱动的兼容性结果。

本轮GitHub API确认二者均已合并，合并commit分别为`963526a41ad8477a8098138008a2653ea7cfd901`和`4becbe3988470b7a040ec4b473f07e4df834fe9c`。这是修复的来源信息，不等于已在本机运行或解决全部4090建模误差。

已只读核对本地版本：Accel-Sim `c5296df152c99a28dd64e5d9560bd58a8fd2e774`，GPGPU-Sim `68e1cd30efaecbd71b496822f9d88a5803b33841`，NVBit安装脚本为1.7.3。上游主树`d930ad6d02c09bb56867132583735aba0389cff4`的递归文件清单未发现4090/Ada专用配置或opcode文件；[适配作者fork](https://github.com/PrabinKuSabat/accel-sim-framework)当时dev树`8c74a5ee03660c1a94a2a5a083a8e2e8c033579b`也未公开带4090/SM89名称的文件。当前最具体、可审查的适配材料在issue配置与评论中，仍需组合并本地验证。

## 本项目下一步

先保留完整真实GPU测量，随后在隔离目录尝试该RTX4090配置路线，不修改已有冻结模拟器。校准至少包括指令/warp数、GPC周期、L1/L2访问/命中、DRAM流量及本机时钟；先独立校准，再用实际Llama2/BERT kernel作为验证集，避免用目标整模型性能反向拟合配置。缺指令/死锁/计数错误必须显式拒绝或标注，不能用RTX3070/3090代理冒充已验证4090。

GPU最终误差为GPU模拟对实机误差；MLX最终误差为保留并发的事件驱动模型对端到端执行模拟的误差，见[协议](mlx-gpu-performance-comparison.md)。
