# 完整模型RTX4090实测：GPU模拟误差的硬件参照

2026-09-09已执行完整模型，而不是单层或FLOPs外推。每个模型加载后，将实际GPU参数逐张量复制回CPU，与checkpoint位模式核对；正式测量前执行独立参考比较与插桩等价检查，核对每个forward的全部层和参数消费。Python只驱动真实GPU测量，不执行MLX周期模拟。

## 实测结果

同一RTX4090，10次预热、50次重复，eager attention，TF32及低精度归约关闭。权重与初始输入驻留，Llama2每次重复重新创建KV cache；生成输入使用实际预测token，不喂入参考token。

| 完整任务 | CUDA流区间中位数 | 同步模型wall time中位数 | 含任务后处理wall time中位数 |
| --- | ---: | ---: | ---: |
| BERT三组QA：28/64/64 tokens | 21.964 ms | 21.991 ms | 22.213 ms |
| Llama2：8-token prefill + 两次decode | 104.608 ms | 104.644 ms | 104.650 ms |

CUDA流区间的P10–P90分别为BERT 21.716–22.953ms、Llama2 101.429–109.730ms。逐forward中位数：BERT 6.936/7.439/7.446ms；Llama2 prefill 36.659ms、decode 33.876/33.712ms。合计中位数不是各forward中位数相加，统计由原始样本独立重算。

测量范围包含正常CUDA并发执行能力，没有将SM/block/warp串行化。这里的CUDA event区间是当前stream从开始到结束的经过时间，可能包含主机launch供给空隙，**不是kernel时长简单求和**，也不是已测得的纯计算时间。BERT后处理包含双logits读取和相同portable C跨度选择；Llama2 greedy EOS判断和token反馈在模型循环中。checkpoint加载、tokenization和初始H2D不计入这些驻留测量，不能称冷启动端到端性能。

本次应用按PyTorch eager默认方式在当前CUDA stream提交；BERT三组独立用例按顺序提交，Llama2 decode按真实token/cache依赖推进。这不关闭kernel内部SM/block/warp并发，但尚未测多请求、多stream并发吞吐。GPU模拟误差实验必须回放相同提交策略；若MLX将多个独立QA请求同时准入，不能直接拿本文三组顺序提交的合计时间作公平的跨架构速度比。多请求并发应另建两侧一致的实验，不重标本次数据。

## 正确性与运行条件

- Llama2为公开dense基座，32层、6,738,415,616参数、291张量；6181个实际GPU源调用与原完整参考覆盖一致。三步token393/372/338、KV长度8/9/10。96000个FP16 logits与原GPU参考逐位一致，50次重复输出稳定。
- BERT为本地dense QA baseline，12层、108,893,186参数、199张量；1141个实际GPU源调用与原完整参考覆盖一致。三组答案Zurich/Zurich/Oslo，双logits通过预先设定的`.005 + .005*abs(reference)`条件，50次重复输出稳定。不是结构化作者hybrid。
- GPU UUID为`GPU-619e0c86-ae77-e523-7e85-97195a01f08b`，SM89；驱动595.84。测量区间交叠的频率样本：BERT8个、Llama2 38个，SM2535MHz、显存10251MHz，采样节流mask均为0。仅是这些时刻的观测，不声称捕获所有瞬态。
- 未锁频、未改变450W功耗上限。运行前存在约43MiB的图形进程，环境快照保留，不声称整卡完全独占。五个CPU模拟任务同时运行；GPU脚本CPU线程为2，wall time包含当时主机环境影响。
- 18项计时/统计/任务反馈测试通过。审计器从实际文件重算统计、误差、token/答案和频率交叠，核对完整模型层/参数/调用覆盖及来源摘要。

归档入口为[`gpu-measurement-publication-001`](../artifacts/tagged/gpu-measurement-publication-001/manifest.json)，包括原始样本、计时协议、频率记录、实际logits、源算子清单、参考与代码摘要，不含真实权重或运行库二进制。

## 尚不能得出的结论

这些是GPU硬件测量，不是GPU模拟结果，也不是GPU模拟误差已经通过。尚需匹配并校准RTX4090模拟器后按`(T_sim-T_real)/T_real`计算误差。

MLX事件驱动／端到端误差是另一组实验，两者都须保留并发和相同资源约束。不能用上述GPU时间、历史串行MLX周期或CPU宿主模拟耗时替代该误差。当前也没有GPU warp发射效率、有效lane和cache/DRAM细分计数，因此不从这些总延迟直接断言资源利用的根因；后续需独立profiling和机制消融。
