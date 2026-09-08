# 完整系统长时运行的只读观测

本轮按`trace-patch-target-discovery`选择边界，目标是恢复实际执行进程，不改变模型工作量或数据。

Question:
  完整Rocket系统正在初始化、等待CPU提交、读取描述符、执行哪一层/哪一步的设备任务，还是排空错误？

Patch targets:

1. 宿主进程生命周期与初始化完成。
   why: 区分尚未进入CPU执行、运行中、watchdog和真正终态。
   join keys: attempt目录、PID、ELF/程序/profile摘要。
   fields: 启动/结束时间、退出码、初始化摘要、状态；不记录张量值。
2. C++ RoCC提交接受和设备任务终态边界，另加稀疏心跳。
   why: 这些边界对应真实编译任务，能连接到forward/layer/batch。
   join keys: 提交序号与编译的launch-map，关联source_operator_id、source_ordinal、forward_id、layer_idx、batch_index。
   fields: phase、周期、请求/响应计数、busy/完成/错误、描述符进度、宿主单调时间。

Do not patch:
  不修改CPU指令、采样/argmax、算子微程序或数值执行；不复制大张量，不向执行器回填参考值。心跳不是层完成证据，设备提交日志也不代替CPU控制/视图的源完成检查。

Validation:
  比较开/关观测时的输出及完整设备事件/周期；检查每次提交/终态数量与映射一致；限制事件日志长度并保留最新快照。观测I/O错误不能改变计算结果，必须单独记录。源码和独占二进制在运行期间冻结，观察等待超时不重启任务。

## 实施与验证

`clocked_rocc/Progress`只在提交/终态变化和选定心跳周期读取标量状态。`progress.json`原子替换为最新状态，`progress.jsonl`最多保留50000条事件；达到上限后最新快照仍更新并标记日志截断。日志写入失败停止观测并单独记录，不修改数值计算。`memory-init.json`在RAM初始化完成、内存时钟尚未推进时记录摘要，不把它当CPU执行。

宿主runner现在保存独占源码快照、模拟器副本、ELF/profile/输入及动态库摘要；逐用例`execution.json`记录PID和终态。watchdog明确记为未通过，不假称请求已排空。固定种子用于实际系统A/B对照。

[12项过程/观测回归](../artifacts/tagged/system-attempt-tests-001.xml)验证输出、事件和周期在开/关观测、日志限额以及I/O失败情况下保持一致。[191项相关回归](../artifacts/tagged/full-system-runner-tests-001.xml)通过。

[实际Rocket A/B证据](../artifacts/tagged/model-e2e/llama2-rocket-full-001/preflight-equivalence.json)使用同一ELF、模拟器及种子73129：两次45节点图的完整设备报告相同（仅移除观测器自身统计），21次提交和21次终态均能按launch-map关联。该预检通过后才启动完整模型系统用例。

`progress.phase=done`只表示最近的设备任务结束；不能据此推断CPU剩余控制任务、整个forward或完整模型已经结束。完整成功仍以实际ELF最终输出检查及独立模型门禁为准。
