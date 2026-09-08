# 原生整图宿主开销：按生命周期变化触发缓冲清理

本次在可开发的 `mlx-host-opt-v1` 工作树中修改 C++ 主机端实现，四个完整模型尝试的冻结来源、输入和二进制均未更改。没有跳过模拟边沿、修改算术、扩大资源或更改 RTL。

## 完整图的受限定位

[完整图受限插桩](../artifacts/tagged/full-graph-host-hotspots-001/report.json)保留原6181节点、873配对及全部真实权重，只在20000000周期时明确拒绝继续执行，退出码1且没有result.json。此记录不是缩小模型的通过结果，也不是完整推理证书。

插桩记录中 `PhysicalMemory::collect()` 调用约227万次，占约18%的采样self-time。原 `launch_ready()` 在每个物理端口空闲的边沿扫描所有弱引用；大量边沿之间没有源完成或张量引用释放，扫描不会回收任何内容。插桩时间只用于定位宿主开销。

## 修改与语义边界

`ready_graph.cc` 增加内部 `collection_pending` 标记：源/批次完成、执行器引用被释放时置位；下一个物理端口和响应分发器均空闲的边沿仍按原时机清理，然后复位。没有引用释放的边沿不再重复扫描。

此优化只作用于 `Runner` 已知的生命周期位置，没有缓存通用 `PhysicalMemory::collect()` 的结果或改变其接口。新绑定内部的清理检查、最终读回后的无条件清理仍保留。矩阵跨batch执行、视图别名、输入消费和最终输出保留都继续使用原规则，物理地址复用时机不变。

## 验证

[196项回归](../artifacts/tagged/collection-tests-001/regression.xml)通过，覆盖完整登记图、正常/扰动生成、配对事件、生命周期及0/32/512个常驻附加输入。

[旧/新独立二进制对照](../artifacts/tagged/collection-equivalence-001/report.json)对13个完整图重放，实际输出、事件、模拟周期及资源记录完全相同，并通过ASan/UBSan/泄漏检查。基线 `host-opt-candidate` 已含前一轮失败准入缓存，故本对照隔离本次清理修改。

完整模型的受限窗口同样保留全部图，旧/新都在20000000周期退出1，源begin/complete日志逐字节相同；没有导出完整目标状态或推理结果。单次非插桩宿主窗口耗时为42.47秒与34.94秒，仅作诊断记录，不视作稳定基准或完整模型加速比。

脚本为 `scripts/profile_mlx_full_graph_host.py` 和 `scripts/verify_mlx_ready_collection.py`。本阶段不重启或替换四个长任务，也不以宿主实现改进打开原GPU数值、全部模型/输入、系统推理性能或RTL门槛。
