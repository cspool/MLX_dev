# 共享物理模型的限额观察方案

按 `trace-patch-target-discovery` 的边界选择方法制定本方案；不修改模型算法、权重或算子算术。完整推理仍是验收对象，局部观察只用于诊断。

```text
Question:
  完整Llama2 prefill到达首个真实尺寸linear时，前处理与矩阵各耗费多少模拟器宿主时间？
  该linear的实际物理输出是否与已登记数值参考一致？

Patch targets:
  1. 已有逐算子执行边界
     why: layer边界不能区分RMSNorm与Q矩阵，单条DMA边界又过细。
     join keys: source_operator_id、forward_id、layer_idx、phase、单调事件序号。
     fields: 算子开始/结束/失败、共享目标周期、独立标注的宿主单调时间。
  2. 明确选中的算子完成且请求排空之后
     why: 只有真实结果已产生，才允许观察；不能观察输入并替后端计算答案。
     join keys: source_operator_id、forward_id、layer_idx、输出dtype/shape。
     fields: 已读字节数、输出SHA256摘要、诊断读回的起止周期；不保存整张量。

Do not patch:
  不逐lane/逐指令增加日志；不读取参考文件进入计算进程；不拷贝完整权重/大张量。
  不把观察文件作为SSA资产，也不将参考token或激活回填到模型。

Validation:
  同输入有/无观察的确定性logits与token一致；目标算术/原有访存量不变。
  摘要与独立参考文件SHA256比较；观察次数与显式ID集合一致，具备完整连接键。
  单个观察最多1MiB、总计32MiB；经公共端口有界流式读回，不分配整张量副本。
  诊断读回会增加额外总线访问/目标周期，必须单列；不伪称有/无观察周期相同。
```

默认关闭逐算子宿主时间与输出摘要观察。首次诊断仅选择prefill第0层的source_operator_id=50，不由框架高层模块名猜测矩阵边界。已有完整参考清单/诊断中的同ID输出作为进程外对照，摘要观察器不接收它的数据路径。

超时或周期限额结束的运行即使产生部分摘要，也仍是未完成诊断，不产生全模型成功或推理性能证书。
