# C++ 稀疏映射测试内存

2026-09-08。系统主机桥原先为整个映射窗口预分配数据、有效位和逐字节写入代数，并限制在1MiB以内。新增 [mapped_memory.cc](../system_sim/physical_device/mapped_memory.cc) 按实际写入分配4KiB页，支持大地址偏移的真实字节读写。它只改变模拟器的宿主存储表示，不引入目标硬件分页、缓存、换页延迟或额外RF/SPM。

## 数据与完成约束

- 每页包含4096个数据字节和512-byte初始化位图；未写过的字节不能读。读取会先验证完整区间，再复制数据，不能在后半段失败时向调用者留下部分有效描述符。
- CPU装载只分配数据页和位图。设备首次写入某页时，才为该页分配4096个64-bit写入代数；每次launch的输出必须逐字节匹配本次代数。CPU覆盖不更新设备代数，不能满足下一次launch的写满要求。
- 地址、长度及边界使用64-bit检查；跨页读写保持准确的字节尾部。越界、空指针非零访问和非法代数明确拒绝；零长度访问不分配页。
- 设备仍只有原先的访存事务、响应延迟与完成规则，忙时CPU仍不得访问payload。页持续保留至设备实例销毁；这里没有新增页回收或目标存储容量复用策略。

插件要求基址和容量按4KiB对齐，容量至少8KiB，映射末端不超过本测试接口的40-bit地址上限。这是软件测试接口边界，不是对论文地址宽度或实际SoC配置的推断。

`host_memory_backing`分别报告逻辑容量、驻留页、数据字节、位图字节和写入代数字节。统计不包含map、指针、分配器等开销，不能当作总RSS，也不能当作MLX硬件存储面积。

## 大地址空间与资产装载是两个条件

`scripts.run_mlx_spike_graph`支持 `--device-bytes` 和 `--data-offset`。同一编译图可以在16GiB逻辑窗口内，将所有数据移动到距设备基址超过4GiB的位置，由实际RV64代码装载、提交设备任务并核对输出。参考字节仍只用于完成后的检查器。

默认embedded方式仍将输入资产和命令嵌入CPU低地址ELF：命令加资产超过48MiB时，在读取/复制权重文件前拒绝；链接后还检查程序和栈没有超过配置的64MiB CPU内存。后续新增[文件资产输入区](mlx-file-asset-loading.md)，允许CPU从独立文件支持的地址区复制真实权重而不嵌入ELF；装载完整性仍需实际执行并核对，**解除插件1MiB限制本身不等于完成14GB级权重装载**。完整编排计划也不是模型ELF执行证据。

后续需要单独实现并验证完整资产流入、CPU/设备共享地址及访问权限、真实外存接口和时钟耦合。完整物理模拟运行 `llama2-physical-full-001` 的冻结源码没有被本改动替换；该run也不能重新标注为实际CPU装载。

## 验证与重放

正式证据为 [sparse-mapped-memory-003](../artifacts/tagged/sparse-mapped-memory-003/report.json)：8项新增回归通过，低/高偏移图完整device.json相同，29个历史dense用例除新增`host_memory_backing`字段外逐字段相同；独立契约及全部图/链的sanitizer结果一致。16GiB窗口生成图实际驻留5页：数据20480 bytes、位图2560 bytes、设备写入代数32768 bytes；不是16GiB模型数据已消费。

矩阵/向量/搬运wire、主机控制/编排、联合ELF与稀疏内存的[125项回归](../artifacts/tagged/sparse-mapped-memory-tests-001.xml)全部通过。

早期001因测试输出父目录未创建而发生fixture错误，没有通过报告；002为修正启动目录后的8项开发检查。003使用统一脚本创建独占目录、绑定来源并完整重放，作为本次正式证据。

验证入口 [verify_mlx_sparse_memory.py](../scripts/verify_mlx_sparse_memory.py) 包含：

1. C++直接契约：16GiB容量、跨4GiB偏移、跨页/尾部、未初始化失败不部分复制、逐字节写入代数及3000步独立dense字节参考随机对照。
2. 同一45源节点、三步生成图在低/高偏移的实际RV64执行；全部源完成标记和最终logits/token字节检查，比较完整设备事件与计数。
3. 对历史29个矩阵/向量/搬运/控制ELF用例重放；除新增的宿主存储统计外，设备报告必须与旧dense实现完全相同。
4. 独立C++契约启用ASan/UBSan及泄漏检查；插件图和29个链用例启用ASan/UBSan，第三方Spike的泄漏检查仍关闭。普通/插桩版本的结果与计数必须一致。

```bash
.venv/bin/python -m scripts.verify_mlx_sparse_memory \
  --output artifacts/tagged/sparse-mapped-memory-NEW \
  --dense-baseline artifacts/tagged/spike-memory-chain-003
```

必须选择新输出目录。通过后的报告绑定源码、旧基线报告、实际ELF结果和sanitizer结果；失败不会生成通过报告。这里的检查不放行完整模型、Chipyard系统或推理性能门槛。
