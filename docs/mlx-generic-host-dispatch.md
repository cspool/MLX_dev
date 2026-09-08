# 编译生成的通用 RV64 主机编排

2026-09-08。新增 [graph_runtime.c](../system_sim/physical_host/graph_runtime.c) 与 [graph_lowering.py](../system_sim/physical_host/graph_lowering.py)，将已验证的四类路由组织成通用任务表。测试固件不再手写embedding、矩阵、softmax或argmax的调用顺序；它只调用通用调度器并在完成后检查结果。

## 编译与执行流程

```text
完整编译图 + 已检查的存储生命期
  → 重定位物理绑定、资产装载表、任务表与二进制命令
  → RV64 CPU实际复制资产、按表分派
      ├─ 主机控制：执行C/RV64例程
      ├─ 设备任务：提交wire、等待完成
      └─ 视图：已验证布局消除，不提交伪造搬运
  → 每源算子完成标记 → 最终输出字节核对
```

每个任务携带源序号、源operator_id、batch序号/数量、命令地址与长度。调度器检查顺序、任务种类、命令尺寸及范围，只有实际任务完成后才更新源完成表。矩阵一个源调用的多个batch必须顺序排空，不能仅处理首个batch后算作源算子完成。

资产从编译图的真实literal或文件offset生成payload，CPU通过实际load/store将其送入设备映射内存；初始资产与中间结果分开，参考输出不作为任务输入。整数literal不经float窄化，FP16/Boolean初始化保持既有C++路径的FP32中间舍入规则。

参考字节只出现在执行后的检查器中，不传给通用调度器。CPU使用实际argmax结果写入编译图中的token缓冲，后续embedding仍从该缓冲取索引。失败任务不会被记为完成；轮询超时返回错误，调用者必须继续保有相关存储并排空/终止执行世界，不能直接复用地址。

## 当前实际证据

[graph-dispatch-001](../artifacts/tagged/graph-dispatch-001/report.json)通过正常与权重扰动两张完整三步生成图：45个源节点、15次主机控制、21次设备任务和9个布局消除。全部源完成标记及最终logits/token字节在真实RV64 ELF中核对。扰动后参考重新计算，token变为0/0/0；没有回填参考token。

[graph-runtime-001](../artifacts/tagged/graph-runtime-001/report.json)额外用实际RV64执行15项调度器契约检查：非法头部、任务种类/顺序/batch、缺失源完成、错误主机命令、非法设备命令以及装载越界/覆盖scratch等。非法任务未触发设备launch。

[117项相关回归](../artifacts/tagged/graph-dispatch-tests-001.xml)通过。普通和扰动图的ASan/UBSan插件结果与release的完整device.json一致，文件位于 `artifacts/tagged/spike-graph-asan-001/`。仍只检查插件/C++依赖，第三方Spike LeakSanitizer关闭，不声明整个Spike无泄漏。

## 完整 Llama2 编排计划

同一通用编译器已生成完整Llama2的6181源节点、12133任务项及33286912 bytes的实际命令数据；其中1739个view任务没有执行描述符。所需映射范围为13481947840 bytes，包含对齐、地址复用产生的最大地址跨度及保留区。

稀疏存储改动后的[完整计划重编译](../artifacts/tagged/graph-plan-sparse-001/report.json)已完成；plan.json和command_blob.bin与graph-dispatch-001的完整计划逐字节相同。这仍只是编译/地址绑定等价检查。

该结果位于证据中的 `full-model-plan/`，**仅是计划，未装载权重或执行完整模型ELF**。[C++稀疏映射测试内存](mlx-sparse-mapped-memory.md)已解除原插件1MiB限制；后续[文件资产来源](mlx-file-asset-loading.md)让CPU从独立输入区执行真实load/store，不再要求大权重嵌入ELF。默认embedded方式仍限制命令加资产48MiB，files方式只对命令作此预检，均检查64MiB CPU内存边界。完整资产必须实际复制、读回核对才能声明装载通过，不能凭逻辑容量或编译计划替代。

```bash
.venv/bin/python -m scripts.verify_mlx_graph_dispatch \
  --program artifacts/tagged/model-e2e/llama2-physical-full-001/program.json \
  --lifetimes artifacts/tagged/model-storage-002/model-lifetimes.json \
  --output artifacts/tagged/graph-dispatch-NEW
.venv/bin/python -m scripts.verify_mlx_graph_runtime \
  --output artifacts/tagged/graph-runtime-NEW
```

输出目录须选新路径。`scripts.run_mlx_spike_graph` 是通用入口，接受编译程序、生命期和独立结果参考；`--plan-only`明确不执行。它记录命令、payload、ELF、编译输入、参考文件与源代码身份。`--device-bytes`设置逻辑测试窗口，`--data-offset`设置相对设备基址的数据偏移，两者都不修改目标RF/SPM或访存延迟。

## 剩余系统工作

独立[外部时钟设备控制器](mlx-clocked-device-controller.md)已用总线响应取得描述符和张量数据，并验证纯状态查询、重试、错误排空与reset身份隔离。后续[RoCC传输分支](mlx-clocked-rocc-integration.md)已在真实Rocket/cache中通过正常与权重扰动两张45节点图，原Spike插件分支仍保留。两个传输使用同一任务分派/完成算法，但属于不同构建与执行证据。

下一步是将完整模型资产流入接到实际系统地址空间，扩展并核对实际外存配置、完整模型编排以及时序/资源边界。Spike分支仍是轮询推进设备tick的测试内存；新RoCC分支虽由实际系统边沿驱动，但目前仅验收登记图，不能作为完整模型性能。长期运行的 `llama2-physical-full-001` 使用冻结的共享原生执行器，没有被新主机或RoCC路径替换。

后续[独立16GiB配置](mlx-wide-system-memory.md)已补齐高位地址与受检初始化，并在高地址通过相同正常/扰动图；完整系统镜像组合与全部模型执行仍待完成。ELF预装载属于未计时程序初始化，不能借此省略或伪称CPU/权重搬运性能。

完整要求模型和输入集合的结果核对完成前，系统成功、推理性能和RTL门槛继续保持关闭。
