# 四类后端共享物理内存的 C++ 模型执行器

新增 [model_system](../simulator_ext/model_system/model_executor.cc)，将矩阵、浮点向量、搬运和控制周期组件接到同一原生物理地址空间。它不再为每个算子单独绑定一组本地Tensor数据指针。此处的“物理”指显式地址、权限、生命周期和请求/响应，不表示已经运行在Chipyard缓存或真实Rocket主机上。

## 实际数据路径

```text
初始资产字节/权重文件偏移 → 预装载地址绑定
  → Arena虚拟Tensor / pin → 后端MemoryPort请求
  → AddressSpacePort / 全局请求ID → 注册队列、回压与重试
  → 共享内存实际读写 → 响应/完成 → 下一算子及自生成token
```

编译程序必须显式选择四个 `scheduled` 后端；每个源节点恰好一个可执行计划。任何功能后端选择、缺失或重复lowering都拒绝，不能在虚拟Tensor不可读时偷偷回退。主张量入口也已新增 `--control-backend scheduled` 和 `--control-schedule-options`，并输出逐调用的 `control_windows`；其集成证据为[control-window-002](../artifacts/tagged/control-window-002/report.json)。

Arena负责分配/回收和只读权限；输入pin与输出写pin持续持有到后端完成且队列排空。内存端点只保存弱的虚拟Storage引用，不替SSA延长设备分配生命期；过期映射在安全点回收，实际背板随之释放。地址复用取得新的分配身份，跨算子局部请求ID通过共享RequestTokens重新绑定。即便Arena随后将缓冲seal为只读，端点也查询当前权限，不沿用过期的可写标志。

新输出的宿主背板可以是零填充数组，但其设备字节最初标记为无效。只有真实store提交才置有效位；读未初始化字节或在输出仍有未写字节时宣告完成均拒绝。有效位和响应确认是不同条件：算子还必须等写响应排空，才能释放pin或把输出交给下一算子。已预装载的输入明确标记有效，不伪造一串初始化DMA。

四类后端共享一个单调原生时钟；每个窗口的开始/结束都由实际tick和端口响应推进，不能用独立执行报告代填。矩阵batch广播与非连续B寻址、bias的独立只读region、视图别名、cache拼接、mask和argmax均走此路径。最终logits/token通过同一端口读回后才写报告，不直接读取虚拟Tensor，也不注入参考中间值。

## 当前验证范围

[physical-model-002](../artifacts/tagged/physical-model-002/report.json)绑定源码、二进制、编译输入与结果，覆盖33项相关测试、8个成功图执行；[广泛回归524项通过](../artifacts/tagged/model-e2e/physical-integrated-tests-002.xml)。

- 三步生成/cache图的45个节点同时使用四类周期后端。带重试用例363个请求、72次nack，实际234读+129写=363，响应和所有缓冲最终排空；与独立执行的三个logits向量逐位一致。
- 扰动实际embedding权重后，结果与重新计算的PyTorch参考一致，生成token由1/2/2变为0/0/0，后续logits也随实际自生成输入改变。
- 文件权重使用真实绑定offset读取；另检查批广播、非连续矩阵、bias、地址复用、关闭trace、容量不足和禁止回退。
- 直接内存契约测试证明：零值也必须由store产生；部分未初始化读取、封为只读后的写入、已退休地址及外来Arena存储均拒绝。

ASan/UBSan通过普通生成/重试、权重扰动、批矩阵与直接内存契约。logits逐字节相同；仅归一化报告中的输出文件路径后，完整事件/计数与release相同。文件位于 `artifacts/tagged/physical-model-asan-001/`。

同一报告还将完整公开Llama2重新编译为870 matrix、2622 vector、2659 memory、30 control，源节点、资产和最终输出依赖与历史完整程序不变。**这部分是全模型编译核对，不是新物理执行器已执行完整Llama2。** 历史 `llama2-memory-scheduled-001` 的96000-logits数值通过仍属于其当时源码和后端组合，不移用为本路径通过。

## 明确保留的系统缺口

独立[RV64主机控制运行时](mlx-rv64-host-control.md)已补实际ELF的load/store、循环和argmax证据，但尚未连接到本执行器或Rocket；当前完整物理运行的后端/源码身份不因此改变。

随后[主机/矩阵联合桥](mlx-spike-host-matrix-bridge.md)在独立Spike插件中连接了实际CPU与矩阵后端；只证明登记链式用例的数据流，不是此完整模型执行器已经使用真实CPU，也不是实际系统时钟/内存连接。

1. 当前数据是模拟器预装载。实际权重字节及文件offset进入内存，但没有CPU/DMA装载指令、取指/启动、缓存一致性或冷启动周期证据。
2. 控制执行器逐条解释RV64叶指令，tensor I/O与外层循环仍为描述符绑定；没有真实Rocket ELF控制程序。现有tagged native-v2的小负载Chipyard ABI不能直接视为支持本Tensor程序或新FP32 profile。
3. 跨算子执行目前串行。各算子内部仍有真实有界多上下文调度，但跨算子共享RF/SPM仲裁、CDC/路由和多层重叠尚未完成，不能并行启动多个独立SPM实例来假装共享资源。
4. 全模型的新路径执行尚未完成。默认每窗口周期上限适合组件用例，不保证覆盖完整尺寸；正式运行前需根据真实工作量配置上限并评估模拟器执行成本，不能缩小模型或按FLOPs外推。
5. 当前全局周期只有已建模后端/内存及抽象读回部分；真实主机和系统开销未建模，不是TTFT、吞吐或完整推理时延。空矩阵batch在此入口明确拒绝，不能把缺少窗口的空分支当作已验证lowering。

失败会中止本原生执行世界，不生成成功报告；尚不提供真实SoC的忙时reset/错误恢复。连接HellaCache后必须在错误恢复中继续持有pin，直到外部事务真正排空，不能直接销毁上下文后重用地址。

## 重放

完整模型的新运行入口是 [run_mlx_physical_model.py](../scripts/run_mlx_physical_model.py)。它编译四类scheduled路径，复制attempt独占二进制及源文件快照，绑定程序/系统配置/运行库摘要，并在运行前后核对完整权重与tokenizer身份。成功退出还需通过[物理执行证据检查](../src/mlxsim/model_physical_evidence.py)：逐节点与batch窗口覆盖、共享时钟、指令/访存计数、诊断与最终读回、参数预装载及最终资源排空。随后保存原GPU误差比较，不放宽门槛；声明数值合约另由 `scripts.verify_mlx_model_numeric` 检查。

`execution.json` 的completed仅指C++进程完成，不等于数值或系统验收。宿主watchdog或目标周期上限终止时保留失败状态，不生成完整模型比较通过；宿主耗时与目标模拟周期分开报告。

新矩阵 `cache_control` 是宿主实现优化，缓存只随准入/退休变化的块优先级、准入空间与资源检查结果，PC/FU/DMA仍每周期真实仲裁。[matrix-control-cache-001](../artifacts/tagged/matrix-control-cache-001/report.json)的54项相关回归及A/B对照证明输出、逐事件trace和目标计数不变。受控用例两种实现均为355272目标周期，控制重算由355272次降为72次；记录的宿主中位时间约2.155/0.061秒，不是模型推理加速比。

限额全尺寸诊断 `llama2-physical-cost-002` 已完成prefill第0层Q矩阵（8×4096乘4096×4096）。[65536-byte输出摘要与登记参考一致](../artifacts/tagged/model-e2e/llama2-physical-cost-002/observation-conformance.json)；随后在算子53触发10亿共享周期上限，仍是未完成诊断。按[观察方案](mlx-physical-observation-plan.md)只做完成后的有界摘要读回，参考值不进入计算进程；额外读回周期单列。宿主时间显示Q计算约131.35秒，不能用它替代MLX目标推理时间或外推论文性能。

当前555项广泛回归通过，见 `artifacts/tagged/model-e2e/physical-runner-tests-001.xml`；摘要观察的ASan/UBSan数据和事件也与release一致。新的完整运行已启动于 `llama2-physical-full-001`：每窗口上限10^12、共享上限10^13，宿主watchdog为172800秒；只提高保护上限，不改变模型尺寸、工作量、延迟或资源容量。状态以[实际运行续接记录](../artifacts/tagged/model-e2e/active-validation.md)与进程为准，尚无完整结果。

```bash
.venv/bin/python -m scripts.verify_mlx_physical_model \
  --inventory artifacts/tagged/model-e2e/llama2-reference-005/inventory.json \
  --prior-program artifacts/tagged/model-e2e/llama2-memory-scheduled-001/program.json \
  --output artifacts/tagged/physical-model-NEW
```

底层入口为 `build/mlx-physical-model/mlx-physical-model program.json system-options.json output-directory`。system-options包含明确的Arena base/bytes、共享max_cycles，以及memory latency/accept_period/nack_every/trace_limit；示例见证据目录各用例。它只接受四类scheduled程序，没有BLAS或Python执行选项。

可选 `operator_progress` 输出带source_operator_id/forward/layer/phase的宿主单调时间；`observe_operators`只对显式选择的已完成输出做SHA256。单项最多1MiB、合计32MiB，采用流式端口读回，不保存整张量副本。观察不会改写模型值，但产生额外总线访问，不能宣称有/无观察的全部周期相同。

下一步仍按“C++完整模型正确性 → 真实系统集成与结果对照 → 推理性能 → RTL”的门槛推进，不能以本组合图或编译清单替代要求模型及输入集合的验收。
