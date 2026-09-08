# 模型内存计划与 RISC-V 控制指令原型

2026-09-07。承接[浮点向量微程序](mlx-vector-microcode.md)，继续补齐完整模型中的非矩阵/非浮点调用。仍先实现 C++ 行为，再接系统，不修改 RTL。

## 内存与布局路径

新增 `--memory-backend planned`。编译器 [model_memory_program.py](../src/mlxsim/model_memory_program.py) 跟踪每个 SSA 值的 dtype、shape、元素 stride、offset、存储根和存储容量；[memory_program.cc](../simulator_ext/memory_model/memory_program.cc) 独立检查运行时输入布局并消费计划。

| 类型 | 编译/执行规则 |
|---|---|
| transpose、slice、select、unsqueeze、expand | 生成可检查的仿射视图，不分配新数据；运行时重新推导布局并核对共享存储 |
| view / reshape | 按连续子空间计算合法 stride；支持 stride=2 等非密集但可表示的 view。不可表示的 `view` 明确拒绝；`reshape` 可生成实际复制计划 |
| cast / contiguous | 区分同 dtype 的真实别名、强制 copy、布局规范化和类型转换；`preserve_format` 在非重叠密集布局上保留 stride，非密集布局则物化 |
| cat / cache 拼接 | 逐段选择实际来源并写出新缓冲；只有允许的 `[0]` 空输入可忽略，其他空输入仍需满足非拼接轴约束 |
| embedding | 从实际 int64 索引选择权重行，检查行范围，再读出数据 |
| where | 先读真实 Boolean predicate，再选取张量或标量来源，按输出类型转换；int64 值不经 float 中转 |
| detach / lift / 推理 dropout | 明确的无数据修改别名；训练态 dropout 不可消除 |

视图语义与复制条件参考 [PyTorch view](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.view.html) 和 [Tensor.to](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.to.html)。追踪现在把 `memory_format` 保存为实际枚举名称，不丢失为未知 Python 类型；这遵循本次使用的追踪技能，只补元数据，不增加中间张量拷贝。

复制计划使用四个 64-bit 数据寄存器和 128-byte 输入/输出暂存区，每个小块不超过 64 bytes。短模板包含 LOAD、CONVERT、STORE；embedding 增加索引装载，where 增加 predicate 装载。控制层计算形状/地址，数据实际经过装载、转换和存储，不把整张量复制记为零成本视图。相同 dtype 的搬运保留原始位，包括 FP16 NaN payload 和符号零。

这些是本项目的搬运控制描述符，不冒充论文 PE opcode 或已完成的 HellaCache DMA。不同 device 的转换不得作为别名。`planned` 保留同步功能实现；新的 `scheduled` 通过公共端口执行，支持物理地址绑定与响应驱动，但尚未连接真实 HellaCache，见下节。

完整公开 Llama2 已在 [`llama2-memory-001`](../artifacts/tagged/model-e2e/llama2-memory-001/numeric-conformance.json) 完成 2,659 个内存计划：1,739 个检查过的视图和 920 个实际物化操作，实际读取 38,235,409 bytes、写出 37,732,558 bytes，写出量与编译工作量约束一致。三个 logits 向量与既有数值合约结果逐位相同。该数是 C++ 行为搬运量，不是系统 DMA 性能测量；此运行当时还有 30 次控制调用未进入新控制入口。

其余30个源调用为：15个arange、9个整数add、3个le、3个argmax。控制入口已接入同一张量执行器；[`llama2-all-lowered-001`](../artifacts/tagged/model-e2e/llama2-all-lowered-001/numeric-conformance.json) 已完成全部6181个调用，旧功能辅助入口为0，声明数值合约下96000个logits逐位一致。仍不是Rocket或完整系统执行通过。

## 响应驱动的 C++ 搬运状态机

`--memory-backend scheduled` 选择 [memory_model::Simulator](../simulator_ext/memory_model/memory_schedule.h)，`--memory-schedule-options` 提供独立的 JSON 参数。它复用经过检查的布局/搬运模板，不调用同步 `execute` 来预先计算输出：

```text
索引/predicate请求 → 响应装入寄存器 → 选源地址
  → 数据LOAD请求/响应 → CONVERT发射/完成 → STORE请求/响应 → 下一元素/退休
```

每次仅有一个在途访存，保持四个64-bit数据寄存器、128-byte输入/输出暂存区；另显式记录一个64-bit转换结果锁存与一个64-bit请求数据锁存。局部搬运槽总在64-byte块内，尾部不产生越界访问。LOAD_INDEX、LOAD_PREDICATE 和 LOAD 的数值只能来自响应，STORE 的实际数据来自转换寄存器；响应确认前不能返回完整输出。回压不改变请求归属或寄存器值，响应消费与依赖指令发射不在同一周期。一次转换的延迟与访存延迟分别配置；外部端口的实际响应决定访存完成，不用本地倒计时替代它。

输入region按 `input_layouts` 的名称字典序排列，输出为最后一个region。外部模式要求调用者提供输出容量/布局，Tensor的data/writable允许为null。视图重新推导stride/offset并保留原Storage所有权，不分配新数据、不发访存；这只表示设备数据访问被消除，不表示真实CPU处理描述符不耗时。空输出也校验完整指令模板；embedding_dim=0仍通过响应逐个校验索引。

[绑定源码的组件证据](../artifacts/tagged/memory-window-001/report.json)包含76项相关回归、18个成功搬运/布局用例。覆盖四种dtype、非连续与stride=2布局、根SSA释放后的别名、实际索引/predicate与标量选择、FP16 NaN payload/符号零、非法转换、错误/变化/撤回响应和物理权限检查。故障分别作用于公共地址适配器及无保护的逻辑测试端口，确认状态机自身也拒绝坏响应。重试组合例有336个逻辑请求、403次下游接受、67次nack，实际提交168读+168写，没有重复写入。

编译验证同时重新处理完整公开Llama2清单：870 matrix、2622 vector、2659 memory、30 control，每个节点恰好一个后端；新旧执行节点、权重绑定与token依赖一致，只切换内存后端/周期选项。该组件报告明确 **只执行组件与三步组合生成图，没有执行完整模型**。

随后独立的[完整模型重跑](../artifacts/tagged/model-e2e/llama2-memory-scheduled-001/numeric-conformance.json)已结束：全部6181个调用实际执行，BLAS=0、未编译功能入口=0；2659个memory窗口的32446193次请求和响应全部排空，实际读取38235409 bytes、写出37732558 bytes。291个参数张量均消费，声明数值合约下96000个logits逐位一致，token/cache链核对通过。原GPU比较仍有1145/2524/22个logits超差，阈值未放宽。此run的matrix/vector仍是功能微程序、控制仍是rv64_leaf，不能把局部memory周期相加当完整推理性能，也不是Chipyard系统验收。

ASan/UBSan的带重试布局链、外部embedding与slice/select/设备复制链均通过，输出、完整事件和计数与release一致，证据在 `artifacts/tagged/memory-window-asan-001/`。局部周期不是推理性能；当前host控制、全局跨算子时钟/资源和真实系统缓存仍未闭环。

## 控制指令原型

最新增量见[控制张量周期组件](mlx-controller-window-simulation.md)：叶指令逐条推进，公共MemoryPort提供张量数据；`--control-backend scheduled` 已接主runner并参与四类后端组合图。新267项相关回归及191个实际Spike预期对照通过。历史 `rv64_leaf` 完整模型运行仍属于原功能入口，不能改标为新控制周期运行；真实RISC-V load/store与Rocket循环/ABI仍待完成。

[model_control_program.py](../src/mlxsim/model_control_program.py) 生成真实 RV64I/M/F 编码的寄存器指令片段；[control_model](../simulator_ext/control_model/control_program.cc) 在 C++ 中解释这些片段，张量地址/循环由有界绑定层提供。它不是完整 CPU 模拟器，未执行 RISC-V load/store 或真实 Rocket 取指，也不计主机周期。

`--control-backend rv64_leaf` 启用此路径。与矩阵/向量微程序及内存计划同时启用时，当前 Llama2 的 6,181 个源调用均恰好对应一个执行路径：870 matrix、2622 vector、2659 memory、30 control。原功能辅助入口计数单独记录，完整 lowering 不允许再进入该入口。零指令视图仍由独立布局检查器验证，不能直接跳过。

| 算子职责 | 指令语义 |
|---|---|
| arange | ADDI 维护实际计数寄存器，输出来自指令结果 |
| int64 add / mul | RV64 ADD/MUL，保留完整 64-bit 数据并按模 2⁶⁴ 回绕 |
| 比较 | 整数 SLT/XORI 或浮点 FLE.S；浮点输入使用 FP32 比较域 |
| argmax | 流式保留 best value/index；SLT 或 FCLASS.S/FLT.S 产生更新条件，BEQ 控制索引与 best 更新，保持第一个最大值/NaN 规则 |
| 转换检查 | 子集实现 FCVT.S.L RNE，用于独立 ISA 一致性测试；不据此给 PE 增加 int64 算术或 Zfh 单元 |

寄存器状态为 32 个 64-bit GPR 与 32 个 64-bit FPR，检查 x0、单精度 NaN boxing 和 fflags。编译片段总长度不超过 32 words；执行分支不得越界或无限循环。依据为 [RV64I](https://docs.riscv.org/reference/isa/v20260120/unpriv/rv64.html) 与 [F 扩展](https://docs.riscv.org/reference/isa/v20260120/unpriv/f-st-ext.html)，这些是已有主机 ISA 语义，不是推测的 MLX PE 整数功能。

验证分两层：

- GNU RISC-V 汇编器生成的真实机器字与 Python 编码器逐字一致；20 项控制组件测试覆盖整数精度/回绕、NaN 分类/boxing、FP flags、转换、条件更新和非法分支拒绝。
- [`controller-isa-003`](../artifacts/tagged/controller-isa-003/report.json) 将 191 个独立预期用例同时交给 C++ 片段解释器和实际 Spike ELF 执行，全部通过。int64→FP32 的预期通过整数位算法计算，避免先转 double 导致双重舍入。Spike 使用本地未修改的 `bf4b1e09ed8e7a11ecff9891b12ce5d7f3375722` 源码构建；只作 ISA 功能参考，不等于 Rocket/Chipyard 系统验证。

## 当前回归与剩余工作

[当前 158 项组件/集成回归](../artifacts/tagged/model-e2e/memory-control-tests-002.xml)通过，其中内存/组合图新增 24 项、控制原型新增 20 项。内存测试覆盖别名在根 SSA 释放后仍可用、非连续 view、必须复制的 reshape、混合 dtype、位级复制、实际索引/predicate 数据、布局与程序损坏拒绝。新增组合图包含三个生成步骤、真实 cache 拼接、归约、矩阵、mask 和 argmax，验证四类路径同时工作且 `functional_entry_calls=0`。

仍需完成：

1. 功能全模型已完成数值/覆盖核对；继续扩展输入与模型范围，并推进完整周期/系统执行，不能用组合小图替代它。
2. 为控制绑定层补实际 host load/store、循环/ABI 以及 Rocket/系统执行证据；当前只能证明 RV64 叶指令语义。
3. 矩阵、向量和新搬运周期组件已有公共访存端口；继续接入控制绑定、统一跨算子就绪/完成事件和真实系统缓存，补齐 CDC/路由与完整模型周期执行。
4. 扩展要求模型和输入集合。旧的单输入数值合约通过、矩阵周期组件、控制 ISA 对照均不能替代完整系统验收。

空 embedding 的边界已在冻结运行结束后修正：即便 embedding_dim=0，仍加载并校验每个索引；非法负值或越界不能因输出为空而被忽略。该修正不改变当前 embedding_dim=4096 模型的编译产物。

所有新增报告继续保持系统/Rocket/推理性能未验证。不得用搬运条数或 ISA 指令数外推完整推理性能，也不启动新的 RTL 实现。
