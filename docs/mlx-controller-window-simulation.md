# 控制张量的响应驱动 C++ 组件

承接[内存/控制计划](mlx-memory-control-plans.md)，将arange、整数add/mul、le、argmax的张量绑定推进到公共访存端口。冻结重跑结束后，主编译器/runner已接入 `--control-backend scheduled`、`--control-schedule-options` 和逐调用 `control_windows`，缺失控制lowering明确拒绝。[control-window-002](../artifacts/tagged/control-window-002/report.json)完成267项相关回归、191个实际Spike预期对照及四类后端三步生成图。新[共享物理执行器](mlx-shared-physical-model.md)也已使用此控制路径；完整模型在这些新路径上的执行与真实Rocket门槛仍未完成。

## 实际执行范围

[control_schedule::Simulator](../simulator_ext/control_schedule/control_schedule.h)接收既有 `mlx-controller-rv64-leaf-v1` 描述符，持有独立的寄存器、phase、PC、行/列位置、在途指令与访存身份。输入region 0/1对应前两个张量参数，输出region 2；标量是描述符常量，不产生伪造的内存读取。

```text
实际LOAD请求/响应 → 寄存器绑定 → 一条RV64叶指令发射/完成
  → 显式PC/分支/下一phase → STORE请求/响应 → 下一元素或行
```

RV64寄存器算术和浮点语义复用已经验证的解释器，每次只执行一个非分支机器字。BEQ/BNE由单步器维护phase内PC，检查目标范围/对齐和256步上限；未走到的指令不退休，不把整个phase先计算完再按长度补周期。最多32个模板字，32个64-bit GPR和32个64-bit FPR（共512-byte架构寄存器），x0、NaN boxing和fflags规则保持不变。这是控制处理器状态，不是给每个MLX PE扩大RF容量。

访存和执行均最多一项在途。输入只在收到匹配响应后绑定到寄存器；一条指令实际到达完成点才修改架构状态；依赖操作下一周期才能发射。输出来自x12或argmax索引寄存器x20，必须等实际写响应确认才退休。默认 `TensorMemoryPort` 仅为独立行为内存，外部端口允许Tensor的data/writable均为null。张量步长与offset参与每次地址计算，响应未使用的高位不能污染Boolean输入。

argmax在每行流式保留best value/index，读取下一候选、递增真实计数寄存器，经比较与条件分支决定是否更新；保留首个最大值和首个NaN语义。不会读取参考token来给出结果。arange(0)仍执行初始化叶指令，但不发访存；其他空输出验证模板后不执行元素循环。

## 不是完整RISC-V CPU，也尚不是系统计时

当前取指来自描述符ROM，张量LOAD/STORE仍由C++绑定层通过MemoryPort产生，**没有执行RISC-V LD/ST或指令缓存取指，也没有运行Rocket ELF中的完整控制循环/ABI**。FP16输入提升、literal绑定与形状循环是明确的绑定服务，不冒充新增RISC-V扩展或论文PE opcode。

ALU、MUL、FP、branch延迟和请求/响应反压可独立配置。这些是组件参数，不是实测Rocket流水线；当前不存在指令/访存重叠或真实CPU缓存模型。组件cycles不能作为主机开销或完整推理性能填入系统报告。

## 当前证据

[control-window-001](../artifacts/tagged/control-window-001/report.json)绑定源码、程序、二进制和测试输出，完成241项相关回归。其中新组件221项，包含191个逐条RV64预期用例和21个成功张量/周期运行。另用实际GNU RISC-V工具链生成ELF，在本地Spike执行同一组191个独立ISA预期，全部通过。Spike对照只证明叶指令语义，不证明张量循环/访存或Rocket时序。

张量用例检查四种数据类型、64-bit整数回绕、广播、外部真实数据与假占位数据的区分、非连续argmax、首个tie/NaN、保留维度、空输出和同工作量下反压/延迟变化。错误身份、error、无请求响应及回压中改变/撤回响应均不能生成成功报告。同步功能绑定仅在目标执行完成后用作对照，不向目标提供输入或中间结果；还逐项核对指令、分支和读写字节计数。

ASan/UBSan覆盖带NaN的非连续argmax、外部整数乘法/回绕与空arange。输出、完整事件和计数与release逐字节一致，文件在 `artifacts/tagged/control-window-asan-001/`。

可重放命令，证据目录必须选未存在的新路径：

```bash
.venv/bin/python -m pytest -q tests/test_control_window_scheduler.py
.venv/bin/python -m scripts.verify_mlx_control_window \
  --output artifacts/tagged/control-window-NEW
```

## 下一步

补充的[实际RV64主机控制代码](mlx-rv64-host-control.md)已在独立Spike ELF中执行真实load/store与循环，完成40个用例、65536种FP16模式及主机ABI字节核对；完整Llama2的30个控制调用也已有命令编译绑定。这不是当前C++控制组件已经被真实CPU替换，模型runner和Rocket/HellaCache集成仍需继续。

1. 主编译路由、C++张量入口、source_identity及组合图已接入；继续在新路径执行完整模型，不能把旧rv64_leaf功能运行重新标为控制周期运行。
2. 共享原生物理缓冲/时钟和pin已有组合图证据；继续实现真实装载、跨算子共享资源/CDC与系统级错误排空。
3. 补真实RISC-V主机代码、load/store、循环/ABI与Rocket/HellaCache执行证据，接完整模型的端到端系统结果对照。要求模型及输入集合通过之前，不进入推理性能或RTL扩展。

历史001报告保持 `integrated_into_model_runner=False`；新002报告通过独立集成用例记录为True。两者都保持 `rocket_execution_verified=False`、`mlx_system_verified=False` 和 `inference_performance_eligible=False`，不能回写旧报告扩大通过范围。
