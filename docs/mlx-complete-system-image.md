# 完整模型系统镜像与资源参数

2026-09-08。新增 [model_image](../system_sim/model_image/image.py) 与 [C++参数解析器](../system_sim/model_image/profile.cc)。现在可以把完整编译图、模型资产、数值参考和后端参数组合为真实RV64 ELF及初始RAM清单。镜像准备、初始化、实际推理和性能验收分别记录。

## 镜像包含什么

当前完整公开dense Llama2-7B镜像为 [llama2-system-image-001](../artifacts/tagged/model-e2e/llama2-system-image-001/report.json)：

| 内容 | 已生成的完整范围 |
|---|---:|
| 源算子 | 6181 |
| CPU任务表项 | 12133（含1739个合法视图消除） |
| 实际命令数据 | 33286912 bytes |
| 初始资产 | 360个，13476831558 bytes |
| ELF常规区段合计 | 34362000 bytes，未嵌入巨型权重副本 |
| CPU资产复制 | 本驻留模式为0 bytes，明确不算冷启动搬运 |

完整层数、hidden size和权重规模没有缩减。指令、Tensor地址、batch数量及源完成表都由原完整图生成；不能只执行首个batch就标记源算子完成。完整命令ELF放在低地址，图基址为`0x88000000`，为代码和栈保留128MiB；数据和程序共同受16GiB物理范围检查。

`--preload-assets`将已声明的输入/权重作为驻留RAM初始状态。资产仅取自编译图的`assets`，不包含参考中间激活；参考logits/token只进入执行后的检查器。CPU仍执行完整任务表和控制算术，后续输入使用真实argmax结果。这个模式不声称模拟了CPU/DMA冷启动装载，已有CPU装载证据也不会被移用为该模式的性能证明。

## 后端资源不再固定为演示值

`--device-profile`经过与运行时相同的C++解析器验证，记录全部有效参数。未提供配置时仍使用原单PE演示模式；显式完整配置采用4×4、每PE两个上下文，以及原生矩阵/向量/搬运的延迟、发射周期和10^12窗口上限。

解析器拒绝未知字段、缺失后端配置、非法周期、不同的PE/context几何和测试故障注入。profile不提供任意增加RF/SPM的接口。DPI保存实际有效配置，runner必须将其与准备时配置逐字段核对；不能只传入配置文件却继续用硬编码单PE执行。

真实外部内存响应仍决定访存完成时间；后端选项中的本地`dma_latency`不被伪称为系统cache/DRAM延迟。全系统周期上限和宿主watchdog也单独记录，均不是性能结果。

## 已有证据及其边界

- [llama2-system-image-audit-002](../artifacts/tagged/model-e2e/llama2-system-image-audit-002/report.json)：重新编译完整源/生命期后，命令和绑定与镜像一致；与数值参考编译程序比较只允许15个arange和3个cast_device的设备放置归一化。生成反馈链核对通过，设备profile与原生模型参数一致。
- 同一审计实际在C++内存对象中共同装入完整ELF及360个资产，验证区间不重叠、文件字节/零尾部及目标摘要。**没有CPU或模型执行，AXI/时钟计数为0。**
- [resident-profile-chipyard-001](../artifacts/tagged/resident-profile-chipyard-001/report.json)：4×4参数和驻留资产模式在真实Rocket正常/扰动45节点图中通过；各21设备任务、15个CPU控制调用、9个视图消除，全部输出字节及有效profile匹配。这不是完整Llama2执行证书。
- [system-profile-001](../artifacts/tagged/system-profile-001/report.json)绑定14项profile/镜像检查，8个C++参数场景的ASan/UBSan/泄漏检查一致；[184项相关回归](../artifacts/tagged/resident-system-image-tests-001.xml)通过。

原GPU误差门槛的失败记录仍保留。此镜像绑定显式数值合约参考，不表示原GPU数值门槛已经通过，也不代表作者MLX hybrid模型已经绑定或验收。

## 复现与后续

仅准备完整镜像，不启动推理：

```bash
.venv/bin/python -m scripts.run_mlx_clocked_chipyard --phase prepare \
  --large-memory --preload-elf --preload-assets \
  --graph-base 0x88000000 --graph-bytes 17045651456 \
  --device-profile artifacts/tagged/full-system-profile-001.json \
  --max-system-cycles 10000000000000 --watchdog-seconds 172800 \
  --cases artifacts/tagged/full-system-image-cases-001.json \
  --output artifacts/tagged/model-e2e/llama2-system-image-NEW
```

输出必须选新目录。实际完整系统运行仍未完成，不应把上述命令的`prepare`改标为运行通过；系统镜像、模型/参考身份、初始化范围及后端profile必须作为同一尝试绑定。完整原生周期运行仍使用其冻结实现，不能重新标为已使用新CPU ELF。完整模型系统正确性、其余要求模型和输入范围通过前，性能与RTL扩展继续暂停。

## 已启动的完整系统尝试

后续已启动`artifacts/tagged/model-e2e/llama2-rocket-full-001/llama2-resident/`：真实Rocket、完整6181源/12133任务、全部360资产，4×4后端profile、16GiB RAM和既定数值参考；没有缩减模型。源码在该attempt的`sources/`，模拟器为独占副本，实际PID/终态见用例的`execution.json`。

同一尝试先完成[开/关只读观测的实际系统预检](mlx-system-run-observation.md)，再启动完整用例。完整ELF及资产的361个初始化段已装入实际系统内存，随后开始执行真实设备任务。当前没有完整模型终态或通过报告；原生完整周期运行与此Rocket尝试分别追踪，不混用结果。

宿主watchdog为172800秒、系统周期上限10^13；这些是运行限制，不是性能结果。运行期间保持记录中的来源/输入/二进制不变，观察超时不重启。终态后还须进行完整模型输出、路由、初始化范围及原GPU/显式数值合约的分别审计。

## v2 配对路径的独立镜像

后续 [v2 大地址空间与完整镜像验证](mlx-paired-wide-system.md)已生成保留全部 6181 源、873 对和 8284 任务的完整镜像，360 个资产及 CPU ELF 的共同初始化、逐字节重建均通过。五个高地址 4×4 登记图也已在实际 Rocket 通过。上述旧版完整运行仍保持冻结版本，不能由这些前置检查推导 v2 完整模型已经执行通过。
