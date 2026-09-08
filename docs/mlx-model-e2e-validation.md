# 模型级端到端验证：先正确性，后性能

这是 2026-09-07 用户补充的验收要求。验证对象是完整模型推理，不是孤立算子、缩小 Transformer block 或按层数外推的运行时间。已有 native/Chipyard 小负载证书只证明其登记范围，不证明完整模型已经在 MLX 系统中正确运行。

## 当前事实与优先级

已经实现的 C++ 调度模型、系统接口与 RTL 组件保留为基础。当前优先补齐模型 → 编译 → C++ MLX 后端 → 系统执行链，暂停进一步 RTL/PPA 扩展；完整模型正确性通过之后才验证推理性能。

最新新增 `--memory-backend scheduled`：2659个内存/视图调用进入响应驱动的C++搬运状态机。独立[完整Llama2重跑](../artifacts/tagged/model-e2e/llama2-memory-scheduled-001/numeric-conformance.json)已经结束，6181个源调用全部执行、每个恰好一条路由，291个参数张量均消费；声明数值合约下96000个logits逐位一致，token/cache依赖通过。32446193个memory请求/响应排空，BLAS=0、旧功能辅助入口=0。原GPU比较仍失败，记录和阈值保留。该run的matrix/vector为功能微程序、控制为rv64_leaf，尚非完整系统/全周期执行，因此不开放推理性能门槛。

后续已将控制scheduled选择接入主runner，并将[四类后端与物理缓冲接到共享原生执行器](mlx-shared-physical-model.md)：524项回归包含生成/cache、实际权重扰动、地址复用、写入有效性和重试。新路径只有组合图实际执行与完整Llama2编译核对；完整模型重跑、真实装载/Rocket/HellaCache和跨算子共享资源仍未验收，不能移用上述历史数值结果。

最新完整物理路径已启动 `llama2-physical-full-001`，在实际结束前不标记通过。此前限额诊断已完成真实尺寸首个Q矩阵，65536-byte输出摘要与登记数值参考一致，但运行在后续算子触发周期上限，不能代替端到端验收。新runner绑定源码快照/独占二进制/运行库，物理证据门禁检查全部窗口、目标时钟、访存和资源排空；555项回归通过。状态与终态处理见[活动记录](../artifacts/tagged/model-e2e/active-validation.md)。

冻结运行之外，[通用RV64主机编排](mlx-generic-host-dispatch.md)已实际执行45节点三步生成图，并生成完整Llama2任务计划；[稀疏C++测试内存](mlx-sparse-mapped-memory.md)通过16GiB窗口/跨4GiB偏移执行和29个旧存储实现等价检查。[文件资产输入区](mlx-file-asset-loading.md)已完成公开Llama2全部360资产、13476831558 bytes的实际CPU复制及目标摘要核对。该尝试没有执行模型算子，也不是Chipyard文件装载时序；装载不等于推理，不计作ME3通过。

另在冻结集合外实现[外部时钟/总线设备控制器](mlx-clocked-device-controller.md)，并经[真实Rocket/RoCC与cache](mlx-clocked-rocc-integration.md)通过正常/权重扰动45节点图和3个阻塞WAIT联合链。实际系统曾暴露窄store总线展开错误，修正后才通过全部输出字节检查。该阶段证据基于256MiB配置和登记图，不能把Spike完整资产装载、小图系统执行和独立完整模型数值结果拼成ME3证书。

后续新增[16GiB系统配置](mlx-wide-system-memory.md)：修正地址边界的32-bit截断/取模风险，正常/扰动图在高地址实际通过。完整360资产也已在同一个C++内存类中初始化并核对，但该初始化没有执行CPU或模型算子；仍需完整模型系统镜像与真实全图运行。ELF预装载单列为未计时初始化，不作为CPU/DMA时间或完整推理性能。

[完整系统镜像](mlx-complete-system-image.md)随后已准备并审计：6181源、12133任务、完整权重和输出检查器不缩减；同一RAM中程序/资产初始化和参考程序绑定通过，4×4有效profile与原生配置一致。驻留资产模式在实际Rocket小图验证，但完整模型系统执行仍未完成。该模式不声称CPU/DMA冷启动性能，原GPU数值差异仍保留。

第一份完整参考证据位于 [`llama2-reference-001`](../artifacts/tagged/model-e2e/llama2-reference-001/summary.json)：实际加载 6,738,415,616 个参数、32 层、hidden=4096 的公开 Llama2-7B 权重，执行 8-token prefill 和两次 1-token decode。无插桩/有插桩的三个 logits 向量和 token 均逐位相同，KV 长度为 8/9/10；总计 6,181 次调用、31 种算子。它明确标为**参考执行与算子清单，不是 MLX 模型执行**。后续已实现这些调用的 C++ 张量语义编译和执行，但 MLX 硬件映射、完整数值验收与系统执行仍未通过，见下文。

当前采用 [`llama2-reference-005`](../artifacts/tagged/model-e2e/llama2-reference-005/summary.json) 作为已绑定源码和精度策略的 GPU 参考：所有 291 个参数张量均有 checkpoint 绑定，无缺失/形状不匹配；仅显式允许旧 checkpoint 的 32 个非持久 RoPE 缓冲，由当前框架按已记录配置处理。三个 forward 各记录全部 32 层的 K/V 绑定与长度。004 起额外绑定真实初始输入、当前 RoPE buffer 和小常量；005 增加首层有限观察点，插桩与无插桩输出仍逐位一致。001–004 保留历史，不混用其源码/精度设置。

观察到 FP16、FP32、int64、bool 混合数据类型。现有向量 IR 的 ADD/FMA/EXP 等原语并不等于具有 ATen 张量算子的 lowering：还缺完整张量分块、广播/步长、权重流入、精度/累加规则、归一化/归约、索引与 KV cache 路径。不能把 dense `Linear` 仅凭名字归成 BSMM，也不能把同精度要求悄悄降成纯 FP16 累加。

## 模型身份必须分别绑定

| 对象 | 本地可用材料 | 不能混淆的身份/待办 |
|---|---|---|
| Llama2-7B | `third_party/models/llama2-7b-hf-msmirror`；两份权重 SHA 与既有官方权重记录一致 | 这是公开 dense 基座，不是作者完整 MLX hybrid 权重 |
| 本地 Llama2 FFT/LoRA | `third_party/checkpoints/llama2-fft-lora-s075-run023` 及 H20 配置 | 配置明确缺 BSMM、无 KV cache，不能当完整 MLX 或标准 cached decode；若纳入，必须按完整前缀重算语义单独验证 |
| InternLM2-7B | `third_party/models/internlm2-7b-msdownload` 有实际权重 | `internlm2-7b-modelscope` 内 `.bin` 是 135-byte LFS 指针；不得把指针当权重。基座、chat、作者 hybrid 分开登记，并绑定其自定义模型代码和运行环境 |
| BERT QA/结构化重建 | `bert-squad-baseline-run014`、`bert-structured-distilled-h38/k*` | 后者必须先构造对应结构再严格加载；不能用 vanilla BERT 忽略缺失/额外权重。模型任务质量与模拟器输出一致性分开报告 |
| ViT、FABNet | 有论文和部分来源/工作负载材料 | 完整拓扑、权重、预处理与任务头尚未绑定，不能用单层模板或随机模型代替模型级通过 |

论文算法实验与 Table III 架构工作负载的尺寸/变体也不能自动视为同一 checkpoint。作者压缩模型与本地公开/重建模型的选择已请求用户确认；在此之前，公开基座可用于前端和算子覆盖诊断，不计为作者 hybrid 验证通过。

## 实施与退出条件

1. **ME0 模型/输入固定**：记录 checkpoint、tokenizer/预处理、完整模型源码/配置、变更层与结构化参数、dtype/累加策略、输入及生成策略的摘要。不改层数、hidden size 或权重规模来取得通过。
2. **ME1 全算子编译覆盖**：从真实前向/生成取得节点、shape、dtype、标量参数、依赖及 cache 绑定；每个节点必须有可调用、已验证的 MLX 算子 lowering 或底层指令生成路径。布局别名、常量折叠和推理 dropout 消除也要有语义证明，不能直接漏掉节点。未支持的算子或精度/形状必须明确拒绝。
3. **ME2 C++ 模拟器完整执行**：同一权重与输入真正进入 C/C++ MLX 后端，完整层、任务头、输出与必要的生成/cache 状态推进都执行。允许有明确合约的后端 MLX 复合算子；不允许用 PyTorch/GPU/NumPy 回退执行缺失的模型计算，再把结果记成 MLX 输出。
4. **ME3 系统级结果对照**：将同一编译产物接入系统，核对实际输入/权重/中间数据搬运、配置、同步和最终结果。逐阶段误差界必须在运行前按精度约定冻结，同时检查最终 logits/任务输出以及确定性 token/类别/QA 结果。只跑参考模型、只比较哈希清单或给模拟器注入 golden 输出均不能通过。
5. **ME4 推理性能**：只有所要求模型及输入用例的 ME0–ME3 全部通过才进入。报告 prefill、逐 token decode、TTFT、吞吐及 host/config/DMA/kernel/同步周期，说明冷/热启动、缓存和装载范围；区分模拟目标周期、模拟器自身运行时间和参考 GPU 时间。禁止用缩小模型乘层数、按 FLOPs 外推或拟合论文加速比替代完整执行证据。

中间激活和 cache 必须来自模拟器前一算子的真实结果，不能逐层回填参考激活。生成验证采用模拟器自身的 logits/token 驱动后续步骤；逐步喂入参考 token 的 teacher-forced 测试不能替代 free-running 生成。参考中间值只能用于比较与定位差异。还应有输入/权重扰动用例，确认目标执行实际消费了配置的数据，而非重复返回预存结果。

至少覆盖正常输入、分块/尾部边界、padding/mask，以及生成模型的 prefill、持续 decode 和 cache 增长；较短输入的正确性不能替代长上下文性能用例。模型间逐一验收，不能只因 Llama2 通过便跳过 BERT、InternLM2 等所要求对象。

## 运行边界与插桩计划

问题：完整推理实际执行了哪些算子，分别属于哪一步、哪一层，以及何种精度和 cache 状态？

插桩点：

1. 生成/forward 边界：标明 prefill、decode、token selection；连接键为 request_id、forward_id，记录 q_len、past_len、kv_len、token_id。
2. 模型/层的前后 hook：标明全部层和任务头，连接 module_path、layer_idx；只记录输入输出的形状、类型和张量 ID。
3. ATen 分派边界：高层 hook 不足以列全算子，因此再记录实际 operator/schema、参数、输入输出 ID、shape/stride/dtype；这一级只建立清单，不冒充可执行编译 IR。
4. Cache 状态边界：记录每层 K/V 的张量绑定和长度，不复制完整 cache 内容。

不修改模型算法、权重或 CUDA 内核；不在热循环逐算子 `.cpu()`/`.item()` 或收集大张量。用于正确性比较的最终 logits 在追踪范围之外保存。验证要求是有/无插桩的确定性输出一致、层数与 forward 数一致、边界嵌套闭合、cache 增长可见。

为定位已发生的数值失败，新增**显式选择、限额、输出侧**观察：首层 48 个算子输出，每个不超过 1 MiB、总量不超过 32 MiB；先在原设备克隆，运行结束后一次性写出。观察文件与 `bindings`/可执行 `assets` 分离，不送回目标执行。连接键仍为 request/forward/layer/source_operator_id。对应 C++ 观察发生在实际算子完成后，并保持同样限额；普通运行默认关闭。此边界选择与有/无插桩等价检查遵循本次使用的 `trace-patch-target-discovery` 技能。

## C++ 张量语义前置实现：不能替代 MLX 系统门禁

新增 `src/mlxsim/model_tensor_compiler.py` → `simulator_ext/tensor_model/`：从真实完整模型清单生成带类型的 SSA 张量程序，再在独立 C++ 进程执行所有节点。权重由原 safetensors 数据偏移只读 mmap，初始输入/小常量显式绑定；中间激活与 K/V 由实际计算产生。每次 argmax 的输出接到下一次 decode，不从参考报告读取后续 token。

原生进程不调用 Python、Torch 或 GPU 后端。矩阵乘法使用显式绑定的 CPU OpenBLAS，FP16 输入提升到 FP32 累加，再舍入为输出 dtype；它目前只是**C++ 功能语义服务，不是 MLX 的 MAC 指令执行或时序模型**。不得将 SGEMM 次数、FLOPs、主机运行时间或物化字节数换算成 MLX 性能。

| 实际来源算子族 | 已实现的 C++ 语义入口 | 通向 MLX 后端/ISA 的剩余目标 |
|---|---|---|
| `linear`、`matmul` | `linear`、`matmul`，含批广播 | 权重流入、受限 RF/SPM 分块、实际 MAC/归约指令、累加精度与 tag 调度；不能因名称相似就当 BSMM |
| `embedding`、`arange`、`le`、`where`、`argmax` | 索引、比较、选择与归约 | 明确 host 控制与设备计算边界；全局地址/整数/Boolean 支持及实际访存路径 |
| `pow`、`mean`、`rsqrt`、`softmax` | FP32 归约/归一化语义 | 为 RMSNorm/softmax 固定具体归约顺序、指数/倒数模式、有效 lane 和精度；生成可执行复合算子/指令序列 |
| `add`、`mul`、`neg`、`silu`、`sin`、`cos` | 广播和 FP16/FP32 算术 | 映射已有原语或有依据的新模式，不能把存在数学函数误记为具有目标 FU/ISA 路径 |
| view/reshape/transpose/slice/select/expand/unsqueeze | 保留共享存储及逻辑步长 | 区分纯别名和必须的 gather/pack/xfer；后者必须真正执行且计入系统访存 |
| `cat`、`contiguous`、`to` | 实际复制、拼接和转换 | KV 存储生命期、DMA 地址、格式转换、可见性与同步 |
| `detach_`、`lift_fresh`、推理 `dropout` | 别名/有条件消除 | 保留来源到目标的覆盖记录，拒绝训练态 dropout 和未支持的可变写入 |

31 种实际 ATen overload 均有逐调用 `native_entry`，但 `mlx_hardware_mapping` 全部仍为 `pending`。因此它只完成 ME1/ME2 的张量语义前置子集，没有完成 ME1 的目标指令覆盖条件。

新增的 `--matrix-backend microcode` 已完成 675 次 `linear` 和 195 次 `matmul` 的完整 C++ 微指令执行，固定有限 RF/SPM 和显式 K 顺序，BLAS=0。另有 `scheduled` 后端接入矩阵 batch 内的固定 PE 映射、多上下文准入/发射、共享 SPM 与写回反压，但仍未完成跨算子层折叠或 Chipyard 集成，不能把上述 `pending` 直接改成完整硬件映射通过。编码、架构依据、数值合约及剩余职责见[矩阵微程序实施说明](mlx-matrix-microcode.md)。

后续 `--vector-backend microcode` 又为 2,622 次浮点算术、归约和非线性调用生成有界向量程序，覆盖 mean/softmax/RMSNorm 所需的基本计算，使用 8 个 RF 向量、320-byte SPM 工作集和至多 32-word ROM。[完整模型的声明数值合约一致性已通过](../artifacts/tagged/model-e2e/llama2-vector-001/numeric-conformance-002.json)：96,000 个 logits 逐位一致，所有参数引用、算子执行和自生成 token 依赖均核对。参考的矩阵/归约控制独立，但原子 libm 明确共享；仍不能把这个子门槛当作系统或作者硬件验收。实现和剩余范围见[浮点向量微程序](mlx-vector-microcode.md)。

数值模式需要结合完整硬件设计推导，不能只沿用当前小负载的 FP16 向量 ABI。论文 Principle 3 仅说明 FP16 是最低稳定精度；Power & Area 段提到精简版移除了高精度流水线；端到端段还明确完整设计支持 RMSNorm 和位置编码。这些是设计依据，**不等于论文公开了本项目所需 FP32 累加器宽度、舍入顺序或具体 opcode**。下一步应分别记录论文事实、重建推断和版本化编码，在 C++ 中补足模式与资源约束，再接系统，不能提前改 RTL，也不能扩大 RF/SPM 来隐藏分块缺口。

### 当前数值状态：声明合约一致，原框架差异保留，性能关闭

绑定当时源码与独占二进制的 BLAS 对照 [`llama2-native-003`](../artifacts/tagged/model-e2e/llama2-native-003/comparison.json) 完整执行了 6,181 个节点，32 层 × 3 个 forward 均完成，三个自生成 token 为 `393/372/338`（`that it is`），与 GPU 参考一致。但预先冻结的 logits 条件 `abs(error) <= 0.005 + 0.005 * abs(reference)` 未通过：最大误差分别为 `0.015625 / 0.01953125 / 0.01171875`，各 32,000 个 logits 中有 `140/708/31` 个超差。001/002 保留历史失败记录。`NATIVE_TENSOR_SEMANTICS_PASS` 日志只表示执行完成，不能覆盖独立比较器的失败状态。

后续 [`llama2-microcode-001`](../artifacts/tagged/model-e2e/llama2-microcode-001/comparison.json) 的完整矩阵路径已不使用 BLAS，实际执行 65,175,028,352 次有效 lane MUL 和同量 ADD；仍为完整 32 层、同一权重和输入、自生成 token 一致。原 GPU 门槛超差数为 `1145/2524/22`，不因更换后端而抹去。另完成[显式矩阵合约参考](../artifacts/tagged/model-e2e/llama2-reference-kasc-001/summary.json)：仅对矩阵采用独立逐 K FP32 运算，其他算子保持 CPU 框架；插桩等价、调用数 675+195 均核实。[对照该参考](../artifacts/tagged/model-e2e/llama2-microcode-001/kasc-numerical-diagnosis.json)仍有 `58/257/11` 个 logits 超差。首层所有已观察 FP16 输出相同，仅部分 FP32 归约/非线性中间值不同；还需继续检查后续层，不能直接把整个差异都归因于舍入。

首层诊断显示，RoPE 和 RMSNorm 虽有很小 FP32 差异，但转换后的 FP16 结果完全一致；首个 FP16 差异出现在 Q 投影 `v50`。对同一输入与真实权重进行仅用于诊断的 FP64 点积复核，Q/K/V 的 GPU 参考相对 FP64 舍入结果分别有 `362/390/731` 个元素不同，C++ 分别有 `40/22/56` 个不同（每个输出 32,768 个元素）。这支持矩阵累加/舍入路径差异的判断，不能据此宣布后续所有差异都已排除语义错误。见[GPU 诊断](../artifacts/tagged/model-e2e/llama2-native-003/gpu-numerical-diagnosis.json)。

另用同一完整权重和输入运行了[独立 CPU 参考](../artifacts/tagged/model-e2e/llama2-reference-cpu-001/summary.json)。它也有 6,181 个实际调用，插桩等价且 token 一致；[C++ 对 CPU 参考](../artifacts/tagged/model-e2e/llama2-native-003/cpu-numerical-diagnosis.json)的 logits 仍未通过原门槛，超差数为 `30/123/39`。CPU 与 GPU 两份参考之间本身也不能满足该门槛，超差数为 `70/1423/11`（记录在 GPU 诊断的 `independent_reference_comparison`）。因此不能简单地改换参考设备来取得通过，也不放宽门槛；后续需要明确可复现的目标归约/舍入合约并逐算子解释差异。跨平台不保证逐位相同亦见 [PyTorch Numerical Accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)；这不是跳过本项目验收的理由。

基础原生/编译/追踪回归有 20 项，覆盖全部语义入口、SSA 别名释放、广播/尾部与空 K、int64 精确比较/argmax、float→bool、实际权重/输入扰动、缺路由/缺绑定拒绝、训练态 dropout 拒绝、未知属性拒绝、观察限额和参考文件篡改/非有限 logits 拒绝。另有 16 项矩阵微程序/参考模式测试及 17 项周期/张量图集成测试，[共 53 项通过](../artifacts/tagged/model-e2e/matrix-window-integrated-tests-003.xml)。它们是组件正确性证据，不是完整模型或系统通过证据。

后续[整数精确 Q/K/V 诊断](mlx-exact-projection-diagnosis.md)覆盖同输入、完整权重和全部98304个投影输出。微程序与独立逐K FP32实现逐字节一致；相对精确点积最终FP16舍入，GPU差异数362/390/731，微程序64/62/93。cuBLAS/cuBLASLt及禁用split-K的九次GPU对照均复现历史GPU输出，未消除差异。该诊断不注入模型中间结果、不更改原参考/阈值，也不能替代后续层或端到端验收。

浮点向量、dtype 前置转换及模型数值验证器增量加入后，[当前共 114 项通过](../artifacts/tagged/model-e2e/vector-integrated-tests-005.xml)。原 GPU 比较仍保留失败，新增的数值合约通过也不会单独满足性能门禁；完整系统执行、剩余 2,689 个功能语义调用的目标/主机路径与所要求模型集合仍未完成。

后续[内存与控制增量](mlx-memory-control-plans.md)已将这些剩余调用全部分配到显式路径：2,659 个内存计划已完成真实完整模型执行与数值核对，30 个控制调用则已接入 RV64 叶指令解释器并启动新的完整重跑。当前 158 项回归通过，控制指令另有 191 个独立 Spike 对照通过。编译路径全覆盖仍不等于完整周期/Chipyard 执行通过；物理地址、host 访存/ABI、向量调度及跨算子 CDC/路由等仍待完成。

该完整重跑现已[验证通过声明数值合约](../artifacts/tagged/model-e2e/llama2-all-lowered-001/numeric-conformance.json)：全部6181个调用实际进入明确后端，旧功能辅助入口和BLAS均为0。另新增[向量周期组件及公共内存端口](mlx-vector-window-simulation.md)，当前200项回归通过。全模型证据仍是功能/数值模式；内存/控制的完整时序、统一系统地址绑定、跨算子并发和Chipyard尚未验证，因此性能门禁未打开。

公共端口现已扩展到矩阵，并加入物理地址/权限绑定和有界重试队列，[当前221项回归通过](../artifacts/tagged/model-e2e/model-io-tests-002.xml)。外部矩阵、跨kernel响应身份及重试不重复提交已验证，详见[公共内存端口记录](mlx-shared-memory-ports.md)。物理端口目前仍由测试内存驱动，不是实际Chipyard缓存；完整系统门禁保持关闭。

## 当前入口与门禁

```bash
.venv/bin/python -m scripts.capture_mlx_model_reference \
  --device cuda:1 --capture-bindings \
  --output artifacts/tagged/model-e2e/llama2-reference-new
.venv/bin/python -m scripts.run_mlx_tensor_semantics \
  --inventory artifacts/tagged/model-e2e/llama2-reference-new/inventory.json \
  --output artifacts/tagged/model-e2e/llama2-native-new
.venv/bin/python -m pytest -q \
  tests/test_model_execution_inventory.py tests/test_model_tensor_semantics.py \
  tests/test_model_matrix_program.py tests/test_matrix_window_scheduler.py
```

当前模型入口仅覆盖完整公开 Llama2 的参考和 C++ 张量语义前置实现。其他模型入口及实际 MLX tensor-to-ISA lowering 仍待补齐。`candidate_route()` 只是规划信息，不得把参考清单中的 `implemented_entry=None` 和 `missing_lowering` 改名为通过。性能门禁明确要求 `mlx_system_verified=True`、`mlx_hardware_mapping_complete=True`、端到端数值通过、全覆盖且无 tensor fallback；suite 门禁还拒绝缺少所要求模型的报告。

每次原生运行使用独占目录和独占二进制，前后核对源码、IR、权重、参考文件及 BLAS 身份。数值失败会保存逐 forward 的最大误差、RMSE、超差元素数和 token 比较，并以非零退出码结束。所有当前报告的 `inference_performance_eligible=False` 保持不变。

最新参考运行会固定 eager attention、禁用 TF32/reduced-precision reduction，并记录实际软件版本。该设置用于先定义参考精度，不代表目标模拟器已经实现 FP32 累加或归一化路径。
