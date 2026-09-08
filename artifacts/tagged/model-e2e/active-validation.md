# 模型验证与周期组件的续接记录

这不是运行状态证书；续接时必须用会话或实际进程重新检查是否仍在运行，不能仅凭此文件判定存活。

## 当前活动：完整共享物理路径运行

### 最新：两项完整运行与组件发布

- 上一进度查询属于已验证等待：实际检查确认原生PID1128327和Rocket PID1626333仍在运行，不以目录或status字段单独判断。持续跟踪原生会话 **70284** 与系统会话 **91280**，不重启。
- `llama2-rocket-full-001/llama2-resident/`已开始完整真实Rocket执行：6181源/12133任务、全部360资产、4×4/2 contexts profile、16GiB物理RAM。`memory-init.json`确认完整ELF加360资产共361个初始化段；这是驻留输入，不是CPU/DMA冷装载。
- 系统前置的 `preflight-plain` 和 `preflight-observed` 均已终态退出0；`llama2-rocket-full-001/preflight-equivalence.json`核对同seed/ELF/二进制，去掉只读observer元数据后完整设备结果一致。每例45源、21设备窗口；不视为完整模型通过。新增系统相关回归191项通过。
- 原生冻结集合继续以其execution.sources为准；系统额外冻结集合以完整用例的execution.sources/inputs/runtime_libraries和独占二进制为准。两项来源SHA在本次发布准备前均核对未变。不要在这些扫描目录下增加文件，也不要修改运行中的profile或编译输入。
- 用户新增要求：等待期间将已验收代码和产出推送远程或发布Release。正在整理组件级Git快照至 `origin` 的 `sys` 分支；权重、临时构建和两个完整运行的活动产物不纳入通过证据。发布索引为 `../publication-20260908-001/README.md`。
- 完整系统专用结果核验仍待补齐；通用runner保持full_model_execution_verified=false。完整模型终态后需要重新绑定全部路由/任务、CPU输出检查、生成反馈及显式数值参考，并保留原GPU差异。性能、作者模型/其他模型及完整RTL门槛不变。

以下镜像准备段落为较早阶段记录；其中“尚未启动”“只跟踪70284”等历史状态以本节和实际进程检查为准。

### 最新：完整系统镜像与显式资源profile

- **71365已终态退出0**，`../clocked-wide-tsi-001/report.json`通过普通TSI加载的同一高地址45节点图，内存初始化段列表为空。旧PID1545533已结束，不再轮询；其额外源码冻结已解除，此后才修改镜像/profile实现。
- 新增`system_sim/model_image/{profile.hh,profile.cc,profile_dump.cc,image.py,CMakeLists.txt}`。C++profile解析记录全部有效矩阵/向量/搬运参数、拒绝非法/不一致几何及故障注入。DPI通过MLX_CLOCKED_PROFILE读取独占profile，运行报告必须与准备时有效值完全相同；无配置仍保留demo-1pe。
- runner新增--phase prepare、--preload-assets、--device-profile和显式系统/宿主上限。驻留资产清单只来自program.assets；CPU资产复制数为0，完整任务/控制/输出检查仍保留，不冒充冷启动CPU/DMA时间。
- `llama2-system-image-001/report.json`已完成完整镜像准备：6181源、12133任务、33286912-byte命令、360资产/13476831558 bytes；ELF常规区段34362000 bytes，未嵌入巨型权重。数据图基址0x88000000，前128MiB留给程序/栈。profile为4×4、contexts=2、窗口10^12、系统上限10^13。**没有启动该完整CPU系统镜像推理。**
- `llama2-system-image-audit-002/report.json`为最新完整审计：重编译源/生命期后命令和地址绑定相同；数值参考仅允许15 arange+3 cast_device放置归一化，生成反馈链一致；profile与原生调度参数一致。完整ELF和全部360资产共同加载到同一C++ RAM实例、目标摘要匹配且不重叠，CPU/模型执行与AXI/时钟计数为0。001是增加完整编译重放/profile核对前的历史审计。
- `../resident-profile-chipyard-001/report.json`已通过真实Rocket正常/扰动45节点图：4×4有效profile与驻留资产模式实测一致，各21设备任务、全部输出字节/完成标记通过。它是模式集成检查，不是完整Llama2系统执行。
- `../system-profile-001/report.json`绑定14项profile/镜像检查和8个C++参数场景的ASan/UBSan/LSan对照；`../resident-system-image-tests-001.xml`184项回归通过。
- 7678/69885/43657/71700/96482/10398/49806/12408/40927均已终态。现在只继续跟踪完整原生周期运行 **70284**，其他系统构建/图运行无待轮询会话。
- 下一步是完整系统镜像的实际运行与模型级验收，不能将prepare/初始化报告或小图通过变成全模型通过；实际系统时间、资源和初始化范围必须保持独立标注。原GPU门槛失败、作者模型身份和其他模型/输入范围仍未关闭，性能/RTL继续后置。

### 最新：16GiB真实系统与受检初始化

- 新增`system_sim/wide_memory/`和`system_sim/chipyard/MLXWideMemory.scala`，独立`MLXClockedLargeRocketConfig`。SV/DPI地址保留64-bit，显式检查物理范围并减去RAM基址；底层仍是原mm_magic，不是已校准DRAM时序，不修改PE/RF/SPM。旧SimDRAM的32-bit DPI/取模不能作为大地址正确性证明，故未直接扩大旧配置。
- `../wide-memory-002/report.json`通过11项pytest、10个C++作业及ASan/UBSan/LSan；真实SV/DPI比较相差4GiB的不同值、容量末端、窄写入，证明封装不截断地址。`../wide-memory-tests-001.xml`共170项相关回归通过。
- `../clocked-wide-chipyard-002/report.json`实际通过正常/扰动两张45节点图：设备树RAM为base=0x80000000、size=0x400000000（16GiB），图窗口从0x181000000起1MiB。正常例内存端确实收到最高读地址6459294208，21设备任务/所有CPU结果检查通过。ELF预装载单列为host initialization，不算CPU/DMA时间。
- `../clocked-wide-chipyard-001`（75220）已终态退出1，原因是未用permissive转发+loadmem选项，CPU没有执行。随后修正runner并用25503完成002；不覆盖旧失败。Verilator单独测试还修正了64-bit参数字面量及缺失--vpi的构建调用，源码逻辑不以构建失败标为通过。
- `llama2-wide-assets-001/report.json`完整360资产/13476831558 bytes在新C++内存类中初始化、源/目标摘要全部匹配，计划覆盖6181源/33286912-byte命令。图基址0x88000000，为程序保留128MiB；没有CPU或模型执行、时钟/AXI计数为0，不能与小图结果拼成完整系统模型证书。
- 96081/14671/70527/31366/13209/1883/75220/72442/25503/24616/44339/60936/28264均已终态。
- 当前额外活动为 **71365**：`../clocked-wide-tsi-001/`，正常45节点高地址图使用常规TSI装载（不加--preload-elf），复用已验证Large二进制；模拟进程PID **1545533**最近确认存活。不能因观察超时重启，结束后检查实际ELF标记、memory.json以及最终report.json。此运行期间保持run_mlx_clocked_chipyard.source_identity里的系统/主机/编译来源不变。
- 当前仍只另跟踪完整计算 **70284**；最近PID1128327存活，prefill layer_idx=15，算子1096执行中。其冻结来源SHA校验没有变化，没有启动重复完整推理。
- 下一步是完整模型系统镜像组合、实际完整后端资源/周期上限profile及全图执行；目前RoCC示范DPI仍固定单PE和组件周期上限，不能直接把全模型计划当作完整系统运行。新内存容量/输入初始化通过不放行模型性能或RTL。

### 完整资产实际CPU装载已结束（不是推理）

- 会话 **19731已终结，退出码0**，runner1380484与Spike1381174均已结束，不再轮询。`llama2-cpu-assets-001/report.json` 已通过：360资产共13476831558 bytes，其中291个checkpoint资产共13476831232 bytes；实际CPU源读1684603950次，逐资产顺序字节数均完整、目标CPU写字节数相同、360个SHA均匹配，失败/遗漏为0。ELF的text/data/bss合计18012 bytes。计算launch/设备读写/源算子执行数均为0，不作为推理通过。
- 启动期间先核对输入文件身份并生成绑定；`execution.json` 在实际Spike启动前写入，之后记录其PID和终态。不能把暂时没有execution.json、观察超时或目录存在当成进程结束；先检查runner/实际Spike存活。
- 资产从C++只读文件区经实际CPU load/store进入设备稀疏内存，第一次计算前或load-only终结时诊断读回SHA；CPU读每资产恰好一次，未初始化/摘要错误拒绝。load-only执行源节点和设备计算窗口均应为0，不能计为完整推理通过。
- 装载终态后其额外冻结已解除；此后才在graph_runtime.c增加RoCC传输分支。旧装载证据保持其当时的来源和独占二进制，不重新标为新分支通过。**完整计算70284的冻结集合继续有效**，不得修改其中后端、编译器或runner。
- 终态后检查execution.json/spike.log、device.json、asset-source.json及report.json。必须所有360资产CPU读计数、顺序字节数、目标摘要及源/输入/二进制身份通过，才能声明装载通过；不将其改称全模型计算或Chipyard外存时序通过。

### 冻结期间的实际主机控制增量

- 最新实际RoCC连接：`system_sim/clocked_rocc/`、`system_sim/chipyard/MLXClockedRoCC.scala`、`scripts/run_mlx_clocked_chipyard.py`。纯仿真信号桥连接外部边沿控制器，无PE/FU功能RTL。CPU图运行时通过编译分支切换RoCC提交/状态，数据复制与任务完成算法共用。
- `../clocked-rocc-001/report.json` 4项直接协议检查和ASan/UBSan/LSan重放通过；`../clocked-rocc-host-tests-001.xml` 21项相关回归通过。实际接口使用核心tag位，不占DCache仲裁端口位；SimpleHellaCacheIF负责重放，CPP不重复NACK处理。内部source_id暂是提交序号，不能当模型源operator_id。
- `../clocked-chipyard-001`（26892）终态构建失败：Verilator末尾GNU++14覆盖C++17。修正后的`clocked-chipyard-002`（93262、normal PID1431229）也已终态，退出13：21任务/23361请求响应完成，但输出字节错。不得再轮询这些已结束会话或将其标为通过。
- 原因是RoCC桥未按StoreGen展开窄写数据。DCache直接使用已经展开的s1_data，之前的测试端点错误地按逻辑低位值写入，掩盖了该问题。改成真实字节通道选择后旧代码三个数值用例全部失败（`../clocked-rocc-store-regression-001/`，76572已终态）；已仅修正adapter.cc中的总线展开，并补byte高位用例。`../clocked-rocc-002/report.json` 5项检查及ASan/UBSan/LSan通过，`../clocked-rocc-tests-001.xml` 159项回归通过；99360/32556均已结束。
- 实际新系统复验 **88422已终态退出0**，`../clocked-chipyard-003/report.json` 已通过：normal与perturbed各完成45源标记、全部logits/token字节，21设备任务及23361请求/响应排空。PID1458333已经结束，不再轮询。
- **79933也已终态退出0**，`../clocked-chipyard-wait-001/report.json` 的三步embedding→matrix→softmax→RV64 argmax阻塞WAIT链全部通过：FP16和FP32普通token为1/2/3，FP16扰动为2/3/0，每例9任务排空。旧PID1461717及后续FP32进程已结束；这些是实际系统登记链，不是完整模型执行。
- 当前实际新系统设备树外存为256MiB，图缓冲绑定0x81000000、1MiB窗口；不能移用Spike 16GiB文件输入区/完整装载证据。本构建/执行期间保持其inputs/source_identity中的新桥、主机、编译来源不变；仍须等待实际进程/终态，不能凭目录或构建成功宣告图通过。
- 45107/16175/96032/26892/86086/93262/76572/99360/32556/88422/79933均已终态；19731也已完成。此刻只持续跟踪 **70284（完整计算）**。实际系统图及WAIT报告的sources已复核匹配，完整计算冻结来源未变化。最新边界见docs/mlx-clocked-rocc-integration.md。
- 下一步完整模型级系统工作是实际外存容量与完整资产流入、完整图的系统运行及跨算子资源/时序约束；当前真实系统只有256MiB外存，不能移用Spike 16GiB地址空间或把已通过的小图放大解释成模型通过。保留所有早期失败，尤其不能再用“任务/请求均完成”替代输出字节核对。

- 最新独立系统控制器：`system_sim/clocked_device/device.hh/.cc` 与独立CMake构建，完全位于两个活动attempt的冻结集合之外。每外部边沿tick一次，状态/请求观察纯读取；描述符与矩阵/向量/搬运数据均经同一注册总线队列响应取得。完成要求输出写响应覆盖且队列排空，内部64-bit请求ID跨launch/reset不回卷。
- `../clocked-device-002/report.json` 绑定15项pytest、16个C++驱动作业及ASan/UBSan/LSan重放；普通/插桩输出、事件和错误结果一致。覆盖权重扰动、无/多次状态查询等价、反压/NACK、描述符错误、写响应错误、周期上限排空、失败恢复及reset身份隔离。`../clocked-device-tests-001.xml` 154项相关回归通过。
- 开发001因混合auto类型声明构建失败，已修正；正式clocked-device-001曾漏计周期上限检测边沿，002增加该边沿及阶段总数检查。成功链8288外部边沿、2696事务（描述符2656+张量40），这些不是CPU/模型性能。测试数据为端点预初始化，不拼接为实际CPU加载/Chipyard系统证书。
- 70072/19288/24080/65948/77302/10947均已终态。当前仍需跟踪70284与19731；两个冻结源码集合及新组件正式来源校验均未变化。
- 最近确认：完整计算PID1128327存活（约4小时45分，prefill layer_idx=10、算子764执行中）；完整装载Spike PID1381174存活（约20分），execution.status仍为running。没有新的完整计算/装载终态证书，不重启或放宽数值门槛。
- 下一步实际RoCC连接已确认本地边界：LazyRoCC自动插入SimpleHellaCacheIF，该层负责s1 store数据寄存、s2_nack重放及异常assert。requestor侧新桥不能再次重发同一个cache NACK；须验证有限tag到内部ID的绑定、命令返回反压、byte/half/word访存与真实Rocket ELF。尚未修改/完成这条实际系统桥，不视为ME3通过。

- 最新文件资产路径：asset_source.hh/.cc、spike_asset_source.cc、physical_host/asset_source.py，64KiB宿主读缓存、逐区域权限/偏移检查。默认embedded入口保持，files入口避免复制完整权重进ELF。runner独占插件与execution状态记录已加入。
- `../file-asset-source-001/report.json`：14项新增检查，非对齐/超过4GiB文件偏移、只读/间隙/截断/映射错误拒绝，普通与权重扰动45节点图经实际CPU装载后与embedded方式全部设备事件/计数一致（仅新增asset_load_audit）。错误摘要和未写目标均在计算访存前拒绝；ASan/UBSan图/装载/原生探针对照通过。`../file-asset-source-tests-001.xml` 139项回归通过。
- 开发中的64KiB摘要局部缓冲曾耗尽Spike协程栈导致普通构建退出损坏，已改成有界堆缓冲并重新验证。file-asset-graph-dev-001和file-asset-source-dev-001保留失败；dev-002/003的14项检查通过，正式证据为file-asset-source-001。
- 94312/50913/40644/15272/19804/50017/40670/29610均已终结。当前需跟踪70284（完整计算）与19731（完整装载），不轮询其他旧会话。

- 本轮收尾时两个execution.sources均通过SHA核对；完整计算PID1128327仍存活、prefill layer_idx=9的算子700执行中，尚无最终结果。完整装载也未终结，尚无report.json通过证据。保持两个运行及来源冻结，不将宿主RSS或进程运行时间换算为装载百分比/MLX性能。

- 最新稀疏存储：physical_device/mapped_memory.hh/.cc替换插件dense数组；4KiB按需页、逐字节初始化位图，只有设备写过的页分配写入代数。CPU装载不会满足新launch的写满要求，读取完整验证后才复制。
- `../sparse-mapped-memory-003/report.json` 为正式证据：8项新增检查、3000步C++随机dense字节对照、同一45节点图在16GiB窗口低/跨4GiB偏移实际执行；两个完整device.json相同。旧29个链用例除新增宿主统计外完全一致，ASan/UBSan图/链与普通结果一致，独立契约额外开启LSan。
- 插件原1MiB上限已移除；仍是40-bit软件接口范围内的映射测试内存，不是实际SoC容量。runner新增--data-offset，命令+资产超过48MiB会在加载文件前拒绝，链接后检查64MiB CPU RAM/栈边界。完整权重资产流入仍是后续独立任务，不能把大逻辑窗口或完整任务计划说成模型已执行。
- 70981/18264/13614/38212/76198/52481均已结束；001因pytest父目录未建立出错，002开发检查8项通过，003正式验证通过。`../sparse-mapped-memory-tests-001.xml` 125项相关回归通过；`../graph-plan-sparse-001/report.json` 重编译完整6181节点计划，plan.json及command_blob.bin与graph-dispatch-001逐字节相同，仍未执行数据。本轮完整模型仍只跟踪70284，最近PID1128327存活（约4小时，prefill layer_idx=9、算子647执行中），冻结来源校验无变化。没有重启全模型或修改其后端。

- 最新通用主机编排：physical_host/graph_runtime.h/.c、graph_lowering.py及scripts/run_mlx_spike_graph.py。CPU按编译任务表混合执行主机控制、设备wire和已验证视图消除；资产经真实CPU load/store复制，每个source在所有batch/任务完成后才更新完成表。
- `../graph-dispatch-001/report.json`：正常与权重扰动两张45源节点图实际执行，15主机调用、21设备任务、9视图；全部完成标记与最终输出字节在ELF中核对。完整Llama2仅生成6181源/12133任务、33286912-byte命令计划，所需映射13481947840 bytes；当时插件限制1MiB，后续稀疏表示见上文，完整资产装载仍未完成，不能把计划当数据执行。
- `../graph-runtime-001/report.json` 实际RV64通过15项调度器契约检查。`../graph-dispatch-tests-001.xml` 117项相关回归通过；`../spike-graph-asan-001/{normal,perturbed}/device.json` 与release逐字节一致。参考只在检查器使用，不送入任务输入。
- 50570/85888/77031/14523/64935/49231/43401/79798均已结束；完整运行仍只跟踪70284，冻结来源没有变化。本轮新增均在冻结集合之外。下一步实际大内存/资产流入、CPU/设备时钟及Rocket/HellaCache连接，不启动重复长时运行或修改其后端。

- 最新内存wire：physical_device/memory_wire.h/.hh/.cc、memory_lowering.py、memory_wire_dump.cc。固定15872-byte控制描述符，保留存储root与重复SSA索引、操作参数、view/transfer区分及4条搬运模板；不扩大PE或搬运暂存资源。
- `../memory-wire-lowering-002/report.json` 覆盖2659源调用（1739视图+920transfer），全部Cpp解码/构造与编译计划一致。`../memory-wire-asan-001/decoded.json` 与release一致。
- 四类新报告矩阵003/向量003/内存002/host-lowering-002已由 `../joint-wire-lowering-001/report.json` 合并：6181源调用各一条路由；验证12133描述符，视图折叠后10394个执行描述符。该清单不是模型数据执行证书。
- 插件增加memory_model后端；`../spike-memory-chain-003/report.json` 29个实际ELF含CPU token→embedding→matrix→softmax→CPU argmax，以及CPU le→where和共享root切片→cat。FP16/F32全部logits和token匹配；`../spike-memory-chain-asan-001` 所有29个device.json与release一致。
- `../memory-wire-tests-002.xml` 109项回归通过。1020/51076/53573/14012/85120/78980/44408/78637/13376/33052/78700/55895均已结束。完整运行70284仍存活，最新日志在prefill第6层；本轮未修改冻结来源。下一步完整主机/设备编排和真实系统接口，不提升编译/组合图结论。

- 最新向量wire：新增physical_device/vector_wire.h/.hh/.cc、vector_lowering.py、vector_wire_dump.cc；固定4288-byte描述符保留实际ROM、阶段索引、常量与精度控制。phase表是有界控制描述符，不当作额外免费PE ROM；硬件控制存储/状态机成本仍待设计。
- `../vector-wire-lowering-002/report.json`：完整2622个向量调用、11243136 bytes，C++解码/后端构造和编译器逐字段一致，没有执行全模型数据。`../vector-wire-asan-001/decoded.json` 全部解码结果与release一致。
- 插件增加向量后端，与矩阵共用同一映射内存/请求身份/完成检查。`../spike-vector-chain-002/report.json` 的17个实际ELF包含矩阵→向量mul或softmax→RV64 argmax→下一输入，FP16/FP32普通/扰动输出匹配。softmax原子libm共享已明确记录。
- `../spike-vector-chain-asan-001` 同17例设备JSON与release全部逐字节相同；LSan仍关闭，仅声明ASan/UBSan检查。`../vector-wire-tests-001.xml` 共77项相关回归（45项新增向量wire）通过。
- 3382/84259/6524/28920/54944/58801/11685/12568/44751/58273均已结束。当前持续跟踪70284；本轮仍未修改其冻结来源。下一步补内存/搬运wire及完整主机/设备编排，不把插件组合图升格为模型/SoC通过。

- 最新矩阵wire lowering：`system_sim/physical_device/lowering.py` 从实际linear/matmul节点推导M/N/K、bias、转置与batch索引；共用physical_host.tensor_descriptor。新增address_plan.py重放已检查的绑定和SSA生命期，wire_dump.cc负责真实C++解码/后端构造检查。
- `../matrix-wire-lowering-002/report.json`：完整6181节点中的870个矩阵调用展开6822窗口，7422336 bytes，所有可执行字段/布局/地址与C++解码逐字段相同。没有执行这些全模型窗口数据；不能提升为端到端通过。首次001因文件列表扩展名判断错误失败，已修正并用002重新验证。
- `../spike-matrix-chain-006/report.json`：9个实际ELF链已改用编译器生成并链接的matrix_wire.bin，不再手写矩阵描述符。`../spike-matrix-chain-asan-002` 所有device.json与release一致。`../matrix-wire-asan-001/decoded.json` 全6822窗口解码也与release一致。
- `../matrix-wire-tests-002.xml` 共32项相关回归通过。2827/88195/52396/57686/65867/72654/65450/55227/23999/87506均已结束；当前仍只持续跟踪70284。

- 新增 `system_sim/physical_device/`（matrix_wire.h/.hh/.cc、spike_matrix.cc、CMakeLists）和 `scripts/verify_mlx_spike_matrix_chain.py`，均在70284冻结来源之外。为C++解码共享Tensor结构，physical_host/control_runtime.h增加了C/C++兼容声明，ABI字节布局与运行时算法不变。
- `../spike-matrix-chain-005/report.json`：9个实际RV64 ELF用例让CPU写payload/1088-byte矩阵描述符、等待C++矩阵后端、运行真实argmax，并用真实token选择下一次输入。FP16/FP32普通链1/2/3、扰动链2/3/0，logits位模式匹配；非法头部/机器字、未初始化读取、恢复均检查。
- `../spike-matrix-chain-asan-001/report.json` 同9例通过ASan/UBSan，device.json与release全部逐字节相同。第三方Spike未整体插桩，LeakSanitizer显式关闭，不能称全Spike无泄漏。
- 该桥是插件拥有的64-KiB映射测试内存（基址4GiB），一个status load驱动一个后端tick；不是CPU周期耦合、真实DRAM/HellaCache或完整模型运行。当前只接矩阵，通用编译wire编码、向量/搬运、完整联合编排仍待完成。
- 构建/检查会话52759/40911/18536/3680/20202/29529/31539均已终态；新测试94227只用于本桥和host代码回归。70284仍是唯一需持续跟踪的完整模型运行。

- 新增 `system_sim/physical_host/{control_runtime.h,control_runtime.c,start.S,link.ld,lowering.py}`，全部在70284的冻结来源之外；没有修改正在运行的模型后端或编译器。C函数在真实RV64 ELF执行控制算术/比较/argmax、地址计算、load/store与循环，仍不是当前模型runner已使用真实CPU。
- `../physical-host-003/report.json`：40个控制/拒绝用例+65536种FP16转换模式通过实际Spike；数据在4GiB以上，检查输入/输出guard及fflags，22个Python序列化命令与链接后C ABI逐字节一致。`../physical-host-tests-002.xml` 9项回归通过。早期physical-host-001/002保留历史。
- `../host-lowering-001/report.json`：完整6181节点布局/生命期核对后，30个控制调用全部生成640-byte主机命令，共19200 bytes。绑定地址来自已核对的lifetime replay；此文件不证明实际模型数据已在该CPU路径运行。
- 当前主机ABI操作号只选择C例程，不是PE opcode或旧native-v2 RoCC命令。下一步在冻结集合之外推进主机/设备联合ABI与ELF编排，之后再做真实Rocket/HellaCache连接；不改正在运行的C++程序去接新主机路径。
- 本轮新测试/验证会话34451、88801已结束；持续只跟踪70284。新增文档为docs/mlx-rv64-host-control.md。

- 已启动 `llama2-physical-full-001/`，exec会话 **70284**；最近工具确认runner PID1127933、native PID1128327存活。续接时先用该handle及实际进程确认，不能因观察等待超时重启。`scripts.run_mlx_physical_model` 负责独占二进制、源码文件快照、运行库与完整资产身份、进程终态及执行/数值证据。
- 模型仍为完整公开dense Llama2-7B，全部6181节点、真实权重、8-token prefill+2 decode，不是作者hybrid。四类后端均scheduled，使用共享Arena/物理背板；仍预装载输入、跨算子串行，不是实际Rocket/HellaCache或推理性能。
- 配置在 `../physical-full-config-001/`：每窗口max_cycles=10^12、共享10^13，latency/资源不变；仅观察source_operator_id=50的输出摘要。宿主watchdog172800秒。完整周期执行预计显著长于此前功能微程序运行，不能缩小模型或用外推代替它。
- **运行期间冻结runner.source_identity涵盖的文件**：tensor/tagged/matrix/vector/memory/control及其schedule、model_io、model_system、model_storage、编译器与model_physical_evidence。源快照在run/sources，二进制在run/mlx-physical-model。不要继续修改这些源码或启动重复全模型尝试；可做只读检查、文档及冻结集合之外的独立系统host/ABI工作。
- 终态后先读execution.json/native.log，区分C++执行失败、宿主watchdog和原GPU比较失败。只有native/result.json存在且完整执行/资产检查通过，才有comparison.json。再运行 `scripts.verify_mlx_model_numeric`，传本run、source=llama2-reference-005/inventory.json、numeric-reference=llama2-reference-numeric-001/inventory.json，输出新的numeric-conformance.json。验证器已支持真实physical class、窗口/内存/时钟/资源与运行库绑定，不把新物理报告伪装为旧功能报告。

### 本轮已完成证据

- 矩阵宿主控制缓存：`../matrix-control-cache-001/report.json`。54项相关回归；cached与逐周期scan的所有模拟事件、输出和目标计数相同。样例355272周期不变，控制重算/资源检查从355272次到72次；宿主计时是实现成本诊断，不是MLX推理性能。
- `llama2-physical-cost-001`（会话76940）已终结，10M矩阵窗口上限在算子50触发，宿主16.70秒，未完成模型；没有比较证书。
- `llama2-physical-cost-002`（会话84429）已终结。算子50完整Q矩阵完成：target周期从5715190到761149263，宿主约131.35秒；摘要观察额外327680周期。Q的32768个FP16值/65536 bytes与独立登记参考SHA256相同，正式 `observation-conformance.json` 已通过。随后在算子53触发10亿共享周期上限，总宿主188.56秒；**不是完整模型通过**。
- 按trace-patch-target-discovery选择算子完成边界，有界流式摘要观察只记录ID/shape/字节/SHA，不复制大张量，不向模拟器提供参考内容。详细计划为docs/mlx-physical-observation-plan.md；有/无观察数值与原算术/访存量一致，额外读回单列。
- `../physical-model-003/report.json` 绑定新观察/执行检查器用例，48项相关回归、20个成功图运行。广泛555项回归在physical-runner-tests-001.xml。`../physical-observer-asan-001/observed/` 与release在仅归一化输出文件路径后完整JSON和摘要一致。
- 4071/54453/38669/41688/68649/71473/46266/7244/22323等诊断与测试会话均已终结，不再轮询。当前只跟踪70284。

以下为历史阶段记录；其“没有活动运行”等状态只对当时归档有效，以本节最新记录和实际进程为准。

## 最新结论：搬运周期后端完整重跑已结束

### 最新集成：控制周期入口与共享原生物理执行

- `--control-backend scheduled` / `--control-schedule-options` 已接入 `compile_inventory`、主C++ Kernels/main、控制窗口统计和数值验证器；runner的source_identity已包含control_schedule。缺失控制计划拒绝，不再回退到功能控制入口。`../control-window-002/report.json`：267项相关回归、191个实际Spike预期、45节点三步生成图同时使用四类scheduled后端。
- 新 `simulator_ext/model_system/` 将Arena虚拟Tensor/pin、四类后端、AddressSpacePort、全局RequestTokens与注册请求队列连接到同一物理背板和时钟。数值只在端点的真实读/写提交中访问；权重/初始资产为明确的预装载，不是CPU/DMA装载。主入口 `build/mlx-physical-model/mlx-physical-model program.json system-options.json output-directory`。
- pin持有到窗口和请求排空，结果经公共端口读回，缓冲过期后回收。新输出逐字节有效位由真实store设置，禁止读未初始化字节或未写满就完成；物理写入查询Arena当前只读权限，seal后的旧缓存权限不能放行写操作。
- 正式证据 `../physical-model-002/report.json`：33项相关测试、8个成功图执行，涵盖三步生成、权重扰动、mmap offset、批广播/非连续矩阵、bias、重试/回压、容量/回退拒绝。普通带重试图363请求、72nack，234读+129写；权重扰动后的完整logits与重新计算参考一致，token变为0/0/0。
- 该报告还绑定完整Llama2编译结果870/2622/2659/30，各节点/资产/输出依赖与旧完整程序相同；**没有在新共享物理入口执行完整Llama2**，不能把编译核对移称端到端通过。默认每窗口周期上限是组件级配置，正式大模型运行需显式设置并评估C++执行成本；不得缩小工作量或按FLOPs外推。
- 广泛回归524项：`physical-integrated-tests-002.xml`。`../physical-model-asan-001/{generation,perturbed,batched}/` 与release数值逐位一致；只归一化输出文件路径后，完整JSON事件/计数一致。直接physical-memory-contract也通过ASan/UBSan。
- 当前无活动模型/构建/pytest会话。92153/35475/49921/38293/15286/6759/76803/49619/2991/71589/26277/93333均已终结，不再轮询。开发证据physical-model-dev-001、physical-model-001保留历史，最新绑定版本为physical-model-002。
- 下一步：为新共享物理路径建立完整模型运行/数值门禁及源码/独占二进制绑定；结合全尺寸工作量核对仿真开销和周期上限，继续实际权重装载、Rocket/HellaCache ABI与跨算子共享RF/SPM/CDC。当前跨算子串行、主机描述符/装载未计时，不能当完整推理性能；旧tagged native-v2系统证书不自动支持这些FP32/Tensor profiles。

以下为已结束的memory scheduled完整模型历史记录；不能用其源快照代表以上新集成代码。

- **当前没有活动模型运行或构建会话需要等待。** exec **89631** 已收到终态退出码1，错误来自保留的原GPU logits比较；`native/result.json` 完成6181个调用，BLAS=0、功能辅助入口=0。不得再轮询89631或将已结束的运行重启为同一attempt。
- `llama2-memory-scheduled-001/numeric-conformance.json` 已由会话 **53287** 成功生成：291个完整参数张量均消费、三个FP16 logits向量共96000值逐位一致，自生成token及cache/层数绑定核对通过。原GPU门槛仍失败，三个forward超差数1145/2524/22，token仍为393/372/338；没有放宽阈值。
- 本run实际完成2659个memory窗口（1739视图、920物化），32446193个请求/响应全部排空，读取38235409 bytes、写出37732558 bytes。matrix/vector为microcode、memory为scheduled、control为rv64_leaf；不是完整模型全周期或Chipyard系统执行，性能继续关闭。
- 冻结运行结束且数值核对完成，现在可以继续集成独立 `control_schedule` 与 `model_storage`；**历史run的源码/二进制身份保持原样**，后续源码变更不应给它重新贴新版本标签。
- 本轮新增分配器相关会话49223/72698/92839/74407/48111/49480/62576均已终结。`model-storage-001` 是直接开发重放，正式绑定源码证据是 `model-storage-002/report.json`；ASan/UBSan完整重放与release输出一致。
- 下一步：先接 `--control-backend scheduled`、主C++入口/控制窗口计数、source_identity和四类后端三步生成组合回归；再把Arena的物理region/pin接到统一模型执行与实际装载/缓存事务。不要仅因功能数值通过就跳到性能或RTL。

## 搬运重跑期间的记录（已归档）

### 等待期间的独立控制增量

- 另新增独立 `simulator_ext/model_storage/`、`tests/model_storage_contract.cc`、`tests/test_model_storage.py`、`scripts/verify_mlx_model_storage.py`，不在被冻结runner的source_identity集合内。Arena仅管理物理地址/Storage所有权/pin/只读seal，没有加载模型数据。`../model-storage-002/report.json` 为当前证据：15项测试、6181节点生命期重放、1739视图、4442新输出分配，峰值13481704384 bytes，最终全部释放。`../model-storage-asan-001/model-lifetimes.json` 与release报告逐字节一致。
- Arena尚未集成到模型runner或真实访存事务；Pin必须由后续执行/错误排空管理者持有到实际响应完成，不能遇到超时就释放。测试16-GiB地址空间不是已配置的Chipyard容量。集成时将 `model_storage` 与 `control_schedule` 一起加入源码身份；先完成冻结运行的最终数值核对。

- 新增 `simulator_ext/control_schedule/`、`tests/control_external_memory.cc`、`tests/test_control_window_scheduler.py`、`scripts/verify_mlx_control_window.py`，均在运行中的source_identity集合之外；未修改冻结来源。独立构建目录为 `build/mlx-control-window`，不影响run独占二进制。
- `control_schedule::Simulator` 使用MemoryPort响应绑定张量数据，逐条发射/完成RV64 ALU/FP片段，显式BEQ/BNE phase-PC，流式argmax与真实写确认。描述符ROM及循环/访存绑定仍不是完整RISC-V LD/ST/取指或Rocket ELF；不声称CPU时序已验证。
- `../control-window-001/report.json`：241项相关回归（221项新组件，其中191项ISA用例）、21个成功张量周期运行；同191个独立预期在实际Spike ELF上再次通过。ASan/UBSan三例位于 `../control-window-asan-001/{argmax,mul,empty}/`，输出和完整JSON与release一致。
- 尚未新增compiler/runner的control scheduled选择，也未把此目录加入主runner的source_identity；**必须等89631真实终结后**才做这些集成。下一步接同一张量入口及全lowering组合图，再推进统一物理缓冲/时钟和真实主机ABI；不要把独立组件报告改标为全模型或系统通过。
- 此轮独立build/test会话28473、29111、33766、86627均已完成，不再轮询。

### 活动模型运行

- 活动运行：`llama2-memory-scheduled-001/`，exec会话 **89631**。最近一次工具检查仍运行，PID984594已运行15分52秒、CPU99.9%，日志进入prefill第21层（operator=1428）；不能把这里的记录当成之后仍存活的证明。
- 运行命令为 `scripts.run_mlx_tensor_semantics`，输入 `llama2-reference-005/inventory.json`，matrix/vector=`microcode`、memory=`scheduled`、control=`rv64_leaf`、threads=1。完整公开dense Llama2-7B、真实权重、同一8-token prefill+2 decode；不是作者hybrid或完整系统运行。
- **运行结束前冻结其绑定的编译器、C++后端、公共端口和runner源码；不要一边运行一边修改这些文件或启动重复运行。** 独占二进制已复制到运行目录。可以只读检查，文档可继续更新。
- 组件证据已完成：`../memory-window-001/report.json`，76项相关回归、18个成功周期/端口用例；完整编译覆盖870 matrix +2622 vector +2659 memory +30 control，逐节点唯一后端，历史planned功能程序逐字段不变。新广泛回归269项在 `memory-window-tests-002.xml`。
- 新 `memory_model::Simulator` / `--memory-backend scheduled` 将索引/predicate→LOAD→CONVERT→STORE串为真实就绪/响应状态；虚拟Tensor无data/writable指针也可工作。保留独立同步planned路径用于数值回归。视图不发请求；零宽embedding仍逐个加载/校验索引；坏模板即使空输出也拒绝。
- 带重试布局链：336逻辑请求、403次下游接受、67nack，实际168读+168写。`../memory-window-asan-001/{layout,embedding,views}/` 通过ASan/UBSan，二进制输出、JSON trace与计数和release逐字节一致。旧构建/pytest会话82258/77921/37212/44870/51981/13739/88522/59273/21503/41832均已终结，不再轮询。
- 当前重跑尚无最终结果。它可能仍以退出码1结束于原GPU数值比较；先检查 `native/result.json`、`comparison.json` 和日志，区分执行错误与比较失败。原门槛不得放宽，旧数值合约结果不能移用为新运行通过。
- 运行真实结束后，用 `scripts.verify_mlx_model_numeric`，指定本run、`llama2-reference-005/inventory.json` 和 `llama2-reference-numeric-001` 的参考清单，输出本run的新 `numeric-conformance.json`。该验证器已支持scheduled memory的路由、每调用窗口、排空与写入量检查；matrix/vector仍是microcode，不是完整模型全周期。
- 后续重点：控制绑定的公共访存/真实host load-store与ABI，统一模型物理缓冲分配/装载、实际Rocket/HellaCache响应、跨算子资源与CDC；再完成要求模型/输入集合的系统正确性。模型推理性能与RTL扩展仍关闭。

## 最新公共内存/物理绑定增量

- 矩阵和向量周期模型现在共用 `model_io::MemoryPort` 与独立 `TensorMemoryPort` 行为端点；矩阵计算引擎不再直接读取/写入Tensor数值。矩阵的外部端口支持无本地data/writable指针的虚拟Tensor。
- 新的 `AddressSpacePort` 做region→64-bit物理地址绑定、读写权限、边界/溢出、自然对齐和可写别名检查；共享RequestTokens隔离不同kernel重用的本地id。`QueuedPhysicalPort`提供注册式单事务队列、nack重试、响应保持和仅空闲可reset的契约。尚不是HellaCache实际信号适配。
- 当前证据 `../model-io-002/report.json`，21项新增后总回归221项（`model-io-tests-002.xml`）。79项相关端口/矩阵/向量测试包含10个成功外部内存用例。
- 带重试矩阵例：315逻辑请求、378次下游接受、63次nack，实际提交258读+57写=315，不以幂等数据掩盖重复写。输入/输出均经外部物理测试内存，不从虚拟Tensor读取值。
- `../model-io-asan-001/` 的retry/strided/k65输出与release数据、trace、计数一致，公共端口契约也通过ASan/UBSan。该轮归档时没有活动模型运行；当前活动重跑见本文开头。
- 下一步：内存/复制计划与控制绑定层迁移到公共端口，统一模型物理缓冲分配/装载与实际Rocket/HellaCache响应；跨算子资源/CDC及完整系统周期仍未完成。模型性能与RTL扩展继续关闭。

## 最新向量周期增量

- `llama2-all-lowered-001/` 已完整结束；会话 `65497` 已收到终态，勿再轮询。全部6181个源调用实际进入明确后端，`functional_entry_calls=0`、BLAS=0。`numeric-conformance.json` 已核对权重/程序/参数、token/cache链及96000个逐位一致logits。原GPU比较仍失败，退出码1来自该比较器。
- 上述运行结束后才接入新向量周期后端，不能将历史功能结果改称周期执行结果。
- 新代码 `simulator_ext/vector_schedule/` 与公共 `simulator_ext/model_io/memory_port.h`；`--vector-backend scheduled`、`--vector-schedule-options` 已进入同一张量入口。每PE两个8-vector RF帧、每上下文5个SPM向量，全局128向量；常规向量FU与quarter-width SFU独立在途，共享RF写口。
- 外部MemoryPort支持只有shape/stride/容量、没有本地数据指针的Tensor。错token、错误/无请求响应、回压时改变/撤回响应均拒绝。当前只是公共接口和向量端口，并未把其他后端或HellaCache全部接上。
- 当前200项回归：`vector-window-tests-002.xml`。`../vector-window-002/report.json` 绑定41项向量周期测试与25项内存/组合图测试，33个成功周期/端口用例。4x4峰值25上下文占125/128向量；组合生成图同时执行matrix scheduled、vector scheduled、memory planned、control rv64_leaf。
- `../vector-window-asan-001/` 的归约65、重叠、外部虚拟内存三例通过ASan/UBSan，输出、完整trace和计数与release一致。
- 当前没有需要等待的模型或构建进程。矩阵与物理绑定的后续增量见文件开头；搬运/控制与实际系统连接仍待继续。完整模型和其他要求模型/输入范围尚不满足系统或性能验收，RTL继续后置。

## 最新内存/控制增量

- `llama2-memory-001/` 已完整结束，会话 `94756` 已收到退出码 1（原 GPU 比较失败，不是执行崩溃），勿再轮询。`numeric-conformance.json` 已验证三个 logits 共96000值在声明数值合约下逐位一致。内存计划实际调用2659次：1739视图、920物化、读38235409 bytes、写37732558 bytes。
- 控制路径已经接入统一入口：`--control-backend rv64_leaf`。当前编译分类为870 matrix +2622 vector +2659 memory +30 control，没有未分配源调用；`functional_entry_calls` 独立计数并禁止全lowering时回退。
- 完整重跑 `llama2-all-lowered-001/` 已结束，见文件开头的最新结果。其源码快照与独占二进制保留，不再有运行期源码冻结。
- 当前158项回归通过，文件 `memory-control-tests-002.xml`；包含三个生成步骤、真实cache拼接、matrix/vector/memory/control共同执行且functional_entry_calls=0的组合测试。
- `../controller-isa-003/report.json` 记录191个C++与实际Spike ELF的独立ISA对照，全部通过。Spike可执行文件为 `build/riscv-fesvr-build/spike`，源码commit `bf4b1e09ed8e7a11ecff9891b12ce5d7f3375722`。仅验证寄存器叶指令，不等于Rocket执行或系统时序。
- 空embedding的索引校验已在memory-001结束后修正，包含非法index且embedding_dim=0的拒绝测试；正常4096维模型编译产物不变。

本次完整重跑结束后，运行 `scripts.verify_mlx_model_numeric`（已支持memory/control分类）绑定结果、参数、程序和token链；保留原GPU误差记录。接下来仍需向量/内存/控制的统一时序与真实系统地址/访存、跨算子CDC/路由，以及其他要求模型。不要仅因显式编译路径全覆盖就放行推理性能或RTL。

## 最新已完成结果（向量微程序增量）

- `llama2-vector-001/` 已完整结束；会话 `52575` 已收到终态，不再轮询。6181 个实际调用全部执行，其中 870 个矩阵、2622 个浮点向量调用走微程序，BLAS=0。原 GPU 比较仍失败并保存，退出码 1 来自该比较器。
- `llama2-reference-numeric-001/` 已完整结束；会话 `98534` 已收到退出码 0，不再轮询。矩阵/归约/图控制独立，原子 libm 明确共享并记录 SHA；无插桩/插桩结果一致。
- `llama2-vector-001/numeric-conformance-002.json` 是当前完整数值合约证据：291 个参数均实际消费，三个 logits 共 96000 个 FP16 值逐位一致，自生成 token 链与 cache/层数核对通过。仅归一化了 18 个 CPU/CUDA 放置元数据；其他程序字段及 assets 相同。
- 最新组件集为 114 项，`vector-integrated-tests-005.xml`；向量 ASan/UBSan 三例在 `../vector-asan-001/`，mean/softmax 宽度8192与 FP16→FP32 softmax(17) 的数据/微指令计数一致。
- 新代码：`simulator_ext/vector_model/`、`src/mlxsim/model_vector_program.py`、`model_dtype_lowering.py`、`model_numeric_reference.py`、`scripts/verify_mlx_model_numeric.py`。`--vector-backend microcode` 已接入普通张量运行入口。
- 完整运行结束后才集成 softmax(dtype=FP16) 的输入前置转换修正，并拒绝未登记的 narrowing mean.dtype；当前 Llama2 编译程序逐字段不变。后续又加了无效 tail shuffle 拒绝。旧运行绑定其当时 C++ 源码/二进制，不重新标注为新版本。
- 仍有 2689 个源调用为 C++ 功能语义（索引、mask/select、布局、复制/转换等）；向量周期调度、跨算子层折叠、完整 Chipyard 内存链及其他要求模型仍未完成。原 GPU 门槛失败没有被抹去；数值合约通过不单独放行推理性能或 RTL。

上述向量阶段与内存/控制重跑均已结束。以下为矩阵阶段历史记录。

## 矩阵阶段历史

- C++ 全模型微程序尝试 `llama2-microcode-001/` **已结束**；会话 `6750` 已收到终态，勿再轮询旧句柄。6181 个源节点全部执行，675 linear + 195 matmul 走微程序，BLAS=0，完成 65,175,028,352 次有效 lane MUL 和同量 ADD。三个 token 一致；原 GPU logits 比较失败，退出码 1 来自比较器，不是执行崩溃。证据在 `comparison.json`。
- 显式 K 顺序数值参考 `llama2-reference-kasc-001/` **已结束**；会话 `94598` 已收到退出码 0。完整 CPU 模型的无插桩/插桩输出相同，实际覆盖 675 linear + 195 matmul；完整清单和参考输出已保存。
- 两项运行结束后，才修改了张量执行器以接入新周期组件。因此微程序 001 的源码身份是当时的版本，不可重新标注为当前版本。无需重启已完成运行；后续每次实际回归仍使用独占目录/二进制并核对源码身份。

已经完成：首层完整 Q/K/V 的正式数值报告 `llama2-microcode-001/matrix-conformance.json` 为 98304 个元素逐位相同；对独立矩阵合约参考的 `kasc-numerical-diagnosis.json` 仍有 58/257/11 个 logits 超差，原门槛未放宽。53 项组件/集成回归通过，周期组件当前证据在 `../matrix-window-005/report.json`；ASan/UBSan 的 K=65、反压和混合精度三例位于 `../matrix-window-asan-002/`，与普通构建的数据/trace/计数一致。

新增源码为 `simulator_ext/matrix_schedule/`；`--matrix-backend scheduled` 已接入同一 C++ 张量执行器和编译入口，严格禁用 BLAS。但完整模型**周期执行**尚未完成，非矩阵算子仍为功能语义服务，跨算子 CDC/路由和 Chipyard DMA 接口仍未接入。

后续（以最新向量阶段结果为准）：

1. 保留原框架 GPU logits 比较，可能仍失败；不要放宽原门槛。
2. 数值合约层的一致性已经通过；保持与原框架/硬件数学单元验证的区别，扩展输入与模型范围，不以单输入通过代替完整验收。
3. 继续用 `scripts.compare_mlx_tensor_observations` 保存原 GPU/CPU 与显式合约参考的区分；后者不能替换历史失败。
4. 验证实际 MUL 有效 lane 数与模型 MAC 总量一致、`blas_calls=0`、所有 forward/token/cache 依赖真实执行。更新模型文档，不把微程序功能通过当作多 tag/阵列/Chipyard 系统通过。
5. 继续目标侧非矩阵 lowering 和微程序状态到有界多上下文/访存系统的集成；不提前做模型推理性能或 RTL。
