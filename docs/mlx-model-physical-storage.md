# 模型物理缓冲的地址与生命期

本组件是公共内存端口到完整模型系统集成之间的地址所有权层。它只管理物理地址区间和Tensor的存储生命期，不加载权重、不计算算子、不实现缓存，也不测量推理性能。它在 `llama2-memory-scheduled-001` 冻结重跑期间独立开发，未加入该run的执行路径；该run现已结束，后续集成需新证据。

## C++ 所有权契约

[model_storage::Arena](../simulator_ext/model_storage/buffer_arena.h)接收明确的base、容量和对齐（默认64 bytes，最小8 bytes），在该范围内采用first-fit分配和相邻空闲区合并。分配前检查shape字节乘法、对齐、地址加法与容量；空间不足或碎片导致失败时，不修改已有分配。容量必须来自后续系统的实际内存图，不能因支持64-bit地址就假定Chipyard已经有相同DRAM容量。

返回的Tensor只有dtype/shape/stride/offset和Storage容量，data/writable均为null；它不占用一个同样大小的宿主数据数组。全局物理地址容量与PE RF/SPM容量是不同约束，这里没有扩大PE片上资源。

- Storage持有分配的共享所有权；合法视图共享原Storage，根SSA释放不会释放仍被视图使用的地址。
- `Arena::Pin`保留一个视图及分配。所有SSA引用释放后，只要pin还在，地址仍不可复用。复制pin保持同一租约，最后一个副本消失才解锁。
- 地址回收后可以再次分配，但分配ID单调更新；物理请求仍需使用公共 `RequestTokens`，不能把分配ID当请求ID，也不能靠它替代迟到响应检查。
- 权重可在装载阶段保持可写，实际写入全部完成后永久seal为只读。有pin时拒绝改变权限；只读缓冲不能取得写pin。seal本身不证明数据已经初始化，装载器必须另行证明写入范围与响应完成。
- 空Tensor占用一个最小对齐地址槽，逻辑字节仍为0，不因此产生访存。外来Storage、复制Storage对象冒充同一分配，以及越界/负步长视图均被拒绝。

pin不是DMA完成探测器。调用者必须将它保留到请求和写确认实际排空，错误恢复/超时不能提前丢弃。Arena包装对象的销毁也不使仍存活的pin失效。这一层由单一模拟器控制线程拥有，不提供并行宿主线程同步。

## 当前证据

[model-storage-002](../artifacts/tagged/model-storage-002/report.json)绑定源码、二进制和完整编译程序，包含15项分配器回归和一次全模型地址生命期重放。测试覆盖别名、pin副本、Arena包装对象生命期、只读权限、回收/复用的新身份、碎片失败/合并、64-bit溢出、空张量和500步随机分配/释放。

重放使用完整公开Llama2的6181个编译节点，而非缩小图：

| 检查项 | 结果 |
|---|---:|
| C++重新检查的无搬运视图 | 1739 |
| 创建新输出分配的节点 | 4442 |
| 初始资产逻辑字节 | 13476831558 |
| 本次16-GiB测试Arena中的分配峰值（含对齐） | 13481704384 bytes |
| 最终结果观察并释放后的保留字节 | 0 |

每个输入先取得pin，再处理输出所有权；故意在pin仍保留时执行编译器的SSA释放表，核对地址不能被回收。最后的logits/token缓冲必须仍可观察，观察结束后才释放；全部空闲区最终合并为一个区间。重放只执行视图布局检查，不执行matrix/vector/control算术或transfer数据搬运，报告明确 `model_data_loaded=False`、`model_inference_executed=False`。

ASan/UBSan完整重放通过，所有事件和计数与release逐字节相同，文件为 `artifacts/tagged/model-storage-asan-001/model-lifetimes.json`。这些字节是地址分配核对，不是实际DRAM占用测量、模型带宽或推理性能。

可重放命令（证据目录必须是新目录）：

```bash
.venv/bin/python -m pytest -q tests/test_model_storage.py
.venv/bin/python -m scripts.verify_mlx_model_storage \
  --program artifacts/tagged/model-e2e/llama2-memory-scheduled-001/program.json \
  --output artifacts/tagged/model-storage-NEW
```

## 集成的剩余工作

最新已在[共享物理执行器](mlx-shared-physical-model.md)连接四类后端：pin持有到窗口与事务排空，过期背板回收后地址可复用，物理端点检查当前只读权限和真实写入有效位。组合图已经通过，但这不是实际CPU/DMA权重装载或完整模型系统执行。

1. Arena与控制周期组件已接入共享物理入口，新的验证器绑定全部新增目录；继续新路径的完整模型执行，不复用旧运行身份。
2. 使用真实初始化/权重装载路径填充物理缓冲，核对文件偏移、长度、内容身份与实际写完成。将每个后端的region绑定到相应pin，pin跨真实请求/响应与错误排空持有。
3. 统一跨算子时钟、资源与CDC，再接实际Rocket/HellaCache。16-GiB测试地址空间尚不是已配置的Chipyard内存图，需结合真实系统容量和权重驻留/流入方案验证。
4. 同一完整模型的最终结果、生成/cache状态和所有算子实际执行均通过后，再验证系统推理性能；其他要求模型和输入集合不能由这一个地址重放替代。RTL继续后置。
