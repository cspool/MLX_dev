# 完整模型的原生 C++ 配对执行入口

新增 `scripts/run_mlx_ready_model.py`，驱动已有 `mlx-ready-graph` C++ 可执行文件执行完整模型。原数值图仅增加由编译器推导的有界配对计划；层数、形状、权重和全部源调用保持不变。Python 只做输入编译、文件绑定、进程管理和结果审计，不推进模拟周期，不执行模型算子，也不注入中间结果或参考 token。

## 执行与审计

四个 scheduled 后端通过同一物理地址空间和响应分发器执行。矩阵/向量共享实际 RF/SPM/ROM 和上下文，消费者可等待块完成事件后运行；模板逐字装载计入竞争。控制仍是受时序驱动的 RV64 leaf，不是实际 Rocket CPU；资产由宿主预先装入模型内存，不算 CPU/DMA 装载。这些边界与真实 Rocket 结果分开记录。

`model_ready_evidence.verify_ready_execution` 核对：

- 每个源调用和每个矩阵 batch 均有实际完成窗口；窗口连续覆盖所属源，不因多源重叠而把各窗口周期相加当作总周期。
- 普通依赖在父源完成可见后启动，配对双方同时准入；全部配对事件按各自 epoch 完成并排空。
- 实际矩阵 MAC、向量/搬运写入字节、控制指令及最终读回字节均有对应计数；无功能入口、BLAS 或 Python/GPU 数值回退。
- 共享上下文/RF/SPM/ROM 与模板装载守恒，物理请求、响应、读回、重试和内存回收闭合。

实际输出由 C++ 经受检物理端口读回后写入文件，逐字节比较指定数值参考；若提供原 GPU 清单，则另外直接比较实际 dump 与 GPU 结果。数值参考通过不能覆盖 GPU 门槛失败。进程成功但数值失败时，保存实际输出和失败比较；周期限额等非零退出不能生成通过证书。

每次运行使用独立目录、独占二进制、完整源码快照与动态库摘要，前后检查输入来源。多个参数共用的 shard 在绑定阶段只散列一次，启动前和运行后仍重新核对全部文件，避免对同一大型 shard 重复散列数百次。

## 完整公开 Llama2 的启动前条件

`--scope public-dense-llama2` 要求原始源清单和显式数值参考，核对 32 层、hidden 4096、完整 checkpoint、所有层/cache/forward 记录以及实际 token 反馈链。还直接解析 safetensors 头部，逐一核对 291 个参数的真实形状、F16 精度、文件分片、数据偏移和字节数，参数元素总数必须为 6,738,415,616；不只信任配置中的模型规模标记。源清单必须重新编译成原完整数值图，参考只能存在已声明的设备放置归一化差异。

这些是完整输入条件，不是运行通过。只有实际原生全图完成、工作量和资源审计通过、全部最终结果与指定数值合约一致后，才会记录 `full_native_model_numeric_contract_verified=true`。`actual_cpu_execution`、`mlx_system_verified` 和 `inference_performance_eligible` 仍为 false；这不等同于全部 MLX 模型或作者 hybrid 验收。

## 已有证据

[145 项回归](../artifacts/tagged/native-ready-model-001/regression.xml)通过；[15 个 ASan/UBSan/泄漏检查重放](../artifacts/tagged/native-ready-model-sanitizers-001/report.json)覆盖五个正确用例、五个数值失败用例和五个周期限额失败用例。正常、扰动三步生成、非相邻配对、矩阵完整广播 batch 及尾部形状均有实际 C++ 执行。

完整尝试 `artifacts/tagged/model-e2e/llama2-native-paired-full-001` 已启动，使用 16 GiB、4×4/2 contexts、32 个事件记录，以及与旧原生运行相同的 8 周期物理内存延迟。它不包含旧运行的 source 50 诊断读回，不能据此直接作性能对照。当前尚无完整模型终态；旧原生与两个 Rocket 完整运行也继续使用各自冻结版本。

```bash
PYTHONPATH="$PWD/src:$PWD" .venv/bin/python -m scripts.run_mlx_ready_model \
  --program /path/to/complete-program.json \
  --reference /path/to/numeric-reference-inventory.json \
  --source-inventory /path/to/source-inventory.json \
  --scope public-dense-llama2 \
  --options artifacts/tagged/native-ready-full-options-001.json \
  --event-slots 32 --timeout 172800 \
  --output artifacts/tagged/model-e2e/llama2-native-paired-full-NEW
```

新实现位于独立分支 `mlx-native-e2e-v1`，没有修改三个既有长运行的工作树。实际运行期间继续冻结该尝试的来源、输入、二进制和运行库，不因观察超时重启。
