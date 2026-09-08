# C++ 宿主开销：缓存稳定的准入失败

这是一项模拟器实现优化，不是 MLX 推理性能评估，不改变周期推进、数值指令、资源容量或调度策略。四个正在运行的完整模型尝试继续使用各自冻结版本，本次只在独立 `mlx-host-opt-v1` 分支开发和验证，没有替换其二进制或修改 RTL。

## 定位与修改

[插桩记录](../artifacts/tagged/host-hotspots-baseline-001/report.json)使用完整执行的矩阵/逐点组件（8×128×256，262144 个有效 MAC），不是缩小模型后宣称完整推理通过。`gprof` 记录约 150 万次 `Resources::offer` 调用，该函数约占本次插桩 self-time 的 40%。这个数字只用于定位，插桩时间不是未插桩性能或完整模型估计。

向量消费者等待时，资源经常没有变化，原实现仍反复扫描上下文、RF、SPM 和 ROM。现在每个 host client 只记住一次失败查询的 `(allocation_version, PE)`；命中时返回同样的无资源结果。它不缓存成功 offer、不缓存寄存器/SPM 数值，也不缓存取指、FU 就绪或完成事件。

失效依据来自真实资源变更：准入分配、边沿末尾的退休释放和驻留限制修改都会改变 `allocation_version`。不同 PE 不共享失败结果；调用者、已附着状态、当前边沿与 PE 范围检查仍在缓存判断之前执行。退休在边沿中只作标记，资源到边沿结束才释放，因此同边沿仍不能准入。这些规则由[直接 C++ 契约](../tests/shared_offer_cache_contract.cc)覆盖。新增的是宿主实现元数据，不是扩大模拟硬件的 RF/SPM/上下文资源。

## 验证结果及边界

[182 项回归](../artifacts/tagged/host-offer-cache-tests-001/regression.xml)通过；[独立二进制对照](../artifacts/tagged/host-offer-cache-equivalence-001/report.json)对 10 个完整登记图重放，旧/新/ASan 构建的最终输出、事件、模拟周期和资源记录完全一致（仅规范化输出文件路径）。契约另经 ASan/UBSan/泄漏检查运行，覆盖 RF、全局 SPM、ROM、上下文容量、驻留限制、错误调用者、已分离 client、旧成功 proposal 和边沿末回收。

四轮交替的非插桩宿主测量中，组件耗时中位数约从 3.72 秒变为 3.17 秒。各次执行的 262144 MAC、1546378 个图周期和1556698个含读回总周期不变，没有跳过模拟边沿。此数据包含进程/输入输出开销、存在宿主噪声，只说明该组件上的实现开销变化；不能外推为完整模型、GPU 调度收益或硬件推理加速。

脚本为 `scripts/profile_mlx_host_hotspots.py` 和 `scripts/verify_mlx_host_offer_cache.py`。旧基线由父提交 `2e58eb0` 的 C++ 源码构建；历史插桩报告中的源摘要不被改写成优化后源码。完整模型正确性、所有模型/输入、原 GPU 数值门槛、系统推理性能和 RTL 的门槛保持不变。
