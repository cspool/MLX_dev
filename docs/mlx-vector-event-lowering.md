# 完整向量窗口事件路径

在矩阵窗口之后，C++ 事件核心 schema v4 增加向量操作数准备、SPM 初始化、SFU 发射间隔和向量/SFU 写回竞争。`model_vector_events.py` 依据实际 canonical 微程序生成完整向量窗口，仍不执行 Tensor 数值，也不使用参考耗时拟合延迟。

## 已接入内容

- 逐元素 add/sub/mul/div/maximum、pow2、rsqrt、silu、cos/sin/exp、neg；保留 scalar/tensor 区别、广播、非单位 alpha 和 FP16/FP32 转换。
- 每个操作数的 prepare、下一边沿 SPM 初始化、随后 DMA/本地加载。准备不占用 PE 发射机会，初始化占 SPM 端口但不占 SPM 服务单元；不会为 scalar 常量伪造 DMA。
- mean 与 softmax 的完整分块、补齐、归约树、二进制进位栈、root/final 阶段，以及 softmax 的 max/sum/output 三遍读取。
- 输出尾部的四 lane SFU 分组和无有效 lane 的谓词发射；窄化 softmax 输入的位置按实际微程序保留。
- 计算、SFU、SPM、DMA 在途区间，计算/SFU 重叠和同 PE 多上下文重叠分别计数；服务保留时间仍与实际在途时间分开。

schema v1/v2/v3 的既有行为保持。v4 仍采用全局完成优先顺序，**尚未完整对齐多源执行器逐源 tick 的顺序**；当前编译入口为单个完整向量窗口、固定延迟内存端点、未计时模板装载，不能直接冒充整图或 Chipyard 缓存模型。

## 验证

148 项回归、142 次 ASan/UBSan/LSan 重放通过（115 正常、27 预期拒绝）。其中 48 个完整向量窗口的周期、读写字节、微指令数、算术/SFU lane 数和全部在途占用，与原生 C++ 数据执行一致。原生输出另与功能微指令执行逐位比较；独立重放再次核对原始输出及完整报告。

覆盖全部向量算子、FP16/FP32、scalar 常量、长度 1/3/17/19/65/129/4096、非默认端口周期、SFU/计算间隔、非单位 alpha、窄化 softmax 和满 4×4 阵列。原先 22 个矩阵窗口的对照仍通过。

满阵列用例中两条路径均达到 25 个驻留上下文，但该输入的计算/SFU 重叠为 0；另一同 PE 的 FP16 用例有一致的正重叠计数。前者最初被测试误认为必然重叠，随后按实际结果纠正，并增加后者作为对照；没有修改模拟器去制造重叠。驻留数量不能代替有效并发。

编译入口：

```text
python -m scripts.compile_mlx_matrix_events --family vector --job <完整向量窗口.json> --output <新目录>
```

独立重放入口为 `scripts.verify_mlx_matrix_event_alignment --family vector`。文件名保留原矩阵入口以兼容既有命令；输出明确标记 vector、单窗口和非整模范围。

## 剩余工作

独立搬运、RV64 控制、完整 batch/源图、逐实例块间通信、多源仲裁及真实模板装载尚需接入。矩阵/向量窗口对齐结果不能推广为完整 Llama2/BERT 的性能误差；六个完整运行仍单独追踪。P0 主线优先，GPU 新实验及 RTL/PPA 继续后置。
