# 首层 Q/K/V 数值差异：整数精确点积诊断

本轮继续区分数值执行路径与 lowering/数据路由错误，不修改 MLX 运算模式、原 GPU 参考或误差门槛。诊断读取已保存的中间值作为**孤立算子的输入**，不把这些值送回任何模型推理；四个完整尝试仍运行各自冻结版本。

## 独立 C++ 精确参考

`simulator_ext/exact_numeric/` 是单独构建的诊断程序，不链接进入 MLX 执行器。每个有限 FP16 数都是 `2^-24` 的整数倍，因此乘积可用 `2^-48` 为单位准确表示。工具用 signed int128 累加完整点积，最后以整数运算实现 binary16 round-to-nearest/ties-to-even，包含 subnormal、舍入到零、跨指数边界和溢出。K 被限制到 `2^20`，最大累加绝对值小于 `2^100`，不超出 int128。

另一路独立计算逐 K、分离的 FP32 MUL/ADD，供核对现有 `mlx-matrix-f32-kasc-v1` 微程序；它不是精确点积的替代品。精确点积只在末尾舍入，而 MLX 声明模式在每次 FP32 累加后舍入，两者本来就不要求逐位相同。

[验证记录](../artifacts/tagged/exact-numeric-validation-001/report.json)包含14项测试及对应安全重放：63488个有限FP16值、190464个中点/邻近整数值、4000个随机精确整数舍入，以及独立 Python 大整数点积对照和非法输入拒绝。三个完整投影也经 ASan/UBSan/泄漏检查，合计402653184个整数乘积；没有用抽样输出代替整个投影。

## 同一输入、完整真实权重

[精确投影记录 002](../artifacts/tagged/exact-projection-diagnosis-002/report.json)绑定原 GPU 观察清单、历史 BLAS/微程序报告及此前记录的输出摘要。三条路径的 `v49` 输入字节完全一致；每个 Q/K/V 使用完整 `8×4096` 输入、真实 `4096×4096` 权重，检查全部32768个输出。

下表为相对“精确点积最终舍入到FP16”的不同元素数：

| 投影 | GPU | 历史 C++ BLAS | C++ 微程序 |
| --- | ---: | ---: | ---: |
| Q | 362 | 40 | 64 |
| K | 390 | 22 | 62 |
| V | 731 | 56 | 93 |

微程序三个投影与独立逐 K FP32 实现**逐字节相同**。相对精确参考，GPU 的最大绝对差为 Q/K `0.001953125`、V `0.0001220703125`；微程序为 Q `0.001953125`、K `0.0009765625`、V `0.0001220703125`。靠近零处的 binary16 码距可能较大，不能只用码距忽略绝对误差。

这确认这三个同输入投影符合声明的微程序数值规则，支持已观察差异来自数值路径的判断；不能由此推断 GPU 内部具体归约树、证明任意输入都正确，或把后续全部模型误差一概归因于舍入。

## 精度开关与 split-K 排查

历史 GPU 清单记录 TF32、FP16/BF16 reduced-precision reduction 均关闭，不能据此重新解释为开启 TF32。另核对当前安装 PyTorch 的源代码：单个 `False` 仍允许 split-K；元组 `(False, False)` 才同时禁用两者。此行为也见[PyTorch 官方说明](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html#reduced-precision-reduction-for-fp16-and-bf16-gemms)（访问于2026-09-08），不是本项目根据输出反推的默认值。

[GPU 对照 002](../artifacts/tagged/gpu-splitk-diagnosis-002/report.json)在独立进程、同一输入/权重、确定性算法、TF32关闭并禁用全FP16累加的设置下，比较 cuBLAS、cuBLASLt 允许split-K、cuBLASLt 禁用split-K。九个输出都与历史 GPU 观察逐字节相同。因此，改变这组选项没有消除当前投影差异；不据此声称已识别内部 kernel 或证明其他形状也无影响。报告记录的是请求/报告的选库偏好，没有采集内部 kernel trace。

首次 GPU 诊断001在默认cuBLAS后端请求关闭split-K时被运行库拒绝，保留为失败配置记录；002显式区分选库及split-K设置后完成。原 GPU 文件未被替换，原始模型误差条件 `abs(error) <= 0.005 + 0.005 * abs(reference)` 没有更改。

## 未完成的部分

原 GPU 的完整 logits 门槛仍失败，后续层误差传播仍需继续定位，新的完整系统/原生尝试仍待终态。PyTorch本身不承诺不同设备/实现的浮点结果逐位一致，见[官方数值说明](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)；这不是免除本项目验收的理由。完整模型、全部模型/输入、性能和RTL均不得由本诊断放行。
