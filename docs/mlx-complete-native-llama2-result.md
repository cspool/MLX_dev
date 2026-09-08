# 完整原生 Llama2 执行：数值合约通过，原 GPU 门槛未通过

`llama2-physical-full-001` 已实际结束，不再是启动或受限窗口检查。[归档清单](../artifacts/tagged/llama2-physical-completed-001/manifest.json)绑定完整程序、原生结果、三步实际 logits、两套参考清单/输出、数值审计、日志及执行源码快照；不含模型权重、宿主大二进制或环境。

## 完整执行和工作量

运行的是公开 dense Llama2-7B，不是论文作者的 MLX hybrid：全部6,738,415,616参数、32层、hidden4096，8-token prefill及两次单token decode。全部291个参数张量被消费，没有缩层/缩宽、替换权重或注入中间参考激活。

实际执行6,181个源调用：870 matrix、2,622 vector、2,659 memory、30 control；矩阵覆盖全部batch，共6,822个窗口。执行65,175,028,352个有效lane MAC，43,460,882,101笔物理请求全部排空；4,802次地址分配与释放守恒，结束时无在途请求或存活buffer。BLAS、通用功能入口、Python/GPU计算回退均为0。详见[原生完整结果](../artifacts/tagged/llama2-physical-completed-001/native/result.json)。

这是统一物理地址空间上的**逐算子串行 C++ 执行器**。四类后端均受响应/时钟驱动，控制是RV64 leaf而非实际Rocket，资产为宿主预装载，不是CPU/DMA冷启动。它不验证后来新增的跨算子共享阵列/配对路径；后者及两个Rocket完整尝试仍需各自终态验收。

## 两个数值结论必须同时保留

[声明数值合约审计](../artifacts/tagged/llama2-physical-completed-001/numeric-conformance-001.json)通过：三步共96,000个FP16 logits与预先绑定的独立控制/矩阵/归约参考逐字节相同；实际argmax得到393/372/338，并核对自身token与KV/cache反馈依赖。原子数学函数与C++ FU共享libm，不能把这一参考描述为完全独立的原子函数实现。

[原GPU框架对照](../artifacts/tagged/llama2-physical-completed-001/comparison.json)仍失败；原条件 `abs(error) <= 0.005 + 0.005 * abs(reference)` 没有放宽：

| forward | 阶段 | 实际token / 是否一致 | 32,000 logits中超差数 | 最大绝对误差 |
| --- | --- | --- | ---: | ---: |
| 0 | prefill | 393 / 是 | 1,145 | 0.015625 |
| 1 | decode | 372 / 是 | 2,524 | 0.0234375 |
| 2 | decode | 338 / 是 | 22 | 0.015625 |

C++进程退出0、所有源完成；主runner随后因GPU数值门槛失败退出1。`execution.json` 的 completed 是C++执行终态，不代表主runner整体数值验收通过。声明合约通过不能覆盖GPU比较失败，也不能用相同token代替logits检查。

此前[精确投影诊断](mlx-exact-projection-diagnosis.md)提供了首层累加/舍入差异证据，但未逐层解释全部GPU差异。本次结果扩大的是**完整实际执行与声明合约一致性**的证据范围，并未自动解决原GPU门槛。

## 来源与使用边界

归档的每个执行源码快照都与 `sys` 的 commit `aae1524831d53314b53e855b07fce6f7ac52f145` 相同；运行前后源码、程序、独占二进制和库身份保持不变。原始完整运行目录、参考文件路径保留在报告中，未重写已绑定内容；归档manifest的 `source_path → path` 映射对应实际复制件。复现需使用这一执行版本及匹配摘要的完整权重，而不是把当前开发分支的新算子实现视为本次已执行代码。

原生计数是执行/排空审计证据，未换算为系统推理速度、TTFT、吞吐或论文加速比。`mlx_system_verified=false`、`inference_performance_eligible=false` 保持不变：原GPU门槛、完整配对/Rocket系统结果、其他所需模型与作者变体尚未通过；继续先模拟器、再系统、最后RTL。
