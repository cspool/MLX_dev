# 完整原生 Llama2 终态归档

结论：**完整 C++ 执行和声明数值合约通过；原 GPU logits 门槛失败，系统与性能未通过。** 不是“全部模型推理验收通过”的Release。

[结果说明](../../../docs/mlx-complete-native-llama2-result.md)包含模型身份、完整工作量、两套数值结论和使用边界。

- [实际结果](native/result.json)：全部6,181个源调用、291个参数张量，真实物理请求/资源排空，无BLAS或通用功能回退。
- [数值合约](numeric-conformance-001.json)：96,000个FP16 logits逐字节一致，实际token393/372/338及反馈链通过；原子libm共享已披露。
- [GPU对照](comparison.json)：三步超差1,145/2,524/22个元素，原阈值不变；此失败未被重标。
- [manifest](manifest.json)：86个复制文件、63,651,739 bytes，不含本索引和manifest自身。包括完整程序、两套参考清单/输出和执行源码，不包含权重或宿主二进制。

执行版本为 `sys@aae1524831d53314b53e855b07fce6f7ac52f145`，每个源码快照均与该commit逐文件核对；它不等于当前开发分支的实现版本。原报告中的绝对路径未改写，复制件位置由manifest中的 `source_path` 和 `path` 对应；应使用匹配的外部权重和原版本复现。
