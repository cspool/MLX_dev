# 完整模型的系统执行验收器

2026-09-08。新增 [任务执行核验](../src/mlxsim/model_system_evidence.py)、[系统模型验收入口](../scripts/verify_mlx_system_model.py)及[拒绝测试](../tests/test_system_model_evidence.py)。Python只读核验结果、重放编译及重建主机ELF，不执行模型计算，也不推进模拟时钟。

这是[模型级验收方案](mlx-model-e2e-validation.md)的系统证据补充，不改变“C/C++模拟器 → 系统集成 → 正确性后验证性能 → 最后RTL”的顺序。两个完整模型进程继续使用原有冻结来源和独占二进制。

## 需要同时满足的证据

1. 实际进程已经退出，退出码严格为整数0，通用runner完成终态检查，真实CPU只打印一次输出检查完成标记。`running`、watchdog、退出13、单个设备窗口的`done`、进度日志中的完成字段都不能代替CPU终态。
2. 原始编译输入、完整源码集合与attempt源码快照、主机ELF、模拟器、动态库、资源profile和初始化文件摘要一致。审计使用的program、lifetimes和reference必须是运行前已经绑定的输入。
3. 从完整program重新生成每个source的任务序列；逐一核对source ID、ordinal、batch index/count、后端类型、描述符长度与偏移。控制在实际RV64执行，合法视图经lowering验证后消除，其余进入对应C++设备后端。矩阵只完成第一个batch不能算整个源算子完成。
4. 逐设备窗口核对实际取回的描述符、完整输出写入字节、矩阵有效MAC数、上下文准入/退休、RF/SPM/ROM容量及所有待完成状态。transport累计请求增量必须恰好等于描述符的8-byte取回事务加该窗口的数据事务；cache重放仍由SimpleHellaCacheIF负责，不在C++重复计算为新逻辑请求。
5. 设备窗口起止边沿与fetch/run/drain计数一致，前端/设备/AXI在终态排空，实际RAM与设备时钟对应。这些计数用于守恒核对，不因记录它们而开放推理性能门槛；magic memory仍不是经过校准的DRAM模型。
6. 重新编译完整命令和CPU检查程序，要求命令、源码、launch map及最终ELF逐字节等于原运行。核对实际初始化的每个ELF `PT_LOAD`段、全部输入资产及不重叠的物理区间。只允许重放时新的literal容器路径不同；模型初始资产不能变成参考中间激活或golden输出。

CPU检查器本来就核对全部源完成表项、控制/设备/视图计数，以及每个forward的最终logits和token字节。独立ELF重建将实际成功终态与这套完整检查逻辑绑定，而不是只信任一个PASS字符串。

## 完整公开Llama2的额外条件

默认scope为`public-dense-llama2`：核对32层、hidden 4096、完整291个参数张量与6,738,415,616个参数的身份和使用；重编译完整源清单，与实际程序逐字段比较。数值参考仅允许已经声明的CPU/CUDA放置元数据归一化。初始token来自输入，后续embedding必须沿合法的恒等变换读取上一forward实际argmax结果，不能逐步注入参考token；每层cache和全部forward边界也要对应。

系统profile必须等于原生scheduled输入的有效矩阵/向量/搬运参数。显式记录矩阵/归约参考的独立控制与共享原子libm，不能由共享库一致性推导硬件超越函数已经实现或验证。

当前CPU检查器做的是与指定数值参考的逐字节比较，并未向宿主导出新的全量目标logits。因此验收报告明确标记`actual_output_dump_emitted=false`。只有CPU字节检查与ELF重建通过后，才能根据“目标输出等于该数值参考”传递性核对其与原GPU输出的差异；这不是一次新的目标输出dump。原来的GPU误差界保持不变，比较失败仍报告失败，不能由数值契约一致性覆盖。

即使某个完整公开Llama2用例通过，报告也不把它提升为全部MLX模型、全部输入、跨算子共享资源、RTL或推理性能通过。`registered-graph`只用于已登记小图的核验器验证，其完整模型字段始终为false。

## 已完成验证

- [真实Rocket预检的独立核验](../artifacts/tagged/system-model-audit-preflight-002/report.json)：45源、21设备窗口、23361个请求/响应。全部命令与实际CPU ELF重建相同，4个初始资产/72 bytes与实际RAM记录对应，三个forward的全部logits/token字节检查绑定成功；不是完整Llama2执行。
- [229项相关回归](../artifacts/tagged/system-model-audit-tests-003.xml)全部通过，包含任务遗漏/重复/错误batch、MAC不足、输出store不全、错误完成/计数、资源超限、未排空访存、错误ELF与额外golden初始化的拒绝，以及旧编译/接口门禁。
- [对运行中完整尝试的拒绝记录](../artifacts/tagged/system-model-audit-active-rejection-002/failure.json)：实际完整Rocket进程仍在运行时，核验器拒绝生成通过报告，且没有启动新的模拟器或修改原尝试。
- 开发回归001曾把合法的ELF零尾部扩展误写成“非法文件长度”测试；修正的是测试字段偏移，并新增合法零尾部用例，未放宽解析条件。该失败XML保留；预检dev-001和001也是旧审计来源版本，不改写其摘要。

完整Llama2在此验收器中的成功终态分支尚无端到端证据。当前只有真实小图的完整审计和完整运行的拒绝证据，不能把核验器已实现或测试通过当成完整模型已通过。

## 完整运行结束后的命令

只在实际运行结束后使用新的审计目录；运行未结束时会拒绝，不会重启模型：

```bash
.venv/bin/python -m scripts.verify_mlx_system_model \
  --run artifacts/tagged/model-e2e/llama2-rocket-full-001/llama2-resident \
  --program artifacts/tagged/model-e2e/llama2-physical-full-001/program.json \
  --lifetimes artifacts/tagged/model-storage-002/model-lifetimes.json \
  --reference artifacts/tagged/model-e2e/llama2-reference-numeric-001/inventory.json \
  --source-inventory artifacts/tagged/model-e2e/llama2-reference-005/inventory.json \
  --output artifacts/tagged/model-e2e/llama2-system-audit-NEW
```

原生完整物理运行仍用`scripts.verify_mlx_model_numeric`独立验收，不能用本系统报告替代它的执行记录，反之亦然。

此前[组件发布快照](../artifacts/tagged/publication-20260908-001/README.md)已推送为`3ae584f3a183b493749e36adae10b7d81a11bba1`。该目录的摘要描述当时提交的文件，不被本次新门禁重新标成完整模型证据。
