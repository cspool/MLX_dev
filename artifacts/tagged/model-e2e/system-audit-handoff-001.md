# 系统模型验收器续接记录

2026-09-08。本记录补充`active-validation.md`中“完整系统专用结果核验仍待补齐”的历史状态；运行状态仍必须查询实际进程。

- 上一目标阶段已完成组件发布：远程`cspool/MLX_dev`的`sys`分支确认提交`3ae584f3a183b493749e36adae10b7d81a11bba1`，502个文件，发布回归865项通过。用户原有`docs/mlx-riscv-system-simulation-goal.md`修改没有混入提交。
- 新增`src/mlxsim/model_system_evidence.py`、`scripts/verify_mlx_system_model.py`、`tests/test_system_model_evidence.py`及真实预检program的JSON fixture。新增文件不属于两个活动运行的冻结来源集合，也不在其目录扫描范围内。
- 正式真实预检审计为`../system-model-audit-preflight-002/report.json`：完整重新编译、命令/C检查器/ELF逐字节匹配，45源/21设备窗口/23361事务，全部输出字节检查绑定；完整模型字段仍false。
- `../system-model-audit-tests-003.xml`为最新回归，229项通过；001包含一次测试刺激字段偏移错误的失败，002修正后通过，003额外检查奇数ELF入口。历史结果不覆盖。运行中完整系统的最新拒绝证据为`../system-model-audit-active-rejection-002/failure.json`。
- 完整原生运行继续跟踪会话 **70284**、PID **1128327**；完整Rocket运行继续跟踪 **91280**、PID **1626333**。本轮检查时二者均存活，未重启、未修改冻结后端或任何完整图输入，均尚无完整终态验收。
- 后续：先检查实际进程/会话与终态；原生结束后独立运行数值门禁，Rocket结束后按`docs/mlx-system-model-audit.md`运行新系统审计。退出13/数值不符/退出上限与执行成功必须区分。GPU差异、其他模型/输入、跨算子共享资源、性能与RTL仍未关闭，不能标记目标完成。
