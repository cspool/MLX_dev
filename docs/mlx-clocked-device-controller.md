# 外部时钟与总线驱动的 C++ 设备控制器

2026-09-08。新增 [clocked_device/device.cc](../system_sim/clocked_device/device.cc)，把已有矩阵、向量和搬运周期后端接到外部请求/响应接口。它独立于正在运行的Spike插件和完整物理执行器，不修改二者的冻结实现，也没有新增PE/FU/调度器RTL。

## 与轮询驱动插件的区别

上层调用 `launch(descriptor_address, descriptor_bytes, source_id)` 后，每个系统边沿只调用一次 `tick(inputs)`。`request()` 和 `report()` 都是纯观察；不读status时设备仍按外部边沿前进，多读status也不会多执行周期。

```text
外部系统边沿
  → 描述符读取：经真实总线请求逐个8-byte响应取得wire
  → 既有C++后端：经同一总线取得张量值，执行微指令并写回
  → 写请求确认、输出覆盖检查、请求排空 → 完成
                     出错 → 保留在途所有权 → 排空 → 错误终态
```

控制器不持有权重/张量数据，也不在launch时直接读取宿主指针。1088-byte矩阵、4288-byte向量、15872-byte搬运wire使用原来的解码器及精度/布局/资源约束；首个描述符字即检查magic与尺寸是否匹配，随后检查完整编码、地址范围与描述符/数据重叠。合法视图仍不要求伪造输出写入。

描述符最后一拍响应提交后，下一个系统边沿对应后端局部cycle 0。后端时钟加原点映射到同一单调传输时钟；请求注册后下一边沿可见，ready不能改变已呈现请求的身份或数据。当前只有一个物理事务槽，不凭软件队列扩充目标并发资源。

## 请求归属、错误与完成

- 64-bit内部请求身份跨描述符、后端窗口和排空后的reset持续递增。它不是直接截断到HellaCache tag位宽的编码；实际信号连接仍需保存逻辑身份与有限tag的绑定。
- NACK只能针对已接受请求重发相同事务；成功响应或错误响应必须匹配在途身份。错身份、重复/过早响应或同时NACK与响应属于传输协议错误，不能被替换成任意合法回复。
- 每次launch清空已确认写入区间，只把成功写响应计为该任务产生的字节；完成时检查逻辑输出的每个元素都被本次确认写入覆盖。不能沿用上次结果作为“已完成”。
- 后端错误或周期上限会停止后端发射，但不会直接撤销已排队/已接受请求。排空期间保持busy；在响应被消费之前拒绝新launch和reset。完全排空后可以处理新的合法描述符。

`fetch_cycles`、`run_cycles`、`drain_cycles`为本次launch的阶段边沿数，包含错误发现的边沿；传输计数为设备实例累计值。有效执行和故障排空均检查阶段边沿总和。描述符验证/后端构造当前在一个阶段边界提交，其实际硬件控制成本尚未建模验证，不能用这里的周期声称完整系统性能。

## 当前证据

[clocked-device-002](../artifacts/tagged/clocked-device-002/report.json)完成15项pytest、16个C++驱动作业，并用ASan/UBSan及泄漏检查重放。普通与插桩的输出、事件、周期、拒绝结果一致。覆盖：

- 矩阵→向量缩放→类型转换的真实外部数据链与权重扰动；FP16中间结果转成FP32输出。
- 不查询状态和每周期额外查询7次的完整报告相同。
- ready反压、延迟响应和NACK；重复请求内容保持，成功提交身份不重复。
- 描述符magic/保留位/对齐/尺寸/重叠拒绝，描述符读错误与数据写错误，错误后排空和恢复。
- 周期上限到达时保持在途请求直至响应；busy时拒绝reset，排空后reset不回卷请求身份。

普通三任务链共8288个外部边沿、2696个成功事务，其中描述符读取2656个事务，实际张量读写40个事务。这是登记组件测试数据，不是模型/CPU/DRAM性能。001是阶段计数修正前的历史证据；002额外保证周期上限检测边沿没有漏计。

设备wire、主机控制/编排、文件装载、稀疏存储及本控制器的[154项相关回归](../artifacts/tagged/clocked-device-tests-001.xml)通过。

```bash
.venv/bin/python -m scripts.verify_mlx_clocked_device \
  --output artifacts/tagged/clocked-device-NEW
```

必须选择新输出目录。驱动器中的内存是独立测试端点，初始数据由测试环境装入，不冒充CPU装载；实际CPU装载由[另一条已登记路径](mlx-file-asset-loading.md)验证，不能混合两个结果宣称完整系统通过。

## 下一步实际 Chipyard 连接的约束

本地 [LazyRoCC.scala](../build/chipyard-native/generators/rocket-chip/src/main/scala/tile/LazyRoCC.scala) 为每个RoCC插入 [SimpleHellaCacheIF](../build/chipyard-native/generators/rocket-chip/src/main/scala/rocket/SimpleHellaCacheIF.scala)。该层已经保存请求、寄存下一拍store数据并处理cache的s2_nack重放。因此，接在其requestor一侧的C++桥不能再把同一个cache NACK重发一次，否则会重复提交。C++通用端点的NACK支持不表示实际RoCC侧也应重复使用它。

后续需验证：RoCC命令/返回反压、64-bit内部身份与有限tag绑定、byte/half/word读写、SimpleHellaCacheIF拥有的重放、异常处理边界，以及CPU fence与结果可见性。该本地接口对cache异常使用assert，不可擅自宣称已有可恢复页故障协议。完整模型仍需真实CPU编排及系统结果核对；性能和RTL扩展继续后置。
