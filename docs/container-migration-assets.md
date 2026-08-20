# MLX 容器迁移与资产复用

更新日期：2026-08-20

## 结论

同一宿主机上的新容器通过 bind mount 可以直接复用整个工作区，包括未纳入
Git 的运行产物、第三方源码、模型、checkpoint 和编译工具链。跨宿主机迁移
时，源码仓库、容器镜像和大容量数据应分别迁移；模型、构建目录和环境产物
不进入项目 Git 历史。

## 当前资产分类

| 资产 | 约占用 | MLX+RISC-V 系统仿真是否必需 | 迁移方式 |
|---|---:|---|---|
| 项目源码、RTL、配置、结果证书、版本/哈希清单 | Git 仓库约 97 MiB（压缩对象） | 是 | GitHub |
| 未跟踪运行产物 | 1.44 GB / 6,612 文件 | 否；均可由脚本重建 | 同机 bind mount；必要时单独归档 |
| 模型权重 | 42 GB | 否；只用于训练/质量实验 | 模型仓库、共享缓存或 rsync |
| 训练 checkpoint | 4.4 GB | 否 | 共享缓存、对象存储或 rsync |
| DSAGEN/Chipyard 工作树与构建 | 14 GB | Chipyard 环境需要，现成镜像已提供 | `chipyard_ready:v1` 镜像或独立上游 checkout |
| Python/CUDA 环境 | 6.3 GB | 取决于实验 | 容器镜像；不要复制进 Git |

`third_party/README.md` 已记录第三方上游 URL、commit pin、许可证和用途；
`configs/training/quality_v1.yaml` 已记录主要模型来源、文件大小和 SHA256。
最终机器结果和关键运行 manifest 已纳入 Git。未跟踪目录主要是重复 lowering
配置、回放副本、网表、仿真可执行文件和中间日志。

## 同一宿主机迁移

保留 `/workspace/MLX_dev`，创建容器时挂载同一路径：

```bash
docker run -it \
  --name mlx_chipyard_dev \
  --shm-size=16g \
  -v /workspace/MLX_dev:/workspace/MLX_dev \
  -w /workspace/MLX_dev \
  chipyard_ready:v1 \
  /bin/bash
```

这会直接复用所有受管和未受管文件，无需上传模型或运行产物。

## 跨宿主机迁移

先迁移源码：

```bash
git clone https://github.com/cspool/MLX_dev.git /workspace/MLX_dev
```

容器镜像没有可用 registry 时可离线传输：

```bash
docker save chipyard_ready:v1 | zstd -T0 -o chipyard_ready_v1.tar.zst
```

把归档复制到新宿主机后加载：

```bash
zstd -dc chipyard_ready_v1.tar.zst | docker load
```

只有重新运行模型质量实验时才同步大模型和 checkpoint：

```bash
rsync -a --info=progress2 /workspace/MLX_dev/third_party/models/ NEW_HOST:/workspace/MLX_dev/third_party/models/
rsync -a --info=progress2 /workspace/MLX_dev/third_party/checkpoints/ NEW_HOST:/workspace/MLX_dev/third_party/checkpoints/
```

若必须保存不可重算的环境产物，单独归档选中的实验目录，不要提交整个
`artifacts/environment`：

```bash
tar --zstd -cf mlx-selected-artifacts.tar.zst \
  artifacts/environment/EXPERIMENT_ID \
  artifacts/results/RESULT_FILE.json
```

## Git 边界

- 应提交：源码、RTL、测试、配置、补丁、文档、小型 golden、结果摘要、来源
  与哈希清单。
- 不应提交：模型权重、checkpoint、虚拟环境、第三方构建树、VCD、网表、仿真
  可执行文件和可再生的大型 lowering/replay 目录。
- 第三方源码优先记录上游 commit 或使用独立 submodule；许可证不明确的源码
  不进行 vendoring。
- Chipyard/RISC-V 系统仿真产生的新原始轨迹保存在工作区或对象存储，只把
  汇总结果、协议、manifest 和必要的小型回归样本提交到 Git。
