# MLX workload lowering toolchain

## Status before H187

The repository had useful but disconnected paths:

- `mlxsim.workloads.compile_workload` produced analytical kernel profiles;
- Figure23 cloned a complete-block template into PE/tag JSON;
- Figure19 separately built FFT/global-BSMM sources, DPU-memory JSON and
  coupled overlay JSON;
- Figure20 composed analytical simulator and GPU evidence in figure-specific
  scripts.

There was no common input schema, graph validation, backend selection,
execution manifest or end-to-end lineage.

## Unified path

The input is `configs/workloads/mlx_fig19_20_23_v1.yaml`. Each graph declares:

- model family, batch, sequence/hidden/FFN dimensions and precision;
- operator IDs, kinds and dependencies;
- lowering adapter and native execution format;
- symbolic source/config references resolved by the experiment config.

In compact form, the path is `model/operator graph YAML -> schema/DAG
validation -> lowering adapter -> native simulator artifact -> execution and
audit manifest`.

`scripts/lower_mlx_workload.py` validates the DAG and calls
`mlxsim.workload_lowering`. Its outputs are native simulator artifacts—not a
new opaque IR:

- `mlx_overlay_json`: blocks, tags, dependencies/events, PE coordinates,
  instructions, pipelines, routing and memory mode;
- `mlx_dpu_memory_json`: the overlay plus non-stop buffer/SPM/DMA settings;
- `analytical_kernel_profile_json`: `Workload`, stages/tags, operations,
  off-chip bytes and output elements consumed by `MLXSimulator`.

`scripts/run_lowered_mlx_workload.py` dispatches each format to the appropriate
native runner and records two replay summaries. The H187 auditor verifies
source-spec hash, DAG order, node lineage, lowering replay, artifact schemas,
execution completion and replay identity.

This toolchain is a repository implementation built from the available MLX
simulator adapters. It is not claimed to be the paper authors' unpublished
LLVM/spatial-assembler toolchain.
