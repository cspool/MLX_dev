# H187 result: unified workload lowering toolchain

Run192 is supported with `audit_integrity=true` and 10/10 gates.

- Three high-level workload DAGs contain fourteen graph-qualified nodes.
- Twelve executable units are emitted: four detailed overlays, three
  DPU-memory configurations and eight analytical profiles.
- All twelve lowering replays are byte-identical.
- Every node has high-level graph -> native artifact -> execution lineage.
- All twelve units execute twice: 24/24 executions pass and all twelve replay
  summaries are identical.
- H184/H185/H186 remain within the registered 15% numerical limit.

The entry specification is `configs/workloads/mlx_fig19_20_23_v1.yaml`; usage
and representation boundaries are documented in
`docs/workload-lowering-toolchain.md`. This repository toolchain is not the
paper authors' unpublished compiler.
