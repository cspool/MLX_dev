# H187 protocol: unified model-graph to simulator toolchain

## Hypothesis

The fragmented Figure19/20/23 compilers can be organized behind one explicit
workload schema and lowering API that preserves model shapes and graph
dependencies, emits each simulator's native execution format, runs every output
twice, and exposes node-to-artifact-to-result lineage.

## Pipeline

```text
model/operator graph YAML
  -> schema + DAG validation
  -> backend adapter
     -> MLX overlay JSON (+ DPU-memory JSON when required)
     -> analytical KernelProfile JSON
  -> native simulator runner
  -> replay hashes + execution summaries
  -> lineage/numerical audit
```

Registered adapters are:

- `fig23_complete_block`: reuses the complete block compiler and adds the H184
  latency-service contract;
- `fig19_coupled_paths`: lowers FFT2D/global-BSMM sources through the current
  DPU-memory and multiport-SPM mapping;
- `analytical_kernel_profiles`: converts Figure20 operator nodes to validated
  `Workload` and `KernelProfile` objects consumed by `MLXSimulator`.

## Acceptance gates

1. All eleven frozen inputs qualify and required results retain status/integrity.
2. One schema validates exactly three DAGs and fourteen unique operator nodes.
3. Every dependency resolves and all three topological orders are complete.
4. Lowering produces twelve executable units: four detailed overlays, three
   DPU-memory configs and eight analytical profiles.
5. Every detailed overlay contains blocks, tags, PE coordinates, event
   dependencies, instructions and explicit memory mode.
6. Every analytical profile contains stages, tags, operations, bytes and the
   original high-level workload shape.
7. Lowering replay is byte-identical and every source node has lineage.
8. All twelve units execute twice (24 executions), complete successfully and
   produce identical replay summaries.
9. Representative Figure23 raw/reported cycles, Figure19 DPU-memory execution
   and all Figure20 analytical profiles agree with their registered contracts;
   H184/H185/H186 remain within 15%.
10. Documentation, source, manifests and tests qualify; the result claims a
    working toolchain rather than an unpublished author compiler.

The immutable result will be
`artifacts/results/unified-workload-lowering-run192.json`.
