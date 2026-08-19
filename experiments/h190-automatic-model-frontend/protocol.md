# H190 protocol: automatic PyTorch FX and ONNX frontend

## Hypothesis

The same real structured-block module represented as a PyTorch FX GraphModule
and an ONNX ModelProto can be imported automatically into one canonical MLX
graph, planned without manual node YAML, lowered to KernelProfiles and executed
reproducibly.

## Frontend pipeline

1. Trace a six-module PyTorch structured block with a custom FX leaf-module
   registry and run FX ShapeProp on a real input tensor.
2. Construct and serialize the equivalent ONNX graph with standard and `mlx`
   domain nodes, initializers and explicit tensor shapes; parse the ModelProto.
3. Legalize both sources to `rmsnorm/bsmm/fft_cmp/attention/bsmm/elementwise`.
4. Form maximal supported CDCs, assign topological tags, round-robin PEs,
   bank-rotating registers and aligned liveness-aware SPM/DMA ranges.
5. Emit one KernelProfile per node and execute every profile twice with
   MLXSimulator.

## Acceptance gates

1. All three frozen inputs qualify and required parents retain status/integrity.
2. PyTorch FX and ONNX are both actual installed/imported frontends.
3. Each frontend yields exactly six legal nodes in the registered order.
4. Canonical kind, shape and dependency sequences match across frontends.
5. All twelve source nodes retain source-name/op/shape lineage.
6. CDCs cover every node once; tags are unique and topologically ordered.
7. PE coordinates, registers/banks and aligned SPM/DMA ranges are legal.
8. Twelve KernelProfiles contain positive operations/bytes/stages.
9. Twenty-four executions finish with finite metrics and replay-identical
   summaries.
10. Source/config/tests qualify and the generated canonical graph requires no
    manually authored operator-node YAML.

The immutable result will be
`artifacts/results/automatic-model-frontend-run195.json`.
