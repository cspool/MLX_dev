# H173 protocol: Xavier-class dense Transformer end-to-end functionality

## Hypothesis

The existing Xavier resource proxy can execute a complete dense Transformer
block rather than isolated BSMM/FFT/SWA kernels. A deterministic two-layer
CUDA chain covering all Figure-21 operator families will match an independent
host reference for three token counts under detailed GPGPU-Sim execution.

## Functional chain

Each layer executes RMSNorm, dense Q/K/V projections, RoPE, causal dense
QK-softmax-SV attention, dense output projection and residual, a second
RMSNorm, gated SiLU FFN, down projection and final residual. Dimensions are
intentionally small (`D=8`, `FFN=16`, N=4/8/16) so every operation can be
executed cycle-accurately while retaining the end-to-end dependency graph.

The host independently performs the same operations and compares every final
element, not only a checksum. Fixed deterministic weights and inputs are shared
across GPU and host paths.

## Baseline identity

Use H56's resource-edited Xavier configuration: eight SMs and Xavier clocks on
the tested Titan-V SM70 timing template. This is a Xavier-class proxy, not a
vendor-validated SM72 model. Functional completion does not promote its timing
to strict hardware validation.

## Acceptance gates

1. H56 config/result and H171 MLX end-to-end evidence qualify by hash/status.
2. One CUDA program contains all eleven registered operator groups.
3. N=4/8/16 each execute two complete layers and exactly 28 kernel launches.
4. All runs use the frozen Xavier config and detailed PTX mode.
5. Each run exits normally with positive cycles/instructions/CTAs.
6. GPU and independent host outputs are finite and maximum absolute error is
   <=1e-5 for every shape.
7. Increasing token count increases executed cycles and instructions.
8. Source/config/run artifacts are hashed and the binary is reproducible.
9. MLX H171 and Xavier H173 jointly cover end-to-end functionality, while their
   different structured/dense algorithms remain explicit.
10. No Figure-21 performance target or fitted timing value is consumed.

The immutable result will be
`artifacts/results/xavier-e2e-functional-run178.json`.
