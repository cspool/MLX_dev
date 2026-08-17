# DSAGEN-memory scalability mechanism

H67 changes only H64's fixed backend to the H66-validated standalone DSAGEN
scratchpad. All 20 configurations run twice with exact replay and conserved
instruction/event/route/memory work. No Figure 23 target is read.

| N | SIMD32/4x4 | SIMD8/8x8 | SIMD32/8x8 |
|---:|---:|---:|---:|
| 512 | 3.947x | 1.024x | 4.095x |
| 1,024 | 4.011x | 1.020x | 4.073x |
| 2,048 | 3.979x | 1.018x | 4.074x |
| 4,096 | 4.012x | 1.022x | 4.091x |
| 8,192 | 4.074x | 1.038x | 4.147x |

Relative to fixed memory, 4x4 runs slow by roughly 3.1x and 8x8 runs by 11.3x.
SIMD32 remains effective because it coalesces four times the lanes per request.
Mesh scaling nearly disappears because all 64 PEs contend for the same
four-entry ordered InputBuffer.

This is a supported reproduction of the *open DSAGEN* memory mechanism, not a
Figure 23 reproduction. Fig. 9(a) depicts multiple scratch-memory attachment
points around the array, while the current adapter funnels the entire overlay
through one upstream DSAGEN buffer. The paper does not identify that buffer as
MLX's scratchpad configuration.

The immutable mechanism result is
`artifacts/results/spad-scalability-run072.json`. It prevents treating DSAGEN's
single-buffer memory as an MLX parameter merely because DSAGEN is the spatial
simulator substrate.
