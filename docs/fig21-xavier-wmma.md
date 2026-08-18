# Figure 21 functional-PTX WMMA attempt

H144 run149 is rejected with `audit_integrity=true` at 5/10 gates. The CUDA
11.8 compute_70 build is genuine: extracted PTX contains both WMMA loads,
`wmma.mma.sync` and WMMA store instructions. The frozen H56 configuration also
retains four tensor-core units per SM at 1.377 GHz.

The first 64-CTA, repeat-16 anchor parses the PTX, initializes the detailed
model and reaches kernel enqueue. GPGPU-Sim then exits with code 139 before
reporting cycles, instructions, CTAs or the application checksum. The
registered foundational-failure rule stops repeats 32/64/128.

No affine model or projection estimate is produced. This is an execution-mode
limitation, not evidence that scalar FMA timing should replace TensorCore
timing. H145 must use Accel-Sim's trace-driven specialized-unit path and retain
its cross-ISA trace provenance explicitly.

Evidence is in
[run149](../artifacts/results/fig21-xavier-wmma-run149.json), with the frozen
plan and stopping rule in
[H144 protocol](../experiments/h144-fig21-xavier-wmma-projection/protocol.md).
