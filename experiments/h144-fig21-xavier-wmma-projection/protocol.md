# H144 protocol: target-free Xavier WMMA projection model

## Hypothesis

A real WMMA FP16-input/FP32-accumulate kernel executes in detailed GPGPU-Sim on
the frozen H56 Xavier configuration, and a two-anchor repeat model predicts two
larger holdouts within 5%. Only then may it emit five dense projection-cycle
estimates from H91's exact QKV/output/FFN FMA counts.

## Kernel and work

Launch 64 one-warp CTAs. Each CTA loads one 16x16x16 WMMA A/B tile, repeats
`mma_sync` into an FP32 accumulator, and stores a 16x16 tile. One repeat is
exactly 4096 FMA equivalents per CTA. Compile PTX for compute_70 with CUDA 11.8
and require actual WMMA load/MMA/store instructions; H56's four tensor units
must remain enabled.

Freeze repeats 16/32 as affine fit anchors and 64/128 as unseen holdouts. The
checksum is analytically fixed by constant inputs. Fit cycles only against exact
WMMA FMA work. If both holdouts pass, sum H91 dense QKV/output/FFN FMA counts,
multiply by 32 dense Xavier layers, and predict one projection total per N. Do
not use Figure 21 speedup targets, GPU efficiency factors, cuBLAS timing, or
H77's sparse CUDA-core model.

## Acceptance gates

1. H56/H91/H143 qualify and retain required status/integrity.
2. H56 uses the frozen 1.377-GHz tensor-enabled Xavier config with four tensor
   units per SM.
3. CUDA 11.8 compilation succeeds and emitted compute_70 PTX contains WMMA
   load, `mma.sync`, and store instructions.
4. All four detailed GPGPU-Sim runs exit normally with positive cycles,
   instructions and CTAs, and pass checksum <=1e-5.
5. Every run records exactly 64*repeat*4096 FMA equivalents and 64 CTAs.
6. The 16/32-repeat affine cycle model has positive slope/intercept prediction.
7. Both 64/128-repeat holdouts pass <=5% relative cycle error.
8. H91 provides five shapes with four positive dense projection components;
   32-layer totals and predicted cycles/seconds are finite and positive.
9. Auditor/source/test contain no Figure 21 target, target factor, efficiency
   fit, sparse projection substitution or post-result model choice.
10. Result remains target-free and labeled a transparent WMMA proxy; Figure 21
    and active completion stay incomplete at 3/8.

The immutable result will be
`artifacts/results/fig21-xavier-wmma-run149.json`.

## Execution stopping rule

If the first 16-repeat fit anchor parses genuine WMMA PTX and reaches kernel
enqueue but the simulator exits abnormally before producing cycle/checksum
statistics, stop the remaining repeat jobs. Record H144 as rejected with the
failure log and emit no projection estimates. Repeating the same unsupported
functional-PTX instruction cannot satisfy gates 4-8; a trace-driven successor
must be separately pre-registered rather than changing H144's execution mode.
