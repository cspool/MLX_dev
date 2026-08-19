# H177 protocol: native RTX 4090 replacement for Figure 24

## Hypothesis

Figure 24 can be replaced by a complete local RTX 4090 experiment without
pretending that the GPU is Orin or RTX3090. Native CUDA service curves for every
FFT depth, BSMM depth and SWA window can extrapolate the 42 exact-work contracts
and produce a transparent MLX-versus-4090 sweep.

## Native service experiment

Use GPU0, identified as RTX 4090/SM89. Ten service configurations cover:

- FFT-CMP depths 18/20/22/24/26;
- BSMM depths 4/5/6;
- SWA windows 128/256.

Each service runs at 65,536 and 131,072 elements for a two-point affine
time-versus-FMA model. A 262,144-element run is held out and must be predicted
within 10%. A separate 1,024-element execution compares every output with an
independent CPU reference. GPU clocks are not locked; identity, driver, clocks
and power limit are recorded.

## Full-work projection

H100 supplies the exact Figure-24 FU work for seven cases and six operators.
H102 supplies target-free full MLX cycles. For each of 42 rows:

1. choose the matching native 4090 service curve;
2. evaluate it at the registered full FMA count;
3. convert H102 cycles at 1 GHz to MLX seconds;
4. report `RTX4090_seconds / MLX_seconds`.

This is a measured-service extrapolation, not direct execution of the full
multi-billion-element tensors. No Figure-24 Orin target participates.

## Acceptance gates

1. H100/H102 inputs qualify and retain full-work status.
2. Runtime identity is GPU0 RTX4090, compute capability 8.9, CUDA 13.2.
3. Exactly ten service configurations are present.
4. Ten correctness smokes match CPU within 1e-5.
5. Exactly 30 timed native runs complete with positive event timings/work.
6. Every held-out service prediction is within 10%.
7. Exactly 42 Figure-24 rows cover seven cases and six operators once.
8. Every MLX/4090 absolute time and ratio is positive and finite.
9. Source, binary, logs, GPU snapshots and results are hashed.
10. Result is labeled native-4090 replacement/exploration and consumes no
    original Orin/RTX3090 target value.

The immutable result will be
`artifacts/results/fig24-rtx4090-native-run182.json`.
