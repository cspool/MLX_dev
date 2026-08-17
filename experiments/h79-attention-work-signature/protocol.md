# H79 protocol: target-free Figure 20 attention work signature

H79 derives a shape-specific execution signature for the two components of
structured Figure 20 attention at N=256/8192: three-branch FFT compression and
dense attention at retained length sN with D=4096 and s=0.5.

The signature follows the corrected spatial-PE contract, not a GPU-SM model.
It records static stages and scalar instruction instances by FU class:
FMA/ADD/SHUFFLE for FFT compression and FMA/FMAX/FEXP/ADD/FDIV for compressed
attention. The H50 source-derived complex radix template contributes four FMA
and six ADD instructions per butterfly pair. Figure 12 and the executable SWA
template require one final FDIV per output element.

The audit must reconcile H75's conventional analytical counts exactly:

- FFT uses ten real FLOPs per radix-2 pair (five per point);
- compressed attention uses two FLOPs per FMA, one FMAX, one ADD, and four
  weighted FLOPs per FEXP;
- H75's analytical total omits the final FDIV, which H79 records separately
  rather than silently changing the frozen parent.

Support also requires proving that H57's one seven-stage FFT proxy cannot
represent either 16/26-stage FFT-compression schedule or the four-stage
compressed-attention FU mix. No Figure 20 performance target is read.

The immutable output is
`artifacts/results/attention-work-signature-run084.json`.
