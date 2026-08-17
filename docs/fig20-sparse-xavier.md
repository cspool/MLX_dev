# Figure 20 sparse-CUDA Xavier transfer

H57 compares paper-static MLX BSMM/FFT proxies with execution-driven Xavier
PTX at N=256 and N=8K. QKV/FFN use B32 five-stage BSMM; Attention uses FFT.
Time is normalized per source-counted FMA-equivalent and clock.

All simulations pass, but none of the eight sparse-CUDA speedups is within 10%
(MAPE 80.4%, maximum error 94.4%). Fixed 15-W Xavier and 6.45-W MLX power also
fails to represent the plotted per-kernel energy variation. Dense Tensor Core
bars remain unavailable because H56 has no WMMA/Tensor Core kernel.

H57 is therefore an honest partial rejection, not a full Figure 20
reproduction. Author kernel definitions, Tensor Core baselines, and activity
power are required before the missing series can be validated.
