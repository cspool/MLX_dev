# Exact Figure 24/25 paths

H101 replaces every historical Figure 24/25 micro-proxy with 48 unique
batch-32 case/operator paths covering variable-depth three-branch FFT-CMP,
B16/B32/B64 QKV BSMM, and W128/Q32 or W256/Q64 SWA.

The four-strip baseline compiles 192 q=4/8/16/32 configurations and executes
each twice through the four-port DSAGEN scratchpad adapter. All 384 executions
complete and replay byte-identically. The audit passes 48/48 full scalar
FU/FP16-byte contracts, 192/192 physical FU-class checks, and 96/96 cycle
holdouts. Cycle MAPE is `1.58e-5` and maximum relative error is `4.81e-4`.
The byte contract covers logical path inputs/outputs, not complete off-chip
DRAM traffic.

The largest B32/N=8192/q32 path takes 504,890,785 cycles. The initial 500M
watchdog was therefore below a legitimate execution; the final watchdog is
derived as at least twice each frozen config's dynamic instruction count.

This is target-free execution evidence, not a Figure 24/25 reproduction. Its
immutable result is
`artifacts/results/fig24-25-exact-paths-run106.json`.
