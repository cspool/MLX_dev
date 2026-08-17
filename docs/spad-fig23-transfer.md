# DSAGEN-memory transfer to Figure 23

H68 freezes H67's exact open-source DSAGEN-memory runs before joining the 15
Figure 23 targets.

| Series | Points within 10% |
|---|---:|
| SIMD32 / 4x4 | 5/5 |
| SIMD8 / 8x8 | 0/5 |
| SIMD32 / 8x8 | 0/5 |
| Overall | 5/15 |

Overall MAPE is 48.21% and maximum error is 73.07%. SIMD32 remains close to the
paper, but mesh scaling collapses to about 1.02x because all 64 PEs share one
four-entry ordered buffer. Joint scaling consequently remains near 4.1x rather
than the paper's 12.8–14.9x.

This rejection separates two provenance claims:

- DSAGEN/dsa-gem5 is a defensible open spatial simulator substrate for MLX
  control, routing, and PE extensions.
- DSAGEN's exact scratchpad organization is not a defensible MLX memory model.

Fig. 9(a)'s multiple array-edge scratch-memory attachment points motivate a
separate multi-port reconstruction, but the number and arbitration policy must
be registered from the diagram before execution rather than fitted to these
residuals.

The immutable result is
`artifacts/results/spad-fig23-transfer-run073.json`.
