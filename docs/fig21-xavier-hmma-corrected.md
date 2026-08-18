# Corrected Figure 21 SASS-HMMA projection model

H151 run156 is supported with `audit_integrity=true` and 10/10 gates. It keeps
H146's four trace replay cycles unchanged and corrects only the source-level
work unit from 4096 to 256 FMA per SASS HMMA.

| Repeat | Corrected FMA work | Unchanged cycles |
|---:|---:|---:|
| 16 | 262,144 | 128 |
| 32 | 524,288 | 240 |
| 64 | 1,048,576 | 464 |
| 128 | 2,097,152 | 912 |

The 16/32 fit still predicts both holdouts exactly. Applying it to the unchanged
H91 32-layer dense projection work yields 1.599/3.198/6.397/12.793/25.586
seconds for N128-N2048.

This remains a source-derived compute-only SASS-HMMA traceg proxy, not captured
cuBLAS timing. The correction is supported by the frozen Volta 16-SASS/PTX
decomposition, not a Figure 21 target. Active completion remains 3/8.

Evidence is in
[run156](../artifacts/results/fig21-xavier-hmma-corrected-run156.json), with the
frozen plan in
[H151 protocol](../experiments/h151-fig21-xavier-hmma-corrected/protocol.md).
