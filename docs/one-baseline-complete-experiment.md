# MLX versus one serial spatial baseline

## Result

The narrowed simulator goal is complete with one main baseline and one MLX
implementation on the same complete functional workload.

| Property | Main baseline | MLX |
|---|---|---|
| Execution model | one active logical layer/tag | data-ready multi-layer execution |
| Active-window limit | 1 | 13 |
| Complete-block cycles | 426 | 341 |
| Complete-block speedup | 1.0x | 1.249x |
| Maximum active tags | 1 | 13 |
| Issues before producer-tag completion | 0 | 41 |
| Functional result | pass | pass |

Both sides use the same programmable spatial array, PEs, FU timing, register
file, four pipelines, skip-hop routing, fixed-memory backend, instructions,
inputs and numerical operations. The baseline retains whole-component
predecessor barriers. MLX replaces only those barriers with exact
store-completion events for the 24 dynamically linked values.

## Performance curve

| Cumulative workload | Baseline cycles | MLX cycles | Speedup |
|---|---:|---:|---:|
| BSMM + FFT-CMP | 221 | 136 | 1.625x |
| + Attention | 306 | 221 | 1.385x |
| + causal SWA | 391 | 306 | 1.278x |
| + elementwise / complete block | 426 | 341 | 1.249x |

All four points exceed the frozen `1.20x` clear-improvement threshold. The
shallower-to-complete reduction in gain is retained rather than hidden; the
complete block still shows the same qualitative paper conclusion that
multi-layer scheduling hides dependency/data-movement latency better than
serial layer execution.

## Functional and same-work evidence

- Complete chain: `BSMM -> FFT-CMP -> Attention -> causal SWA ->
  residual/scale/SiLU`.
- Six covered payload claims: the five operators plus the complete Transformer
  block.
- Both architectures match independently recomputed values at every cumulative
  boundary and all eight final outputs.
- Maximum absolute error: `2.78e-17` against a `1e-12` limit.
- Identical complete work: 466 operations, 162 memory requests, 97 boundary
  events and 139 route hops.
- 16 configs and 48 debug/optimized/sanitized executions; functional-enabled
  and timing-only statistics are identical.

## Evidence

- Paired experiment: [run176](../artifacts/results/data-ready-complete-block-run176.json)
- Final certificate: [run177](../artifacts/results/one-baseline-goal-certificate-run177.json)
- Goal contract: [one-baseline-goal.md](one-baseline-goal.md)
- H171 protocol:
  [protocol.md](../experiments/h171-data-ready-complete-block/protocol.md)

Fresh closeout verification reports Ruff passed and pytest `439 passed, 0
failed, 17 warnings`.

## Claim boundary

This proves the requested same experimental phenomenon and conclusion on a
transparent representative setup. It does not claim exact MLX paper numbers,
an external Xavier implementation, every paper experiment, RTL, area or power.
