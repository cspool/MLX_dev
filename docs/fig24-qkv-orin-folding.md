# Figure 24 QKV Orin repeat folding

## Outcome

H124 run129 is rejected with `audit_integrity=true`. Twelve detailed
GPGPU-Sim Orin runs execute block-128 QKV templates for B16/B32/B64 at
q=1/2/4/8 with exact FMA, CTA, checksum and configuration evidence.

The q1/q2 affine models pass every q4 holdout but fail every q8 holdout:

| Template | q4 error | q8 error |
|---|---:|---:|
| B16, 4 stages | 2.41% | 7.84% |
| B32, 5 stages | 2.39% | 8.17% |
| B64, 6 stages | 1.78% | 7.55% |

Only 3/6 holdouts pass. Holdout MAPE is 5.02% and maximum error is 8.17%.
All 21 diagnostic full-work mappings have exact integer q, but their templates
are ineligible and every full cycle/seconds estimate is null.

## Interpretation

q1/q2 still straddle launch/CTA saturation. The near-identical curvature across
all stage counts points to grid-scale behavior rather than an operator-specific
arithmetic bug. No Figure 24 target, MLX cycle or residual factor is used.

The next admissible test moves the fit to q4/q8 and executes new q16/q32
holdouts for all three templates. If those pass, the 21 QKV Orin estimates can
be emitted under the explicit block-128 proxy label; FFT-CMP and SWA remain
separate unfinished GPU contracts.

That extension is reported in
[fig24-qkv-orin-steady-state.md](fig24-qkv-orin-steady-state.md). q16 passes
within 0.78%, but q32 reveals a shared cache-capacity transition and invalidates
all full estimates again.

Evidence is in
[run129](../artifacts/results/fig24-qkv-orin-folding-run129.json), with the
frozen plan in
[H124 protocol](../experiments/h124-fig24-qkv-orin-folding/protocol.md).
