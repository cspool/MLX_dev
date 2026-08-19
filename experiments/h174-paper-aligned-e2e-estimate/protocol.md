# H174 protocol: paper-informed Figure-21 end-to-end estimate

## Hypothesis

A compact three-parameter model can reconcile the corrected open-simulator
Xavier services and H91's 32-layer MLX work contract with Figure 21's decreasing
end-to-end speedup curve. The output is a paper-informed estimate, not an
independent reproduction.

## Frozen base evidence

- Xavier projection uses H151's corrected 256-FMA/SASS-HMMA semantics, replacing
  the invalid 16x-faster H146 label.
- Xavier dense attention/elementwise use H147's executed SP/SFU/ALU service
  curves.
- MLX work uses H91: 24 structured layers, eight dense layers and 32
  elementwise paths for N=128..2048, batch 8.
- H171/H173 provide actual end-to-end functional executions for structured MLX
  and dense Xavier-class paths.
- H169 marks N=2048's two-kernel MLX split as a limitation; the fit absorbs its
  unreported cost globally and does not invent a per-point penalty.

## Model

For every N:

`Xavier_seconds = corrected_compute_service_seconds + L_xavier`

`MLX_seconds = a_linear * linear_work_TOP + a_attention * attention_work_TOP`

The same three nonnegative parameters (`L_xavier`, `a_linear`, `a_attention`)
are fitted jointly to all five digitized Figure-21 speedups by least squares on
speedup residual. No per-size parameter, lookup correction or interpolation is
allowed.

The fitted values represent effective services for unmatched software/hardware
conditions. In particular, `a_linear` must not be relabeled as the physical
1-TOp/s peak.

## Robustness

Repeat the fit five times, holding out one sequence length each. Report every
held-out speedup and error; require the maximum leave-one-out error <=25%.

## Acceptance gates

1. All seven input artifacts qualify and retain their stated status/integrity.
2. Exact five-row Figure-21 identity and 24+8 layer work are reconstructed.
3. Corrected Xavier base time is positive and increasing for all rows.
4. Exactly three global nonnegative parameters are fitted; no per-point values.
5. MLX/Xavier absolute estimates are positive and finite.
6. Estimated speedup is strictly decreasing and remains above one for all rows.
7. All five fitted relative errors are <=10% and MAPE <=5%.
8. Five leave-one-out diagnostics are emitted with maximum error <=25%.
9. Both MLX and Xavier end-to-end functional parents remain supported.
10. The result is labeled paper-informed estimation, consumes targets openly,
    and claims neither exact paper numbers nor independent validation.

The immutable result will be
`artifacts/results/paper-aligned-e2e-estimate-run179.json`.
