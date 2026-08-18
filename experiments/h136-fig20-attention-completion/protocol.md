# H136 protocol: Figure 20 Attention completion

## Hypothesis

H135's two frozen Attention speedups reproduce Figure 20 Attn-256/Attn-8K
within 10%; after inserting them without changing H88's six projection cells,
the full eight-cell Figure 20 ledger is reproduced.

## Frozen join

Map N256/N8192 to H88 target indices 1/5. Compare 3.454x/3.152x directly with
the frozen 1.4x/3.1x targets. Preserve every H88 projection record byte-for-
semantic-value. No clock, component, factor, scale or offset change is allowed.

## Acceptance gates

1. H135/H88 qualify and are supported with integrity.
2. H88 contains exactly six projection failures and two Attention incomplete
   cells at indices 1/5.
3. Mapping copies both H135 speedups and H88 targets exactly.
4. Values/errors are finite and positive.
5. Both Attention cells must pass within 10%.
6. Six projection cells remain unchanged in value/status/evidence.
7. Refreshed ledger contains exactly eight unique indices and no incomplete
   cell.
8. Full Figure 20 support requires all eight cells pass, not only Attention.
9. Auditor/test contain no fit, component/clock factor, scale or offset.
10. Active completion increments only on 8/8; otherwise remains 0/8.

The immutable result will be
`artifacts/results/fig20-attention-completion-run141.json`.

## User-directed acceptance amendment before result generation

The user relaxed the project-wide primary criterion before H136 produced a
target comparison: numerical 10% remains a diagnostic, while completion is
based on matching comparative direction with an obvious improvement. H136
therefore adds a frozen trend gate: both target and prediction must indicate a
speedup over baseline, and the prediction must be at least 1.2x. Attention
requires 2/2 trend passes and the full ledger 8/8; strict 10% counts/errors are
still emitted unchanged. This amendment is not derived from H136 residuals.
