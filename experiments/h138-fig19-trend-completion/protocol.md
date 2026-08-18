# H138 protocol: Figure 19 dual-criterion trend completion

## Hypothesis

H130's current-coupled MLX simulator preserves all three Figure 19 latency
curves, and its total latency is at least 1.2x faster than H13's independently
executed official open FABNet simulator at all four sequence lengths. Figure 19
therefore completes under the H137 qualitative policy while remaining a strict
numerical rejection.

## Frozen join

Use H130's 12 target/prediction points without changing its 24-layer/1-GHz
composition. Use H13's four `large`/BE-40 upstream simulator latencies as the
comparison baseline; its large absolute mismatch remains a strict diagnostic
and is not normalized. H13's digitized MLX totals must equal H130's targets,
which independently verifies the sequence/series join.

Each attention, FFN and total target/prediction curve must have Spearman rank
correlation at least 0.70 and matching first-to-last direction. Each H13
FABNet-latency / H130 MLX-total ratio must indicate the same above-one direction
as the paper ratio and reach at least 1.2x. No scale, offset, clock, layer,
efficiency or residual fit is allowed.

## Acceptance gates

1. H13/H130 qualify and retain their frozen rejected verdicts; H130 integrity
   and H13 upstream/digitization integrity pass.
2. Both parents contain exactly the same four sequence lengths.
3. H130 contains 12 finite MLX points across exactly three complete series.
4. All three curves pass Spearman and endpoint-direction gates.
5. H130 MLX total targets equal H13's digitized MLX totals exactly.
6. All four official FABNet and current MLX total latencies are finite/positive.
7. H13's four reported-speedup cross-checks remain passing.
8. All four open-simulator comparisons share the paper's speedup direction and
   predict at least 1.2x.
9. Strict diagnostics remain truthful: H130 is 0/12 and H13 is 0/4 within 10%.
10. Figure 19 increments primary completion to 2/8 only on 3/3 curves and 4/4
    comparison passes; strict full-figure completion remains 0/8.

The immutable result will be
`artifacts/results/fig19-trend-completion-run143.json`.
