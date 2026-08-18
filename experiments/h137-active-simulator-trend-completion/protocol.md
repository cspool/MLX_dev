# H137 protocol: dual-criterion active simulator certificate

## Hypothesis

The latest frozen evidence accounts for all simulator-dependent Figures 18-25
under both the user-directed qualitative criterion and the retained strict 10%
diagnostic. Exactly Figure 20 is currently trend-reproduced; no full figure is
strictly reproduced.

## Frozen trend policy

This protocol does not retroactively convert every numerical failure into a
pass. A speedup comparison must retain the same above-baseline direction and
predict at least 1.2x. An ordered non-speedup curve must later pass every
required series with Spearman rank correlation at least 0.70 and matching
first-to-last direction. Missing execution, comparison denominators, workload
identity, or metric identity cannot pass a trend gate. The 10% result remains
separate in every later audit.

Figure 20 already has a complete H136 trend audit. Figures 19, 22 and 25 have
complete numerical points but no frozen trend audit and therefore remain
pending. Figures 18/23 retain identity gaps; Figures 21/24 retain execution
gaps. H137 is a certificate, not a new per-figure target analysis.

## Acceptance gates

1. All eight latest evidence files qualify by byte and hash.
2. Figures 18-25 appear exactly once.
3. Figure 18 remains identity/provenance incomplete under both criteria.
4. Figure 19 is strict-rejected and trend-audit-pending.
5. Figure 20 is 8/8 trend-reproduced but only 1/8 strict cells.
6. Figure 21 remains execution-incomplete under both criteria.
7. Figure 22 is trend-audit-pending; Figure 23 remains identity-incomplete.
8. Figure 24 remains execution-incomplete; Figure 25 is trend-audit-pending.
9. Pending, partial, unmatched and missing-identity evidence is not promoted.
10. Primary completion is exactly 1/8, strict completion is 0/8, and neither
    global goal is complete.

Support means the certificate is internally correct. The immutable result will
be `artifacts/results/active-simulator-trend-completion-run142.json`.
