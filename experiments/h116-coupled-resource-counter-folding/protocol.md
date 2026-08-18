# H116 protocol: coupled physical-resource counter folding

## Hypothesis

The physical productive PE-cycle counters emitted by all 192 H114 live-coupled
runs can be folded independently of Figure 22/25 targets. q=4/8 affine fits
should predict q=16/32 compute, load, store, xfer and FMA counters within 5%,
licensing full-path resource-utilization estimates with explicit counter
semantics.

H115 showed that completed FMA work/cycle does not reproduce Figure 25, while
MCP provides no independent basis to substitute residence. H116 therefore does
not compare either figure. It first tests whether the already implemented H71
physical counters are numerically stable under the new live memory/tile-major
execution. H61 supplies the productive-PE normalization semantics, while H71
supplies the later per-FU/FMA counter implementation; their roles remain
separate.

## Frozen counters

For every H114 q config, read only:

- `productive_pe_cycles_by_pipeline.compute`;
- `.load`, `.store`, and `.xfer` separately, matching Figure 22's physical
  data-supply segments; and
- `productive_pe_cycles_by_fu_class.fma`, retained as physical FMA residence
  and kept distinct from completed FMA issue throughput.

The primary normalization is each projected productive PE-cycle counter divided
by `full cycles * 16 physical PEs`, exactly as validated by H71. A metric that
is zero at all four scales is classified as exact-zero and is not forced through
a relative-error division.

## Acceptance gates

1. Frozen H114/H61/H71/config bytes qualify; H114/H71 are supported with
   integrity, while H61's target transfer is rejected with integrity but its
   productive-PE counter definition is frozen.
2. H114's run manifest qualifies; exactly 48 paths and 192 optimized q summaries
   bind through hashes and match run119 cycles.
3. All five counter keys are present/nonnegative at every q; physical PE count
   is 16 and every run completed.
4. q scales and path/family identities match H114 exactly; no target file is
   loaded.
5. Metrics that are zero at q=4/8/16/32 remain exact zero and are separately
   counted.
6. Every nonzero metric's q=4/8 affine fit predicts q=16/32 within 5%; all
   evaluated holdouts are finite.
7. Full cycles reuse only run119's passing model; no new cycle fit or timing
   parameter is introduced.
8. Every eligible full productive pipeline/FMA utilization is finite and in
   [0,1]; load/store/xfer remain separate rather than post-hoc merged.
9. Full FMA residence is reported separately from completed-work FMA issue;
   neither is renamed as Figure 25 achieved performance.
10. q=full_scale counter projections and exact-zero classifications are emitted
    only for paths whose five metric folds all pass.
11. Recomputing the audit is deterministic; source contains no Figure 22/25
    targets, residual scale, family correction or counter-selection rule.
12. H116 changes no simulator source and no active figure completion status;
    it only licenses or rejects target-free counter evidence.

Support requires all 12 gates. The immutable result will be
`artifacts/results/coupled-resource-counter-folding-run121.json`.
