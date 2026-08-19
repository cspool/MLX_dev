# H167 protocol: Figure 22 held-out resource-domain schemas

## Hypothesis

At least one complete H166 resource-domain schema, applied unchanged across
both operators, all sizes and all four resources, reproduces all eight Figure
22 curves under the frozen trend criterion. If every schema fails, the
remaining error cannot be repaired by counter/capacity semantics alone and
requires an execution/workload change.

## Pre-registered schemas

All schemas use one fixed mapping for the full 64-cell matrix:

1. `physical_pe`: H163/H166's original per-PE physical-capacity counters.
2. `component_issue`: compute global busy time over overlay cycles, external
   SPM read/write service over PE-cycle capacity, and xfer issues over PE-cycle
   capacity.
3. `component_hop`: the same schema but xfer uses routed hop service rather
   than issued instructions.
4. `payload_bandwidth`: compute and xfer as above; load/store payload bytes use
   the inferred 512 B/cycle SIMD8 payload capacity.
5. `wire_bandwidth`: load/store payload bytes use the 1024 B/cycle raw bank-wire
   capacity instead.

The last two deliberately distinguish useful payload from physical bank wire;
neither value is claimed as an MLX disclosure. Every schema also reconstructs
the unified data-supply stack as load + store + xfer, matching the paper's
grouping. No schema may change by resource after target exposure.

## Held-out rules

- Each schema emits 64 component points and 16 data-supply-stack totals.
- Primary trend completion requires Spearman rho >= 0.70 and matching nonzero
  endpoint direction for all eight operator/resource curves.
- Strict completion requires all 64 component points within 10% and is kept
  separate.
- Stack totals are diagnostic but mandatory; they cannot replace failures in
  the labeled component curves.
- No scale, offset, normalization fit, residual or pointwise schema selection
  is allowed.

## Acceptance gates

1. H166 and H60 pass byte/hash and semantic qualification.
2. H166 remains target-free with no selected metric or schema.
3. Exactly five schemas each cover 64 unique component cells and 16 stack
   totals.
4. Every prediction is an exact H166 metric or direct capacity formula frozen
   above; no target enters prediction arithmetic.
5. Every value/error/rank statistic is finite and predictions are in `[0,1]`.
6. Every schema has eight complete ordered-curve audits and all operator,
   resource and stack summaries.
7. At least one single schema passes all 8/8 trend curves; otherwise reject.
8. Strict 64/64 results are reported independently for every schema.
9. Runtime and source checks reject schema mixing, fits and transformations.
10. The result claims only held-out Figure-22 schema transfer; failure sends
    the work to schedule/workload reconstruction, not another denominator.

The immutable result will be
`artifacts/results/fig22-resource-schema-transfer-run172.json`.
