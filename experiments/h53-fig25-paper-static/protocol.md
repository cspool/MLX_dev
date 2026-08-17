# H53 protocol: Figure 25 transfer under corrected PE semantics

## Classification

Target-exposed, validation-ineligible corrective transfer. H53 combines H50's
source-derived arithmetic expansion with H52's paper-static PE semantics. The
Figure 25 cells remain audit-only.

## Hypothesis

Changing only `pe_dependency_model` from the historical scoreboard behavior to
`paper_static` will reproduce all 24 MLX cells within 10%. No arithmetic,
traffic, stage, trip-count, FU, memory, or per-cell parameter may change.

## Frozen contract

`configs/simulators/dsagen_mlx_fig25_paper_static_v1.yaml` binds every H53
document to the corresponding H50 document. The compiler may add only the root
and metadata dependency-model declarations. All 24 detailed dsa-gem5 runs must
retain request/response conservation and contain no register/RF hazard stalls.
Targets are loaded only after immutable measurements exist. A failed surface is
preserved; no residual-driven correction is permitted.

## Immutable output

The sole formal output is
`artifacts/results/dsagen-mlx-fig25-paper-static-run059.json`.
