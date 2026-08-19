# H192 protocol: complete shape and multi-layer coverage

## Hypothesis

The unified frontend/lowering path can cover every Figure23/19/20 target shape
and produce replayable Llama2/FABNet multi-layer compositions through one CLI,
without invoking figure-specific compiler commands manually.

## Coverage

- Figure23: 5 sequence lengths x 2 active windows x 4 hardware mappings = 40
  physical overlay units.
- Figure19: 4 sequence lengths x FFT2D/FFN1/FFN2 = 12 DPU-memory units.
- Figure20: 2 sequence lengths x QKV/Attention/FFN1/FFN2 = 8 analytical units.
- Composition: one Llama2 32-layer (24 structured + 8 dense) plan and one
  FABNet 24-layer plan.

## Acceptance gates

1. All eight frozen inputs qualify and required parents retain status/integrity.
2. One coverage spec validates three graphs/fourteen operator nodes and two
   composition plans.
3. The single lowering CLI emits exactly 40/12/8/2 units by category.
4. All required sequence lengths, windows, mappings and operators are covered.
5. Every detailed overlay/memory/profile artifact passes its native schema.
6. Llama2 and FABNet composition plans cover 32 and 24 layers respectively and
   conserve referenced per-layer work.
7. All 62 units execute twice (124 executions) and complete successfully.
8. All 62 execution summaries replay identically.
9. H189 same-input equivalence remains supported and every execution unit has
   graph/shape/operator lineage.
10. Source/manifests/tests qualify and no figure-specific manual command appears
    between the suite spec and execution manifest.

The immutable result will be
`artifacts/results/full-workload-coverage-run197.json`.
