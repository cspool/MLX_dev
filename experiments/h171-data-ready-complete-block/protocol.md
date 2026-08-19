# H171 protocol: data-ready MLX versus one serial spatial baseline

## Hypothesis

H170's complete-block gain falls below 1.20x because H161 uses a coarse
whole-component predecessor at every dynamic link. Replacing those barriers
only in MLX with exact store-completion events will permit downstream blocks to
start as soon as their linked values exist, preserve full numerical correctness
and work, and raise the complete-block MLX gain over the same single-layer
baseline to at least 1.20x.

## Boundary implementation

Both architectures receive the same 18 new store-event definitions covering
all 24 dynamically linked values. A store event fires after the functional
memory update completes. Every downstream load block waits for the event that
produces its exact address; event multiplicity encodes the position within a
store's address sequence.

- The **baseline** retains all 21 previous whole-component predecessors, so it
  remains one-active-layer serial execution. Its event checks are redundant but
  make event work identical to MLX.
- **MLX** removes exactly those 21 coarse predecessors and relies on the new
  address-matched events. It retains all internal component dependencies and
  uses active window 13.

Instruction count, operations, input memory, load/store addresses, routes,
pipeline/FU timing and functional values are unchanged. Boundary-event work
increases equally from 73 to 97 in both architectures.

## Experiment

Repeat H170's four cumulative prefixes and enabled/disabled runs across debug,
optimized and sanitized builds: 16 configs and 48 executions. Recompute all
component goldens from the original BSMM input and verify every boundary for
both architectures.

## Acceptance gates

1. H170/H161 evidence and frozen sources qualify; H170's 1.167x negative result
   is retained.
2. Exactly 16 deterministic configs and 48 clean executions are produced.
3. All 24 linked addresses map to one of 18 store events and exact downstream
   load waits; emitted event work is 97 for both complete documents.
4. Baseline removes zero coarse predecessors; MLX removes exactly 21, with no
   internal dependency changed.
5. Paired instruction, operation, memory and route work and input seeds remain
   identical.
6. Both architectures match every cumulative independent golden within 1e-12
   and produce identical boundary values.
7. Functional-enabled and disabled timing/statistics are identical within each
   pair; all builds replay identically and sanitizers are clean.
8. MLX is no slower for every prefix and complete-block speedup is >=1.20x.
9. Baseline max active tags is one; MLX issues at least one data-ready block
   before its producer tag globally completes and has multiple active tags.
10. The result claims a complete functional/performance MLX-versus-one-baseline
    experiment only, with no paper target or fitted latency.

The immutable result will be
`artifacts/results/data-ready-complete-block-run176.json`.
