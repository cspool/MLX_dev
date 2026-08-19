# H170 protocol: one main baseline, complete function and performance

## Objective

Close the user's narrowed goal with one baseline rather than a collection of
unrelated comparisons. Execute the same dynamically linked structured
Transformer block on:

- **main baseline:** one active logical layer/tag at a time on the same
  programmable spatial array;
- **MLX:** up to thirteen active tags, enabling cross-layer overlap.

The only architectural difference is the active-layer window. Mesh, PEs,
functional units, register file, four decoupled pipelines, routing, memory,
instructions, values and work remain identical. This baseline isolates MLX's
central multi-layer execution mechanism; it is not presented as Xavier or as
an exact external accelerator reproduction.

## Workload coverage

Reuse H161's one-execution chain:

`hierarchical BSMM -> FFT-CMP -> Attention -> causal SWA -> residual/scale/SiLU`.

Four cumulative prefixes provide a depth curve while the final prefix is the
complete block. Each prefix uses the same original seeds and dynamically linked
intermediate addresses. Both architectures run functional-enabled and
timing-only documents under debug, optimized and ASan/UBSan builds.

## Functional contract

For every prefix and architecture:

- recompute the golden independently from the original BSMM input;
- verify every exposed component boundary, not only the final tensor;
- require maximum absolute error <=1e-12, finite values and exact operation
  completion;
- require baseline and MLX outputs to be identical;
- require functional-enabled and disabled timing/statistics to be identical.

The complete prefix must retain H161's six payload claims: BSMM, FFT-CMP,
Attention, SWA, elementwise and complete Transformer block.

## Performance contract

- Compute speedup as baseline optimized cycles divided by MLX optimized cycles.
- Require MLX to be no slower on every cumulative prefix.
- Require the complete-block speedup to be at least 1.20x.
- Preserve exact instruction, pipeline, operation, memory, event and route work
  between the pair.
- Report active-tag occupancy and pipeline busy/issue evidence so the gain is
  attributable to multi-layer overlap rather than less work.

No paper target, residual factor, launch correction or fitted latency is read.

## Acceptance gates

1. H161 result/config/compiler/auditor pass byte/hash and semantic checks.
2. Exactly 16 deterministic configs cover four prefixes, two architectures and
   enabled/disabled functional modes.
3. Paired documents differ only in active-window identity/metadata.
4. Same-input seed memory and same-work instruction/event/route contracts hold
   for every pair.
5. Exactly 48 executions complete across all three builds with clean sanitizer
   output and build-identical summaries/traces.
6. Both architectures match every prefix golden and each other within 1e-12.
7. Enabled/disabled timing is identical within every architecture/prefix.
8. MLX is no slower on all four prefixes and reaches >=1.20x on the complete
   block.
9. Baseline admits exactly one active tag; MLX demonstrates more than one on at
   least the complete block with unchanged completed work.
10. The result claims a complete same-input MLX-versus-one-baseline functional
    and performance experiment only; it consumes no paper performance target.

The immutable result will be
`artifacts/results/single-baseline-complete-block-run175.json`.
