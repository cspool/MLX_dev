# MLX versus one main baseline: completion goal

## Scope

Complete one representative performance and functional experiment rather than
the full paper. Compare MLX with one main baseline on the identical complete
structured Transformer block.

The main baseline is **single-layer serial execution on the same programmable
spatial array**. It admits one logical layer/tag at a time. MLX uses the same
hardware and program but admits multiple tags and unlocks a consumer when its
exact producer store completes.

This baseline is selected to isolate MLX's central architecture innovation. It
is not labeled Xavier, Orin or an exact implementation from another paper.

## Required functional evidence

- Both baseline and MLX execute the same original inputs and dynamically linked
  `BSMM -> FFT-CMP -> Attention -> causal SWA -> elementwise` chain.
- Both match an independent from-origin golden at every component boundary and
  at all final outputs within `1e-12`.
- The complete chain covers BSMM, FFT-CMP, Attention, SWA, elementwise and the
  complete Transformer block.
- Instruction, operation, memory, event and route work are identical.
- Functional-enabled and timing-only runs have identical timing.

## Required performance evidence

- Use baseline cycles divided by MLX cycles.
- Require at least `1.20x` clear improvement on the complete block.
- Require MLX to be no slower on all cumulative depth points.
- Require repeat/build identity and clean sanitizer execution.
- Attribute the gain to multiple active tags and data-ready issue before global
  producer-tag completion, not less work.

## Exclusions

- No requirement to reproduce every MLX paper experiment or every GPU.
- No requirement for exact paper numbers or <=10% error.
- No RTL, area or power experiment.
- No paper-target fitting, residual scale or invented external baseline label.

Completion requires one immutable certificate plus a fresh full-repository
Ruff/pytest verification.
