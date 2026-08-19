# Paper-aligned MLX versus Xavier end-to-end goal

## Required result

Complete one end-to-end MLX and one major paper baseline, Jetson Xavier, with:

- actual simulator execution of a complete functional operator chain on both;
- five 32-layer Llama2-7B-surrogate performance estimates for
  N=128/256/512/1024/2048;
- the same decreasing speedup phenomenon and conclusion as Figure 21;
- estimates as close as practical to the digitized paper values;
- explicit separation of executed evidence, inferred parameters and paper
  target calibration.

## Claim boundary

Xavier uses a resource-edited SM70 timing proxy for SM72 and MLX uses a compact
functional workload plus full-shape service extrapolation. Exact paper software,
silicon and absolute numbers are not claimed. N=1024/2048 Xavier timing is
projected beyond its 16-GB capacity. H174 openly consumes Figure-21 speedups to
infer three global parameters and is not independent validation.

No complete-paper, RTL, area, power or <=10%-without-calibration claim is
required.
