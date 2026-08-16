# H7 protocol: reported-number consistency audit

## Classification

Exploratory cross-figure arithmetic audit. The visual relationship in Fig. 18 was inspected before this protocol, so H7 is not confirmatory evidence for the simulator. The formulas and 10% gate are frozen before implementing the runner.

## Hypothesis

The paper's reported area/power totals, resource ratios, speedup summaries, and algorithm-normalized speedups are mutually consistent to within 10% under their stated definitions.

## Checks

1. **Table II:** sum the six PE component rows and compare with the PE row; multiply the PE row by 16 and compare with the 4x4 PE-array row. Report reduced/full area and power ratios.
2. **Fig. 18(c):** compute hardware-software affinity as
   `latency_speedup_i / (algorithm_FLOP_saving_i / SpAtten_FLOP_saving)`
   using Fig. 18(a) and Table IV. Compare against every annotated Fig. 18(c) bar.
3. **Table V:** recompute Ours/FABNet LUT, FF, and DSP ratios from the integer resource counts.
4. **Fig. 19:** recompute min, max, and geometric mean from the four annotated end-to-end speedups and verify the prose range. Component ranges are replayed from the prose because the raster does not provide numeric labels for each stacked component.

## Pass criteria

Every derived-vs-reported point has absolute relative error <=10%. Failures remain reported as paper-internal inconsistencies and must not be repaired by changing a source value.
