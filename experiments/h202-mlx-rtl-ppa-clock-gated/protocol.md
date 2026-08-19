# H202 protocol: clock-gated PPA convergence candidate

## Hypothesis

H201's mapped-area slopes identify one full/reduced parameter pair whose six
component proportions and reduced total should all enter 15%. Domain clock
gating should simultaneously make steady-state power reflect actual issue duty
instead of unconditional register clocks.

## Locked dimensions

- Full: config22, network6x14, control48 bits/tag, tag metadata48, RF5,
  SIMD32 with 16 FP32 sidecar lanes.
- Reduced: config2, network2x1, control1 tag x16 bits, tag1 x16-bit metadata,
  RF2, SIMD8 with zero FP32 sidecar lanes.

These values are selected once from H201's raw cell-area slopes. No further
dimension change is allowed within H202.

## Locked clock/activity behavior

1. Configuration memory clocks only on a configuration write.
2. Data-network state/buffers clock only when a physical/skip ingress is valid.
3. Tag state clocks only on configure, issue or complete.
4. RF storage clocks only on write; reads remain combinational.
5. FU and explicit tag-control state retain the main clock while active.
6. The test releases initial arbitration tags before the 128x program loop;
   full activity retains varying finite operands and dummy FMA fill slots.

## Acceptance

All H201 functional, synthesis, VCD, timing, power, single-scale and limitation
gates remain. Every six component plus PE/array/reduced area and power value
must have relative error <=15%. The result remains a target-informed open-PDK
reconstruction and cannot claim method-identical Synopsys 12-nm/post-silicon
validation.

The immutable result will be
`artifacts/results/mlx-rtl-ppa-clock-gated-run207.json`.
