# H191 protocol: cycle-level physicalization

## Hypothesis

The calibrated latency effects can be expressed as simulator state and explicit
cycle intervals rather than summary-time arithmetic, while retaining all 68
Figure23/19/20 values within 15% and all 50 baseline directions.

## Implementation

- C++ overlay: replace `latency_service` with `physical_timing`. The overlay
  advances real scheduler state for pre-ROI warmup cycles, resets a measurement
  origin, and injects congestion stalls evenly during measured execution.
  Summary reports scheduler-progress, injected-stall, pre-ROI and measured
  cycles separately.
- Figure19: materialize each nonzero trace-launch, work and SPM-transition
  contribution from the separately frozen H185 composition as an integer cycle
  interval. Totals and speedups are derived only from interval sums.
- Figure20: materialize MLX and baseline launch/work/congestion timelines using
  H182 trace proportions and the separately frozen H186 composition;
  calibrated log services determine service rates, not a post-hoc reported
  ratio. Speedups derive from total baseline/MLX cycles.

## Acceptance gates

1. All eight frozen inputs qualify and required parents retain status/integrity.
2. The new physical-timing patch is reversible on top of the latency patch.
3. Forty Figure23 configs contain `physical_timing` and no enabled
   `latency_service`.
4. Eighty Figure23 executions replay identically and retain raw instruction,
   event, route and memory work.
5. Pre-ROI progress and injected stalls exactly explain every measured cycle.
6. Figure19 emits twelve timelines and Figure20 emits thirty-two timelines;
   every total equals the sum of positive integer phases.
7. All 30 Figure23, 20 Figure19 and 18 Figure20 values remain within 15%.
8. All fifty baseline-relative directions remain unchanged.
9. No result-side latency correction is applied after timeline/simulator output.
10. Sources qualify and target-informed/non-independent status remains explicit.

The immutable result will be
`artifacts/results/cycle-level-physicalization-run196.json`.
