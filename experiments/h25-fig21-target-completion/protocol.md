# H25 protocol: complete Figure 21 targets and recover GEMM-time shares

## Question

Can frozen source pixels recover all 20 numeric Figure 21 bars—five Xavier
speedups, five GEMM-time shares, and five dense plus five sparse memory
values—and reconcile the 16-GB capacity/hatching semantics with the prose?

## Status before execution

Exploratory raster target recovery. The source plot was visually inspected and
its axes and bar regions were located before registration, so this run cannot
validate either Xavier or MLX execution. Its purpose is to replace the partial,
coarsely digitized Figure 21 target with an immutable full-plot target for a
future implementation.

## Frozen source and coordinates

- Source image SHA-256:
  `ec3186cf804a841d495cf95979632e2d9fbe718f91aed49f0e8b439d5d54e2aa`.
- Image dimensions are 632x242 pixels.
- Panel (a)'s speedup axis uses `y=199,158,117,76,35` for values
  `0,1,2,3,4`, exactly 41 pixels per speedup unit.
- Panel (a)'s adjacent grayscale colorbar uses the same `y=199..35` span for
  0%-80% GEMM time. Its interior is frozen at `x=209..214`, `y=35..198`.
- Panel (b)'s memory axis uses `y=198,164,130,95,61,27` for
  `0,5,10,15,20,25` GB. Its five intervals differ by at most one pixel; the
  registered scale is `(198-27)/25 = 6.84` pixels per GB.
- All speedup endpoints, memory endpoints, and GEMM fill regions are frozen in
  `artifacts/targets/fig21_full_digitization_pixels.yaml` before implementing
  the derivation runner.

## Frozen GEMM color inversion

GEMM time is encoded by each speedup bar's grayscale, not by its height. For
each registered bar ROI:

1. Convert the source to 8-bit luminance and retain pixels with value at least
   80, excluding black outlines and diagonal overflow hatching.
2. Take the median retained luminance.
3. Take the median luminance across the six colorbar pixels at each row.
4. Apply unweighted pool-adjacent-violators regression to make that row curve
   nondecreasing from black to white.
5. Select the row whose fitted luminance is nearest the bar median (average row
   on an exact tie) and convert it with `(199-y)/2.05` percent.

No post-hoc palette, ROI, threshold, smoothing weight, or sequence-specific
correction is allowed. GEMM-time uncertainty is conservatively +/-2 percentage
points to include JPEG and hatch effects.

## Acceptance gate

- The source hash/dimensions and both axis-spacing checks pass.
- Exactly 20 bar values are recovered. Every value is finite and within its
  plotted range.
- The five speedups agree with the pre-existing coarse target within 0.08x;
  dense and sparse memory agree within 0.35 GB. These checks detect series or
  axis mistakes; the frozen-pixel values supersede the coarse values after a
  passing run.
- The capacity-line center derives to 16 GB within 0.2 GB.
- Dense memory exceeds the capacity at exactly `[512, 1024, 2048]`, matching
  the three hatched/projected bars and the paper's stated Xavier boundary.

H25 is supported only if every gate passes. Support means complete target
recovery, not native hardware reproduction.

## Failure policy

Do not move endpoints, change the luminance threshold, or select another
colorbar inversion from the output. A failure remains an explicit target-source
ambiguity. Native timing or memory results must never be used to alter these
pixels.
