# H26 protocol: complete Figure 3 H100-profile targets

## Question

Can frozen source pixels recover all eight roofline markers and both five-bar
series in Figure 3, including the previously omitted QKV-plus-attention FLOP
shares, while preserving the paper's printed roofline constraints?

## Status before execution

Exploratory raster target recovery. Figure 3 and its existing coarse target
were already inspected in H8/H18, and the source was inspected again to locate
the missing orange series before this protocol. H26 is therefore
validation-ineligible and cannot establish native H100 execution.

## Frozen source and axes

- Source SHA-256:
  `86bd9284c3cd6905f37d561adbc61a87a62800fd7eae3a93c5eed3db26637350`.
- Image dimensions: 647x310 pixels.
- Roofline x-axis decade anchors are `(10^1,x=71)`, `(10^2,x=234)`,
  `(10^3,x=397)`: 163 pixels per OI decade.
- Roofline y-axis anchors are `(10^4,y=250)`, `(10^5,y=157)`, and
  `(10^6,y=65)`. The endpoint scale is 92.5 pixels per performance decade;
  the two raster intervals may differ by one pixel.
- CUDA-utilization anchors are `(0,y=276)`, `(0.05,y=224)`,
  `(0.10,y=172)`, `(0.15,y=120)`, `(0.20,y=68)`: 1040 pixels per unit.
- QKV-plus-attention FLOP-share anchors are `(0%,y=276)` through
  `(60%,y=42)` in 10-point/39-pixel intervals: 3.9 pixels per percentage
  point.
- All marker centers and bar tops are frozen in
  `artifacts/targets/fig3_full_digitization_pixels.yaml` before implementing
  the derivation runner.

## Frozen derivation

- For a marker at `(x,y)`, derive OI as
  `10 ** (1 + (x-71)/163)` FLOP/byte and performance as
  `10 ** (4 + (250-y)/92.5)` GFLOP/s.
- Derive CUDA utilization as `(276-y)/1040` and orange-bar FLOP share as
  `(276-y)/3.9` percent.
- Preserve marker identity from shape/color and the plotted 512/8K legend.
  Bar identity is fixed by blue-left/orange-right position within each length.
- Retain the printed roofline values: 2.0 TB/s bandwidth, 1513 TFLOP/s Tensor
  peak, and 102 TFLOP/s CUDA peak. Circles use the Tensor roofline; FFT/BSMM
  triangles use the CUDA roofline.

No marker, axis, series order, or endpoint may change after seeing the formal
output.

## Acceptance gate

- Source hash/dimensions and all four axis-spacing checks pass.
- Exactly 8 markers and 10 bars yield 26 finite numeric values inside plotted
  ranges.
- Derived marker OI/performance agree with H8's coarse values within 8%
  relative error; CUDA utilization agrees within 0.006 absolute. These are
  mistake-detection checks, not fits.
- Every derived marker is positive and at or below its applicable printed
  roofline with the existing 2% raster allowance.
- The five new orange values are within the plotted 0%-60% range.

H26 is supported only if every gate passes. Support means complete immutable
target acquisition; `native_profile_reproduced` remains false.

## Failure policy

Do not move points or substitute the H8 coarse values. A failure remains a
documented raster/series ambiguity. Native CUDA measurements must be compared
against, not used to revise, the frozen target.
