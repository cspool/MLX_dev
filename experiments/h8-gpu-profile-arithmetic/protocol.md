# H8 protocol: Fig. 2/3 GPU profile arithmetic

## Classification

Digitized-profile consistency audit. It does not reproduce native Orin/H100 execution because this host has no NVIDIA GPU. All raster targets and axis mappings are frozen before the runner is implemented.

## Hypothesis

The Fig. 2 execution-time stacks reproduce the annotated Orin speedups within 10%, and every digitized Fig. 3 H100 point lies at or below its stated CUDA/Tensor roofline with a physically valid utilization in (0, 1].

## Frozen interpretation

- Fig. 2 execution groups are `N8K dense`, `N8K fft`, `N512 dense`, `N512 fft`; projection and attention are stacked.
- Speedup is dense total divided by FFT total at the same sequence length.
- The paired cache bars are L2 then L1 hit rate for each group.
- Fig. 3 uses the plotted 2.0-TB/s bandwidth, 1513-TFLOP/s Tensor peak, and 102-TFLOP/s CUDA peak.
- Softmax/QKV circles use the Tensor roofline; FFT and BSMM triangles use the CUDA roofline.
- Achieved roofline utilization is `performance / min(peak, OI * bandwidth)`.

## Pass criteria

- Both derived Fig. 2 speedups differ from their annotations by <=10%.
- Every Fig. 3 utilization is >0 and <=1 (with 2% allowance for raster error).
- Recomputed `to-qkv` Tensor efficiencies match the stored digitized efficiencies within 10%.
