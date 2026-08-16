# H6 protocol: cross-device Xavier holdout

## Classification

Confirmatory cross-device test. Figures 3 and Table IV provide calibration inputs; every point in Figures 20 and 21 is held out. No Fig. 20/21 bar may be used to choose a latency, efficiency, power, memory, or layer-coverage parameter.

## Hypothesis

A hardware-spec roofline model transferred from the paper's H100 utilization profile, combined with the pre-registered MLX event model, predicts the Xavier kernel speedup/energy ratios and end-to-end speedup within 10%. A first-principles Llama2-7B parameter/KV-cache model predicts the Fig. 21 memory bars within 10%.

## Frozen inputs

- MLX: `mlx_full.yaml` and `paper_v1.yaml`, unchanged from H2; 1 TOp/s, 5.8464 W core plus 0.6 W memory.
- Xavier: 1.7-TFLOP/s FP16 CUDA peak and 6-TFLOP/s Tensor Core peak from paper Table IV; 15 W from the same table.
- Xavier memory: 16 GB, 256-bit LPDDR4x, 137 GB/s from NVIDIA's official Jetson AGX Xavier material.
- Dense Tensor-Core roofline efficiency: log-sequence interpolation between the two digitized Fig. 3 `to-qkv` points (0.325 at N=512 and 0.509 at N=8192 after division by the plotted 1513-TFLOP/s H100 Tensor peak). Clamp outside this interval.
- Structured CUDA roofline efficiency: log-sequence interpolation over the five digitized Fig. 3 CUDA-utilization bars `[0.118, 0.136, 0.128, 0.188, 0.151]` at N=`[512, 1K, 2K, 4K, 8K]`. Clamp outside this interval.
- Llama2-7B shape: 32 layers, D=4096, FFN dimension 11008, 32 attention/KV heads, head dimension 128, vocabulary 32000, FP16, batch 8 for the end-to-end/capacity model.
- Structured setting: B=32, s=0.5, and 24/32 modified layers. The latter is the nearest integer layer count consistent with the paper's `>60%` staged application and the four-way layer progression visible in Fig. 15; it is fixed before opening Fig. 21 residuals.

Official baseline sources:

- <https://developer.nvidia.com/blog/nvidia-jetson-agx-xavier-32-teraops-ai-robotics/>
- <https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-xavier-series/>

## Kernel manifest

At N=256 and 8192:

- QKV: three D-to-D dense GEMMs on Xavier; three B=32 hierarchical BSMM projections on MLX/sparse CUDA.
- Attention: dense QK/softmax/SV on Xavier; three chunk FFT-compressions followed by attention at sN on MLX/sparse CUDA.
- FFN1: D-to-11008 dense GEMM versus B=32 rectangular BSMM.
- FFN2: 11008-to-D dense GEMM versus B=32 rectangular BSMM.

GPU time is `operations / (efficiency * min(peak, OI * bandwidth))`, evaluated per kernel component. MLX time and activity-weighted power come directly from the event simulator. Xavier energy uses the paper's fixed 15-W baseline; this simplifying assumption is part of the test and may be rejected by the data.

## End-to-end and memory model

End-to-end time sums QKV, attention, output projection, FFN1, and FFN2 over 32 layers, selecting structured or dense execution according to the frozen 24/8 layer split. Elementwise operations are reported separately but omitted from the primary prediction because the paper does not give their instruction mix.

Memory uses no fitted coefficients:

- dense FP16 parameters from the declared Llama2 shapes;
- structured projection weights at density `2*log2(B)/B` in 24 layers;
- batch-8 FP16 KV cache for all 32 layers, shortened by s in modified layers;
- one live QKV activation buffer.

The 16-GB capacity check uses decimal GB, matching the plot label. Fig. 21 values beyond Xavier capacity remain predictions and are labeled projected, as in the paper's hatched region.

## Pass and failure criteria

- Point-wise absolute relative error <=10% for every speedup and memory anchor.
- Energy is audited separately because constant 15-W power may be falsified by per-kernel activity.
- The geometric mean is never a substitute for point-wise agreement.
- If the holdout fails, do not tune against Fig. 20/21. Reject H6 and require native Xavier measurements or an independently validated Accel-Sim/AccelWattch configuration.
