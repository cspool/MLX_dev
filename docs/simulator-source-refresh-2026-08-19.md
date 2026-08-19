# MLX simulator source refresh — handoff

## Outcome

The open-simulator/source-refresh phase is complete and paused. It did not find
public MLX, SimICT, DFU-E or M2-DFU source. The current implementation should
remain a transparent hybrid surrogate:

- **MLX spatial execution:** DSAGEN/dsa-gem5 plus the project MLX overlay.
- **Counter/timing references:** DAM-RS, NPUsim and STONNE.
- **Volta/Ampere GPU proxies:** Accel-Sim/GPGPU-Sim, explicitly non-native
  until target-device tuning/traces exist.
- **Hopper/H100 candidate:** FlashGPU-Sim pin `f3d4bba`, not yet integrated or
  validated on the MLX H100 workloads.

The author/team lineage and source links are in
[`mlx-open-simulator-refresh-2026-08-19.md`](../literature/mlx-open-simulator-refresh-2026-08-19.md).
The GPU mapping is in
[`mlx-gpu-baseline-mapping-2026-08-19.md`](../literature/mlx-gpu-baseline-mapping-2026-08-19.md).

## What was tested

| Run | Test | Result |
|---|---|---|
| H163/run168 | Seven utilization identities, FU classes and four SRAM ports, target-free | supported, 10/10; no metric selected |
| H164/run169 | All seven identities exposed unchanged to Figure 22 | rejected; every identity 0/8 trend curves, no strict-complete identity |
| H165/run170 | Active windows 1–8; execute legal windows 1–5 over 16 paths | rejected; window5 gives 0.977x–1.102x, only 14/16 non-regressions |
| H166/run171 | External/local loads, stores, xfer issues/hops, DMA/SRAM capacity ledger | supported, 10/10; no schema selected |
| H167/run172 | Five complete resource-domain schemas exposed to Figure 22 | rejected; best MAPE 27.83%, best 16/64 strict, but all 0/8 trend curves |
| H168/run173 | Xavier/Orin/RTX3090/H100 open-simulator mapping | supported; 4/4 candidates, 3/4 local proxies, 0/4 validation-eligible |
| H169/run174 | Same-team patent SPM fusion-capacity contract | supported; N128–1024 match one kernel, N2048 requires an unimplemented two-kernel split |

## Source-derived parameters and confidence

| Parameter | Current value | Evidence class |
|---|---:|---|
| MLX mesh / instruction store | 4x4 / 32 instructions per PE | MLX paper |
| Legal common active window | 1–5; window6–8 exceed 32 slots on at least one exact path | static simulator derivation |
| Current active window | 3 | earlier Figure-9 phase inference; retained because uniform window5 regresses two paths |
| External DMA | 64 B/cycle = 64 GB/s at 1 GHz | historical DPU lineage, not disclosed for MLX |
| SRAM topology | 4 ports, 32 total banks | Figure-9/11 and H69 derivation |
| SRAM raw wire capacity | 1024 B/cycle | 32 banks x 32 B |
| SRAM SIMD8 payload capacity | 512 B/cycle | 32 issues x 16 B payload |
| Attention fusion threshold | `N*D*2 <= 8 MiB` for FP16 | same-team CN119940434B precedent, not confirmed MLX code |

The component-service counter schema materially fixes Figure-22 magnitude but
not its size ordering. Further denominator or bandwidth fitting is stopped.

## GPU baseline status

| Device | Local path | Status |
|---|---|---|
| Xavier SM72 | TitanV SM70 timing with resource edits | functional sparse proxy; dense/end-to-end native evidence missing |
| Orin SM87 | RTX3070 SM86 timing with resource edits | functional proxy; equal work changes 6.149% with CTA shape |
| RTX3090 SM86 | RTX3070 SM86 timing with resource edits | closest ISA-family proxy; native cache/launch tuning and SASS missing |
| H100 SM90 | no local execution; FlashGPU-Sim candidate | requires exact eager/FA2/FFT-CMP port plus native H100 correlation |

Accel-Sim's documented tuner requires microbenchmarks from the target device,
and strict application validation requires matching SASS traces. Resource-only
edits must remain labeled proxies.

## Concrete remaining simulator gap

H93/H94 already fuse Attention into one graph for every Figure-21 shape. The
FP16 N-by-4096 footprint is 1/2/4/8/16 MiB for N=128/256/512/1024/2048.
N=2048 exceeds the 8-MiB SPM but is still modeled as one kernel. Correct timing
is intentionally unavailable until these four fields are sourced or defined by
a new independent protocol:

1. second-kernel boundary;
2. streaming tile shape;
3. intermediate traffic domain;
4. second-kernel launch cycles.

## Resume conditions

Resume this phase only when at least one of the following is available:

- native Xavier/Orin/RTX3090/H100 tuner output and matching application traces;
- author/source evidence for the N=2048 two-kernel split;
- an independent simulator mechanism that changes workload scheduling rather
  than another Figure-22 counter denominator.

Do not use Figure-22/21 residuals to choose those fields.

## Verification

At phase close: Ruff passed; pytest passed `431`, failed `0`, with 17 existing
environment/deprecation warnings.
