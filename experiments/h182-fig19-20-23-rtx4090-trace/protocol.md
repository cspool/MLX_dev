# H182 protocol: target-free RTX4090 trace for Figures 19/20/23

## Hypothesis

A shape-matched local RTX4090 service trace can identify launch, dense-TCU,
CUDA-core butterfly/FFT and scale-regime features missing from the current
Figure19/20/23 mappings. The trace is a workload/timing feature source for
later shared-parameter simulator corrections; it is not itself a paper result.

## Workloads

- Figure19: N=128/256/512/1024, D=1024, FFN=4096, with 2D-FFT and both
  block-butterfly FFN directions.
- Figure20: N=256/8192, D=4096, FFN=11008, with dense FP16 TCU and structured
  FP32 CUDA-core proxies for QKV, Attention, FFN1 and FFN2.
- Figure23: N=512/1K/2K/4K/8K, batch=8 and D=512, with FFT-CMP and BSMM
  service traces.

Every case records all CUDA-event samples, quartiles, operation/shape metadata,
finite-output checks and GPU snapshots. Dense matmuls/FlashAttention use FP16;
structured butterfly and FFT proxies use FP32 with TF32 disabled so their
service class remains distinct from dense Tensor-Core execution.

## Acceptance gates

1. All four frozen identity/device parents pass byte/hash/status/integrity.
2. GPU0 is the registered RTX4090/SM89 UUID before and after the run.
3. Exactly 12 Figure19, 16 Figure20 and 10 Figure23 cases are measured.
4. Every case has the registered 7 or 12 timing samples.
5. Every sample and p25/median/p75 value is positive and finite.
6. Every output is finite and has a finite checksum.
7. Dense and structured precision classes match the protocol.
8. All configured sequence lengths and component names are covered exactly.
9. Runner, auditor and test files are qualified and replayable.
10. Neither runner nor result reads or contains any Figure19/20/23 paper target.

The immutable result will be
`artifacts/results/fig19-20-23-rtx4090-trace-run187.json`.
