# Corrective paper-static replay of Figures 22 and 23

H59 replays the existing Figure 22/23 workload mappings after changing only the
PE dependency model from the historical inferred scoreboard to the
paper-derived `paper_static` contract. Work counts, tagged blocks, event edges,
trip counts, routing, memory timing, FU timing, active windows, and hardware
scaling remain unchanged. The exact-transform audit passes for all 36 configs.

The corrected architecture is not a GPU SM embedded in each PE. The paper
supports a mesh of programmable spatial PEs with hop-encoded xfers, static
ordered tagged blocks, loop/bookkeeping state, heterogeneous FUs, and decoupled
load/store, compute, and transfer pipelines. Hardware elasticity is at the
tag/event boundary across folded layers. Warp/SIMT/CTA behavior and fine-grained
GPU scoreboarding are neither described nor imported into MLX timing.

## Result

| Replay | Points within 10% | MAPE | Maximum error | Status |
|---|---:|---:|---:|---|
| Fig. 22 compute utilization | 13/16 | 5.79% | 12.26% | rejected |
| Fig. 23 scalability proxy | 12/15 | 5.45% | 11.88% | rejected |

The six failing points are:

| Figure | Point | Simulated | Paper target | Relative error |
|---|---|---:|---:|---:|
| 22 | BSMM-8192 | 0.7534 | 0.85 | 11.36% |
| 22 | FFT-64 | 0.9430 | 0.84 | 12.26% |
| 22 | FFT-1024 | 0.9874 | 0.89 | 10.95% |
| 23 | SIMD32, 8x8, N=2048 | 13.2756x | 14.9x | 10.90% |
| 23 | SIMD8, 8x8, N=1024 | 3.3282x | 3.70x | 10.05% |
| 23 | SIMD8, 8x8, N=2048 | 3.3308x | 3.78x | 11.88% |

All 16 dsa-gem5 runs complete with exact instruction, event, and scratchpad
request/response counts, pass the RISC-V guest check, and emit no experimental
register-scoreboard or RF-bank/port stalls. All 20 Figure 23 runs replay
byte-identically. The combined hypothesis is nevertheless rejected because
both figures miss their all-points 10% gates.

The Figure 22 `ss_wait` watchdog remains 100,000 cycles by default. H59 opts
into 10,000,000 cycles because the 8192-point overlays require about 161k
overlay cycles. Progress is fed only on instruction issue/completion, route
hops, or memory completions. This host diagnostic change does not affect the
160,944 BSMM-8192 or 161,266 FFT-8192 reported overlay cycles.

## Replay

```bash
PYTHONPATH=src .venv/bin/python scripts/compile_paper_static_fig22_23.py \
  --output-dir artifacts/environment/h59

MLX_FIG22_CONFIG_ROOT=$PWD/artifacts/environment/h59/fig22 \
MLX_FIG22_OUTPUT_ROOT=$PWD/artifacts/environment/h59/runs/fig22 \
MLX_WAIT_BINARY=$PWD/third_party/dsa-framework/dsa-apps/sdk/compiled/ss-vecadd-gnu.out \
MLX_WATCHDOG_CYCLES=10000000 scripts/run_dsagen_fig22.sh

.venv/bin/python scripts/run_dsagen_fig23.py --experiment-id H59 \
  --config-root artifacts/environment/h59/fig23 \
  --output-dir artifacts/environment/h59/fig23/runs

.venv/bin/python scripts/audit_paper_static_fig22_23.py --verify-existing
```

The immutable result is
`artifacts/results/paper-static-fig22-23-run064.json`. This is a target-exposed
corrective replay, not author-source validation: the exact MLX simulator,
compiler mapping, hardware counters, and raw traces remain unpublished.
