# FFT coupled-counter steady state

## Outcome

H117 run122 is rejected with `audit_integrity=true`. The experiment extends
only the eight FFT-CMP paths: q=16/32 parent runs are frozen as fit anchors,
q=64/128 are newly compiled and executed, and no Figure 22 or Figure 25 target
is read.

All 16 new configurations compile exactly and all 48 executions finish: two
optimized replays per configuration plus 16 clean ASan/UBSan runs. Requests,
responses, bytes, tiles, instructions, ownership, source hashes and optimized
replays qualify.

## Holdout result

The q=16/32 affine fits predict 89 of 96 q=64/128 metric holdouts within 5%.
Overall MAPE is 1.22% and maximum error is 9.65%.

| Metric | Passing holdouts | Interpretation |
|---|---:|---|
| End-to-end cycles | 16/16 | steady |
| Productive compute PE-cycles | 16/16 | steady |
| Productive load PE-cycles | 16/16 | steady |
| Productive store PE-cycles | 16/16 | steady |
| Productive xfer PE-cycles | 16/16 | steady |
| Productive FMA PE-cycles | 9/16 | not steady under this affine model |

The seven failures are confined to FMA residence: BERT-512 q128 (5.54%),
InternLM2-4K q64/q128 (6.62%/9.65%), Llama2-1K q128 (7.24%), Llama2-4K
q64/q128 (6.62%/9.65%), and Llama2-512 q128 (5.54%). BERT-8K,
InternLM2-2K and InternLM2-8K pass all six metrics.

## Interpretation boundary

The rejection does not invalidate the coupled simulator's FFT cycle or
pipeline counters: all 80 cycle/compute/load/store/xfer holdouts pass. It
invalidates only the registered claim that every physical FMA-residence fold
has reached affine steady state.

FMA residence is a latency-weighted busy counter. Figure 25 instead normalizes
completed FMA work by a compute/bandwidth roof. H115 already shows that
completed-work issue misses the paper targets, and H116 shows that residence
is effectively identical to issue for QKV/SWA. Extending q again to make the
FFT residence fold pass would therefore not repair Figure 25 and is stopped.

The next admissible use of run122 is Figure 22: retain the now stable coupled
cycle and productive compute/load/store/xfer semantics, rebuild the exact 16
FFT/BSMM utilization workloads target-free, and freeze that compiler before a
new target comparison.

Evidence is in
[run122](../artifacts/results/fft-coupled-counter-steady-state-run122.json),
with the frozen plan in
[H117 protocol](../experiments/h117-fft-coupled-counter-steady-state/protocol.md).

Replay the immutable audit with:

```bash
PYTHONPATH=.:src .venv/bin/python \
  scripts/audit_fft_coupled_counter_steady_state.py --verify-existing
```
