# H32 result: the 8,192-token accommodation is output-equivalent

H32 is supported exactly as registered. On all 32 fixed Ada-LEval 4k
prompt-length quantiles, the candidate initialized from
`session_len=160000` produces exactly the same raw prediction, extracted
answer, input-token count, generated-token count, and finish reason as H31's
`session_len=8192` baseline when every prompt, seed, decoder setting, model
byte, prefill setting, and physical GPU is held fixed.

| Exact field | Matches | Mismatches | Gate |
|---|---:|---:|---:|
| Raw prediction text | 32/32 | 0 | pass |
| Extracted designation | 32/32 | 0 | pass |
| Input-token length | 32/32 | 0 | pass |
| Generated-token length | 32/32 | 0 | pass |
| Finish reason | 32/32 | 0 | pass |

The result report is
`artifacts/results/internlm2-adaleval-session-equivalence-run036.json`
(44,511 bytes, SHA-256
`8e5eca2418ffad655b2dbcc11f734a0da9683fcab00760b0728ae6ce33b108af`).
The complete 32-record sample log is
`artifacts/results/internlm2-adaleval-session-equivalence-run036.samples.jsonl`
(14,000 bytes, SHA-256
`dcd0ffaeddf6a14c22e38a935e8589a8451641c9e4b665b3efc2bef022f76bca`).
The source/runtime commit recorded before inference is
`42ae86e0e39ff2e8000d8560c66d40f6628c485a`.

## Qualification and capacity

The formal run requalifies every H30 paper, official-repository, dataset,
historical-checkpoint, LMDeploy-source, dependency/GPU, prompt/token, and
capacity gate. It also requalifies the complete run035 report and rank-1 log.
The deterministic selection contains the frozen 32 positions, covers input
lengths 3,455-4,451, and reproduces the registered prompt/seed stream SHA-256
`fa769226...4c0d`. The candidate runs alone on physical RTX 4090 GPU 1, the
same physical device used by the H31 rank-1 baseline.

The candidate requests `session_len=160000`, rope factor 2.0, an 8,192-token
prefill, and the pre-registered 20% cache allocation. At initialization,
LMDeploy 0.2.6 reports that the available KV blocks truncate its serviceable
session capacity to 13,952 tokens. This exceeds both the registered
12,544-token lower bound and the largest selected input plus full 512-token
output allowance, 4,963 tokens. Thus H32 tests the effect relevant to these
requests; it does not claim that this 24-GB GPU can serve a 160,000-token
request.

Every candidate response finishes with `stop`. Actual input lengths span
3,455-4,451 and outputs span 5-15 tokens. Re-extraction and correctness replay
exactly; both baseline and candidate have 11 correct answers. The latter count
is descriptive only, because H32 is not a new accuracy trial. The worker takes
27.454 seconds after the preflight, including 16.893 summed sample-seconds and
180 generated tokens.

## Interpretation and stopping rule

For these exact seeds and full observed 4k prompt-length range, increasing the
effective service capacity above 8,192 does not change even one generated
character or token counter. H30/H31's persistent 4k residual therefore cannot
be attributed to their 8,192-token hardware accommodation.

The result does not validate MLX's 35.9% accuracy bar, reproduce Ada-LEval's
unpublished seeds, or prove equivalence for requests above 13,952 tokens. It
closes only the registered runtime caveat. No cache ratio was selected after
output, and no additional seed or accuracy replicate is permitted; resolving
the remaining gap now requires an independently identified checkpoint,
evaluator, prompt, or task-adaptation artifact.
