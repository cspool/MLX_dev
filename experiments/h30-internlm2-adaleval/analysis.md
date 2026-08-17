# H30 result: the historical public Ada-LEval baseline is rejected

The byte-qualified historical InternLM2-Chat-7B reconstruction obtains 57.8%,
46.9%, and 27.4% accuracy on the exact 1k/2k/4k Ada-LEval BestAnswer sets.
Only the 1k result is within 10% relative error of the corresponding MLX
Fig. 15(c) original bar. The registered all-setting gate therefore rejects H30.

| Setting | Correct | Measured | Std. error | MLX target | MLX relative error | MLX gate | Ada-LEval target | Ada relative error | Ada gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1k | 578/1,000 | 57.8% | 1.562 pp | 52.8% | 9.47% | pass | 58.6% | 1.37% | pass |
| 2k | 469/1,000 | 46.9% | 1.578 pp | 40.6% | 15.52% | fail | 49.5% | 5.25% | pass |
| 4k | 274/1,000 | 27.4% | 1.410 pp | 35.9% | 23.68% | fail | 33.9% | 19.17% | fail |

The official aggregate report is
`artifacts/results/internlm2-adaleval-run034.json` (24,158 bytes, SHA-256
`816731c7c3f6465b1e8a1de0f97661d7e776bc4b2ee9afa0e5099a789afc67c3`).
Its source/runtime commit is
`58575d16311cb4c8bcb4cc2d8a34703f5c8c5f52`. The two complete sample logs are:

| Rank | Records | Bytes | SHA-256 |
|---|---:|---:|---|
| 0 | 1,500 | 630,717 | `3cf389864571c664653fbc1053148c75d74d3233a64e28c60e3cc49c42951770` |
| 1 | 1,500 | 631,352 | `108782168a75e1899ef6d9977010d3c2bbd42464efe16aedd030863785e2b52b` |

## Qualification and independent checks

Every non-target gate passes: the MLX target manifest, official Ada-LEval
revision and source files, all three dataset bytes, historical checkpoint view,
LMDeploy 0.2.6 source hashes, dependency/GPU runtime, all prompt/token canaries,
and the effective-capacity proof. Both rank logs contain exactly 1,500 records;
all 3,000 setting/position pairs are unique and obey the registered even/odd
partition. Reapplying the extractor and correctness rule reproduces every
record, and each aggregate equals the arithmetic sample mean.

All 3,000 prompts have distinct hashes and all 3,000 recorded 64-bit seeds are
unique. There are no unextracted answers. Every request finishes normally with
`stop`; generated lengths are 5-5, 5-18, and 5-81 tokens for 1k/2k/4k. Input
lengths are exactly 488-1,239, 1,339-2,311, and 3,455-4,451 tokens, matching the
pre-inference canaries.

The hardware accommodation cannot explain a length failure: the largest input
plus the full 512-token generation allowance is 4,963, below the frozen 8,192
effective session and prefill limits. The observed maximum generation is only
81 tokens. No request approaches the cap or receives a length finish.

The run takes 661.86 seconds wall time on two RTX 4090 GPUs, with 1,088.20
summed sample-seconds and 16,090 generated tokens.

## Interpretation and scope

The 1k/2k agreement with the official Ada-LEval table supports most of the
historical checkpoint, prompt, extractor, and LMDeploy reconstruction, but the
4k result remains 6.5 percentage points below that independent table. Relative
to MLX, the measured curve is also shaped differently: it is 5.0 and 6.3
points higher at 1k/2k but 8.5 points lower at 4k. A single checkpoint-wide
offset cannot reconcile the three targets.

LMDeploy 0.2.6 uses stochastic top-k/top-p decoding and the official table does
not publish per-request seeds, so H30 cannot distinguish one-run sampling
variation from an unpublished checkpoint/evaluation detail. It does establish
that this one fully recorded public reconstruction fails both registered
all-setting gates. The paper additionally omits its exact checkpoint revision,
prompt/evaluator revision, seeds, and whether the dense `original` model was
task-adapted.

No checkpoint, prompt, chat template, subset, session cap, seed, decoder, or
extractor was changed after observing the result, and no failed setting was
rerun. H30 evaluates only the dense public checkpoint; it supplies no evidence
for the compressed `s=0.5` bars or the MLX training recipe.
