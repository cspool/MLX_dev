# H32 protocol: Ada-LEval session-cap equivalence

## Hypothesis and scope

For 32 fixed Ada-LEval 4k prompts spanning the full observed prompt-length
range, LMDeploy 0.2.6 with the official `session_len=160000` produces exactly
the same fixed-seed raw predictions as H31's `session_len=8192` baseline when
all model, prompt, decoder, prefill, GPU, and extraction settings are held
constant.

H32 is a runtime-equivalence audit, not another accuracy attempt. It is
validation-eligible only for H30's hardware accommodation. It cannot validate
the MLX accuracy bar, select seeds, or replace H30/H31.

## Qualified baseline and selection

Require the complete H30 config, H31 report, and H31 rank-1 sample log with the
byte counts and SHA-256 hashes frozen in the config. Rerun H30's complete
source/runtime/prompt/capacity preflight before loading the candidate model.

Use only H31 replicate 0, rank 1 records, which were generated on physical
RTX 4090 GPU 1. Sort the 500 odd-position records by
`(input_token_len, dataset_position)`. For `i=0,...,31`, select index
`floor(i*499/31)`. This rule depends only on pre-inference prompt lengths, not
predictions or correctness, and includes the global 4k minimum and maximum of
3,455 and 4,451 tokens.

Freeze all 32 positions and lengths in the config. Hash each selected record as
`position:uint32_be || input_len:uint32_be || prompt_sha256:32bytes ||
seed:uint64_be`; require stream SHA-256
`fa769226902d7cc6487e4c3a52f2ab2985a011f39535f992d97c0589551b4c0d`.
The baseline raw text, extraction, input/output token counts, finish reason,
prompt hash, and seed come only from the qualified H31 log.

## Candidate runtime

Run one process under `CUDA_VISIBLE_DEVICES=1`, so both baseline and candidate
use the same physical GPU. Keep the historical checkpoint, BF16 weights,
LMDeploy 0.2.6, chat template, rope factor 2.0, 8,192-token prefill chunk,
`max_new_tokens=512`, top-k 40, top-p 0.8, temperature 0.8, repetition penalty
1.0, EOS handling, exact prompts, and all 32 recorded seeds unchanged.

Set only `session_len=160000` and `cache_max_entry_count=0.20`. The latter is a
target-independent 24-GB resource accommodation: H30 observed 394 blocks at
the default 0.8 ratio on this GPU, so the same free-memory formula predicts at
least 98 128-token blocks, or 12,544 internal tokens. Every selected input plus
the full output allowance needs at most 4,963 tokens. The lower cache ratio
changes available capacity, not token positions, weights, kernels, or sampling
parameters.

WARNING logging retains initialization truncation/failure messages without
echoing every prompt. The existing response gate must reject any invalid
request before it enters the formal sample log. No candidate output may be
generated as smoke; the 32 registered responses are the formal audit.

## Acceptance and failure policy

Require all source, H30 preflight, H31 artifact, selection, GPU, prompt, seed,
and response gates. H32 is supported only if all 32 candidate records exactly
match baseline raw prediction text, extracted answer, input token length,
generated token length, and finish reason. Also re-extract every candidate and
recheck correctness independently.

No partial percentage threshold applies: 31/32 is failure. If the fixed 20%
allocation cannot initialize or provides insufficient capacity, record the
failure and do not select another cache ratio after any model output. If all 32
match, stop session/cap variants and attribute H30/H31's remaining 4k residual
to an unidentified checkpoint/evaluator/training detail rather than the 8,192
cap.
