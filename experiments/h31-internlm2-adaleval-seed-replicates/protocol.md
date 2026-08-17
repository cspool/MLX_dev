# H31 protocol: fixed-seed Ada-LEval 4k replication

## Hypothesis and classification

Across three independently hash-fixed 1,000-row seed schedules, the mean
accuracy of H30's unchanged historical InternLM2-Chat-7B / Ada-LEval 4k stack
is within 10% relative error of both the official 33.9% reference and MLX's
35.9% original bar.

H31 is a post-hoc stochastic diagnostic. It selects 4k because H30 passed the
official 1k/2k gates but failed 4k after the residual was visible. It is
therefore validation-ineligible even if its mean passes. Its purpose is to
distinguish ordinary decoder variation from a persistent public-stack gap, not
to replace H30 or validate the compressed model.

## Frozen inherited stack

Require the 8,423-byte H30 config with SHA-256
`c73aa580151cb36551f9b165182def02c0583ab2423719753fc95d457fb4bd89`
and rerun its complete preflight before model loading. This preserves the exact
Ada-LEval repository/data bytes, historical 2024-03-20 checkpoint view,
LMDeploy 0.2.6 and dependency hashes, InternLM2 chat wrapper, prompt/token
canaries, two RTX 4090 identities, official 160,000-token source capacity,
8,192-token effective capacity, and all source qualifications.

Evaluate the same first 1,000 rows of `stackselect_4k.json`. Preserve H30's
`max_new_tokens=512`, `top_k=40`, `top_p=0.8`, `temperature=0.8`,
`repetition_penalty=1.0`, `ignore_eos=false`, rope factor 2.0, sequential
requests, exact extractor, and exact-equality scoring. Isolate the two workers
with one physical GPU each and assign even/odd positions within every
replicate. ERROR log verbosity suppresses LMDeploy's mislabelled normal-request
warnings only and is not an inference setting.

The 24,158-byte H30 report with SHA-256
`816731c7c3f6465b1e8a1de0f97661d7e776bc4b2ee9afa0e5099a789afc67c3`
is qualified for secondary comparison. Its 27.4% 4k result is not part of
H31's primary statistic.

## Seed schedules

For replicate `r in {0,1,2}` and dataset position `p in {0,...,999}`, form the
ASCII string

`H31|internlm2-adaleval-4k|replicate=r|position=p`

and interpret the first eight SHA-256 bytes as one unsigned big-endian 64-bit
seed. Serialize all seeds in replicate-major, position-minor order as unsigned
big-endian 64-bit values. Require exactly 3,000 distinct seeds and stream
SHA-256 `daa62375cb5a52782118a93beebf45ba21eb2544fb803e2a7484544a4bb945b0`.
This schedule is independent of worker timing and was frozen before any H31
model output.

## Records and statistics

Each worker writes a no-overwrite 1,500-row JSONL. Before every write, require
the independently tokenized wrapped input length, 1-512 output tokens, and a
consistent final stop/length reason using H30's response gate. Record the
replicate, position, seed, prompt hash, raw prediction, extracted answer,
correctness, token lengths, finish reason, rank, and wall time.

Require exactly 1,000 unique positions per replicate, exactly 500 records per
rank/replicate, all 3,000 schedule seeds, exact re-extraction, and arithmetic
agreement between serialized flags and every aggregate. Report each replicate
accuracy, standard error, range, between-replicate standard deviation,
per-position prediction agreement, and comparison with H30.

The sole primary statistic is the unweighted mean of all 3,000 H31 correctness
flags, equivalently the mean of the three equal-sized replicate accuracies. It
must be within 10% relative error of both targets. The joint pass interval is
32.31%-37.29%. An individual favorable replicate cannot substitute for the
mean, and H30 cannot be pooled into it.

## Failure policy

No H31 smoke generation is needed because the identical qualified H30 stack
already completed 3,000 responses. Preflight and unit tests may run before the
formal schedules. After any H31 output, do not change the seed namespace,
replicate count, checkpoint, prompt, decoder, capacity, dataset, or extractor.
A failed mean rejects stochastic variation as a sufficient explanation under
this registered diagnostic; it does not identify which unpublished detail
differs and does not authorize further favorable-seed trials.
