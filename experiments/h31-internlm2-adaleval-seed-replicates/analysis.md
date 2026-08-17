# H31 result: fixed-seed replication does not close the 4k gap

The three pre-registered 1,000-row seed schedules obtain 28.7%, 30.3%, and
27.6% Ada-LEval 4k accuracy. Their sole registered primary statistic is the
unweighted 3,000-response mean, 28.8667%. This is below the joint
32.31%-37.29% pass interval, so H31 rejects stochastic seed variation as a
sufficient explanation under this post-hoc diagnostic.

| Quantity | Replicate 0 | Replicate 1 | Replicate 2 | Registered mean |
|---|---:|---:|---:|---:|
| Correct / rows | 287/1,000 | 303/1,000 | 276/1,000 | 866/3,000 |
| Accuracy | 28.7% | 30.3% | 27.6% | 28.8667% |
| Binomial standard error | 1.430 pp | 1.453 pp | 1.414 pp | — |

The between-replicate sample standard deviation is 1.3577 percentage points
and the observed range is 2.7 points. Against the official 33.9% table, the
registered mean has 14.8476% relative error; against MLX's 35.9% bar, it has
19.5915% error. Both exceed 10%. Even the highest individual replicate, 30.3%,
is below the official 30.51% and joint 32.31% lower bounds; no favorable
replicate could pass the registered mean gate in any case.

The official report is
`artifacts/results/internlm2-adaleval-seed-replicates-run035.json` (28,744
bytes, SHA-256
`798824d79f18d697433dc983bd8953972027d7bc475ec795c550b3c4dedf0779`).
Its source/runtime commit is
`11baa7d10d62ace7cde670354c168c04f950927b`. The complete rank logs are:

| Rank | Records | Bytes | SHA-256 |
|---|---:|---:|---|
| 0 | 1,500 | 662,510 | `965b47b0fded39e9926a8715b2189076a8ffc57758a6f81ffb5cd3f5ede0c58e` |
| 1 | 1,500 | 662,376 | `fee65d35efc19697f8cbeecb5a6d937ad542b43dd9b4de2d166269e26d387f1d` |

## Qualification and integrity

Every non-target gate passes. The complete H30 source/runtime/prompt/capacity
preflight and run034 artifact hashes requalify before model loading. All 3,000
records have the exact pre-registered SHA-256-derived seed values and stream
hash `daa62375...45b0`; each replicate contains every position once, with 500
even/odd records per rank. Prompt hashes, rank partition, response bounds,
re-extraction, correctness, and arithmetic sample means all replay exactly.

There are no unextracted answers. All 3,000 requests finish normally with
`stop`; input lengths remain 3,455-4,451 and generated lengths 5-117 tokens,
well below the frozen 8,192 session and 512-token output limits. The run takes
922.28 seconds wall time on two RTX 4090 GPUs, with 1,673.18 summed
sample-seconds and 18,210 generated tokens.

## What the replications show

Decoder sampling materially changes individual designations. Only 331/1,000
positions have the same extracted prediction in all three schedules. Pairwise
prediction agreement is 421, 442, and 462 positions, while pairwise correctness
agreement is much higher at 816, 861, and 843. Thus seed choice changes many
wrong alternatives without shifting aggregate accuracy enough to reach either
target.

H30's independent 27.4% realization is consistent with the new range rather
than an isolated failure. Against H30, the three schedules respectively make
79/87/67 fixes and 66/58/65 regressions, giving net gains of 13, 29, and 2
examples. Prediction agreement with H30 is only 43.6%-46.4%, yet the accuracy
curve remains persistently low.

## Scope and consequence

H31 is rejected exactly as registered and remains validation-ineligible because
the 4k-only question was selected after H30. No replicate was selected, pooled
with H30, rerun, or used to change a seed, prompt, checkpoint, decoder, capacity,
or extractor.

The result rules out ordinary variation across these three independent fixed
schedules as a sufficient explanation for the 6.5-point H30-to-official gap.
It does not prove the official number is wrong: the official seeds are
unpublished, and a checkpoint/evaluator or task-adaptation detail may still
differ. Before assigning the residual to such an unpublished detail, a small
exact-seed equivalence audit should close H30's 8,192-versus-160,000 hardware
capacity accommodation.
