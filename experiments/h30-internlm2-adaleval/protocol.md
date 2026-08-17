# H30 protocol: historical InternLM2 Ada-LEval baseline

## Hypothesis

The byte-qualified official InternLM2-Chat-7B checkpoint, evaluated on the
three exact public Ada-LEval BestAnswer files with the contemporaneous official
runner semantics, reproduces all three MLX Fig. 15(d) original accuracies
(52.8%, 40.6%, and 35.9%) within 10% relative error.

## Status before execution

H30 is a native, uncompressed checkpoint baseline. It is validation-eligible
for the frozen public reconstruction below, but the MLX paper does not disclose
its checkpoint revision, prompt, evaluator revision, inference engine version,
or random seeds. Agreement therefore cannot establish author provenance, and
H30 cannot validate either compressed s=0.5 bar.

The MLX targets and the official Ada-LEval reference results were both known
before execution. No prompt, chat template, checkpoint, generation parameter,
subset, or extraction rule may be changed from either residual. H30 reports
two independent gates: the MLX paper targets determine the hypothesis status;
the Ada-LEval README values (58.6%, 49.5%, and 33.9%) audit whether the
historical public stack itself was reconstructed.

## Frozen sources

- Ada-LEval repository revision
  2154258d5fa3969ac5429b3132d505570ef8a57a. Require the complete hashes
  of README.md, run.py, and ada_leval/dataset.py in the config. The README
  result segment at lines 78-93 has SHA-256 95dbdc0d...6dbae.
- Exact stackselect_1k.json, stackselect_2k.json, and stackselect_4k.json
  files, each containing 1,000 rows, with the byte counts and SHA-256 hashes
  frozen in the config. StackSelect normal mode takes the first 1,000 rows in
  file order.
- Official Hugging Face checkpoint revision
  4275caa205dbb8ff83930e2c1ce6bc62ec49329c, dated 2024-03-20 and therefore
  available before Ada-LEval's 2024-03-26 runner. Its eight safetensor objects,
  tokenizer object, config, index, and tokenizer files are byte-identical to
  the official ModelScope mirror at the pinned local revision. Build a
  read-only historical view from the 2024 source files and those exact local
  binary objects; require every listed hash before loading.
- LMDeploy 0.2.6, released 2024-03-19 and the latest release when the official
  runner appeared. Require tag commit b69e7176..., the 94,941,630-byte cp310
  wheel SHA-256 4e2d9044...bf4a, and the four source hashes that define the
  pipeline, chat template, and generation defaults.
- The immutable MLX target manifest has SHA-256 c4a22ab8...b41e.

## Frozen prompt and scoring semantics

Use StackSelect.build_prompt byte-for-byte: the upstream meta prompt, question,
all numbered answers in original order, and final designation-only request.
LMDeploy's InternLM2 template adds its exact system instruction with the
historical trailing newline and the im_start/im_end system, user, and assistant
markers.

Before model loading, require all three UTF-8 prompt-stream hashes, raw-token
hashes, and wrapped-token hashes in the config. The wrapped lengths are
488-1,239, 1,339-2,311, and 3,455-4,451 tokens for 1k/2k/4k respectively, so
no input approaches even the checkpoint's native 32k context.

Preserve the official extractor exactly. Search for A1 through An in the
prediction and choose the highest-numbered candidate present; only if no
designation occurs, repeat with bare numbers. Score exact equality with the
stored answer and report the unweighted percentage over all 1,000 rows.

## Historical inference semantics

Launch two ranks on the two physical RTX 4090 GPUs. For each setting, preserve
the official tups[rank::world_size] even/odd partition and process requests
sequentially. Instantiate TurbomindEngineConfig with only
rope_scaling_factor=2.0 and session_len=160000 changed from defaults.

The official pipe(prompt) call in LMDeploy 0.2.6 constructs max_new_tokens=512,
top_k=40, top_p=0.8, temperature=0.8, repetition_penalty=1.0,
ignore_eos=false, and applies the chat template. It then draws an independent
64-bit Python random seed because random_seed is unset. H30 may draw the same
distribution of seeds immediately before inference and pass them explicitly
solely so each sample is recorded and replayable; all other semantics remain
identical. Do not substitute modern greedy defaults or shorten generation.

Pin Python 3.10.20, PyTorch 2.1.2, Transformers 4.38.1, Triton 2.1.0,
NumPy 1.26.4, and SentencePiece 0.2.1 in an isolated ignored environment.
Also pin Protobuf 4.25.3, which LMDeploy 0.2.6 lists in its serving extras and
the historical fast-tokenizer converter requires. Record the GPU names,
driver, package versions, per-request seed, prompt length, output token count,
finish reason, raw prediction, extracted designation, and correctness. Each
rank writes a no-overwrite JSONL log; rank zero produces the aggregate report
only after exactly 3,000 unique sample records pass qualification.

## Acceptance gates

- Every source, repository revision, historical-view file, prompt canary,
  runtime version, GPU identity, row count, index uniqueness, and rank
  partition check must pass.
- Each setting must contain exactly 1,000 records, and its aggregate must equal
  the arithmetic mean of its per-sample correctness flags.
- The H30 hypothesis is supported only if all three measured accuracies are
  within 10% relative error of 52.8%, 40.6%, and 35.9%.
- Independently report whether all three values are within 10% of the official
  Ada-LEval 58.6%, 49.5%, and 33.9% table. That secondary gate cannot replace
  a failed MLX gate.

## Smoke and failure policy

After implementation, one fixed prompt per rank may be generated to verify
historical TurboMind compatibility. Smoke predictions are not persisted and
cannot alter any frozen choice.

After any smoke or formal output, do not change the checkpoint, subset, prompt,
template, engine version, generation defaults, seed distribution, or extractor
based on the residual. A gate failure rejects this public reconstruction. It
does not license a prompt or decoding sweep and says nothing about the
unpublished compressed model.
