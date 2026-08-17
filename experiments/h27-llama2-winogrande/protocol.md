# H27 protocol: Llama2-7B WinoGrande baseline accuracy

## Hypothesis

The byte-qualified official Llama2-7B checkpoint, evaluated by lm-eval 0.4.12
on the pinned WinoGrande-xl validation split under its standard zero-shot
partial-scoring task, reproduces Fig. 15(c)'s original-model accuracy of 90.1%
within 10% relative error.

## Status before execution

This is a native unmodified-checkpoint baseline and is validation-eligible for
the explicitly frozen reconstruction below. The paper identifies Llama2-7B,
WinoGrande-xl, and sequence length 512, but omits the checkpoint revision,
split, prompt, few-shot count, and scoring implementation. Agreement therefore
cannot prove that the authors used the same recipe. No compressed layer, LoRA
adapter, or MLX operator is present in H27.

The 90.1% target was known before protocol selection. To prevent target-guided
selection, H27 uses the upstream harness task unchanged in scoring semantics,
makes its implicit zero-shot default explicit, and does not compare alternative
prompts, chat templates, few-shot counts, or answer-scoring modes.

## Frozen sources

- Official checkpoint:
  `meta-llama/Llama-2-7b-hf@01c7f73d771dfac7d292323805ebc428287df4f9`.
  Before model loading, require the same two safetensor SHA-256 hashes,
  tokenizer SHA-256, six official Git blob IDs, and config signature qualified
  in H16/H19.
- Official dataset:
  `allenai/winogrande@01e74176c63542e6b0bcb004dcdea22d94fb67b5`,
  configuration `winogrande_xl`, validation split. Its 85,928-byte validation
  parquet SHA-256 is
  `f9d914d1818c0aba98cabdf8af2bc4bd943c462b9c5e7062647e05c2ce3315d1`.
  It contains exactly 1,267 rows, with 628 label `1` and 639 label `2`; the
  canonical sorted-JSON-lines content hash is
  `8e8671d6097f314f4ddc6c9734e382810926a1e0c3e76664715786d25b4a78d4`.
  A pre-protocol pinned-revision load reproduced this content hash exactly.
- Harness: `lm-eval==0.4.12`. Require installed distribution-metadata SHA-256
  `072d33a0ff16a67aca5dbf5b2823ba11ad0b467a6ade6d5a2c5432c91959570a`.
  The upstream WinoGrande YAML and Python preprocessor hashes are respectively
  `7a73c28f1c760f1f54b4185f21f8df5927c1fe7278a8520f38ee03cc7f40d9e7`
  and
  `e55327629264a98f1d383fda24645b6779afb53fa0bce2150042b9416b1ae006`.
- The repository-local task preserves every upstream scoring field and the
  preprocessor byte-for-byte. Its only evaluation-semantic clarification is
  explicit `num_fewshot: 0`; its source change pins the dataset revision and
  gives the task a collision-free name. Local task YAML SHA-256 is
  `2045a0bb5ffb47216427100030066509240c592b0140fabfd40ac84b5490bad9`.

## Frozen task semantics

- Evaluate all 1,267 validation examples, with no subsampling or shuffled
  subset. The accuracy denominator must remain 1,267.
- Use lm-eval's upstream partial-scoring formulation. For each candidate, the
  filled sentence prefix is the conditioning text and the common text after
  the underscore is the scored continuation. Select the candidate with higher
  continuation log likelihood and report unweighted exact accuracy.
- Use zero demonstrations, no description, no system instruction, and no chat
  template. Preserve lm-eval's default target delimiter and BOS behavior
  (`add_bos_token=None`). The task's decontamination-query metadata remains
  present, but no example is removed from the fixed validation denominator.
- Set HFLM maximum length to 512 and request the official tokenizer with
  `use_fast_tokenizer=False`. H19 established that Transformers 5.15's unified
  tokenizer backend is ID-equivalent to the official SentencePiece model.

## Runtime and records

- Run the unmodified model in BF16 at fixed batch size 32 on physical GPU 1,
  exposed as `cuda:0`. Require the RTX 4090 device name.
- Pin Python 3.12.13, PyTorch 2.13.0+cu130, Transformers 5.15.0, and Datasets
  5.0.1. Pin seeds to 0/1234/1234/1234 for Python, NumPy, PyTorch, and few-shot
  sampling respectively; no few-shot samples are actually drawn.
- Preserve lm-eval's 100,000 bootstrap iterations. Enable `log_samples` and
  `write_out`; write all 1,267 sample records to the fixed JSONL path and store
  its SHA-256 in the aggregate result. The official paths refuse overwrite.

## Acceptance gate

- All checkpoint, dataset, harness, task-template, runtime, and canonical-target
  qualifications pass before accepting the result.
- The harness returns exactly one `acc,none` point estimate and 1,267 sample
  records. The aggregate accuracy must equal the arithmetic mean of per-sample
  `acc` values to floating-point tolerance.
- Convert the aggregate to percentage points and compare with the printed
  Fig. 15(c) target 90.1. H27 is supported only if relative error is at most
  10%, i.e. measured accuracy is at least 81.09%.

## Smoke-test and failure policy

One validation example may be scored after implementation to check model/task
compatibility. It is labeled non-validation, is not written to either official
path, and cannot alter any frozen choice.

After any smoke or formal output, do not change the revision, split, prompt,
few-shot count, partial-scoring implementation, tokenizer/BOS behavior,
precision, or length based on the residual. A gate failure rejects this
standard baseline reconstruction. It must not be repaired by choosing a
nonstandard WinoGrande recipe, and it says nothing by itself about the paper's
compressed `s=0.75` bar.
