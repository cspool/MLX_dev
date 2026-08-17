# H20 result: inferred FFT + LoRA isolation is rejected

The frozen run completes both full test evaluations, all 256 training windows,
and all 64 optimizer steps. The post-training perplexity is 3.072047 versus the
Fig. 15(d) target 5.7810, a 46.86% relative error. Because the registered gate
is agreement rather than “lower is always better,” H20 is rejected.

| Stage | Perplexity | Target | Relative error | Gate |
|---|---:|---:|---:|:---:|
| Compressed, zero-initialized LoRA | 31.0473 | 5.7810 | 437.06% | fail |
| After 256 windows / 64 steps | 3.0720 | 5.7810 | 46.86% | fail |

The official result is
`artifacts/results/llama2-fft-lora-s075-run023.json`. This was pre-registered as
FFT/LoRA isolation and has `full_mlx_bar_reproduced=false` independently of the
numeric result because B=32 hierarchical BSMM is absent.

## Training and integrity

- Exact official model bytes and both WikiText-2 hashes pass. Evaluation uses
  all 333 H19 windows and 340,659 predicted tokens before and after training.
- Compression is installed in exactly layers 12-31. PEFT exposes 12,492,800
  trainable parameters (0.1851% of 6.751B); all 280 trainable tensors are LoRA
  tensors inside those 20 layers, with none outside the registered scope.
- Mean training loss is 1.6943, falling from 3.3387 on the first shuffled window
  to 1.1798 on the last. The run takes 255.6 seconds and peaks at 14.68 GB
  allocated memory on one RTX 4090.
- The adapter-only checkpoint is 50,008,256 bytes with SHA-256
  `dbec7a10fcf8e7bc852e9ade5cdd22901b54ad6da5dd172e2773b430c0bc6320`.

The PEFT skill shaped the frozen low-rank starting point (r=8, alpha=16),
attention-plus-MLP module coverage, gradient checkpointing, trainable-parameter
audit, held-out evaluation, and adapter-only save. These controls establish that
the failure is not an accidentally unfrozen base model or inactive adapter.

## Causal diagnosis

H20 declared before execution that post-RoPE Fourier mixing over a complete
teacher-forced chunk can expose later input tokens to earlier logit positions.
A post-hoc, target-independent diagnostic reloads the exact adapter hash and
scores two subsets in the same 333 forward passes:

| Diagnostic subset | Predicted tokens | Perplexity |
|---|---:|---:|
| Every within-window next token | 340,659 | 3.0720 |
| Only positions 31, 63, ..., 991 | 10,323 | 18.0464 |

At each selected chunk-end logit, the next-token label lies in the following
chunk and cannot enter that logit's FFT input. Its PPL is 5.87x the all-token
PPL. The all-token value reproduces run023 exactly, ruling out an adapter-load
difference. This does not make the 10,323-token subset directly comparable to
the paper's full PPL target, but it confirms that the apparent 3.07 “improvement”
is dominated by the inferred wiring's invalid teacher-forcing advantage. The
diagnostic is `artifacts/environment/llama2-fft-run023-causality.json`.

## Consequence

No learning rate, step count, rank, layer set, or FFT placement is changed from
the residual. Running `s=0.5` with the same non-causal wiring would compound a
known semantic defect and is not justified. Exact compressed-Llama reproduction
now requires the authors' causal training/evaluation graph, chunk-cache rule,
and BSMM initialization in addition to their LoRA recipe.
