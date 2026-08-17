# H16 result: the inferred Llama2 spectrum protocol is rejected

The byte-qualified Llama2-7B run completed all 32 frozen WikiText-2 windows
and captured the pre-RoPE Q, K, and V projection outputs from all 32
transformer layers. Only 1 of the 42 numerical Fig. 6 bars passes the registered
10% gate, and only 2 of the 3 qualitative Fig. 5 checks pass. H16 is therefore
rejected.

| Audit | Passing points | MAPE | Maximum relative error |
|---|---:|---:|---:|
| Fig. 6 Layer-1 K | 0 / 21 | 92.90% | 96.83% |
| Fig. 6 Layer-16 K | 1 / 21 | 73.05% | 82.25% |
| Fig. 6 combined | 1 / 42 | 82.98% | 96.83% |

The sole numerical pass is Layer-16 frequency group 1, which is the shared
normalization peak and equals 1.0 in both target and measurement. This anchor
does not provide independent shape agreement. The official result is
`artifacts/results/llama2-spectrum-run019.json`.

## Qualitative Fig. 5 checks

| Frozen direction check | Measured dominant groups | Result |
|---|---|:---:|
| Layer-1 K above Layer-16 K | 21 > 1 | pass |
| Layer-1 K above Q above V | K=21, Q=3, V=21 | fail |
| Mean group in layers 1-4 above layers 13-16 | 9.5 > 1.0 | pass |

Thus the native activations support the broad shallow-high/deep-low trend, but
do not reproduce the paper's strict Layer-1 ordering because K and V select the
same highest qualifying group.

## Integrity checks

- Both safetensor shards and `tokenizer.model` match the official Hugging Face
  SHA-256 metadata byte for byte. Six small metadata files match their official
  Git blob IDs, and the 32-layer/4096-hidden configuration signature passes.
- The mirror is transport only: its accepted files are identical to official
  revision `01c7f73d771dfac7d292323805ebc428287df4f9`.
- The WikiText-2 parquet hash is the pre-registered
  `5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91`.
  The run uses the first 32 non-overlapping 1024-token windows with no added
  special tokens.
- The report records BF16 model execution on an RTX 4090, FP32 RFFT
  accumulation, all raw grouped curves, target-image hashes, and 23.09 seconds
  of measured run time.

## Diagnosis without changing the protocol

The disagreement is much larger than the raster uncertainty of 0.00987. Under
the registered shared scale, Layer-1 K occupies only 0.024-0.039 while the
paper bars occupy 0.270-0.783. Layer-16 has the same low-to-high ordering as the
paper, but decays from 1.0 to 0.066 rather than to roughly 0.336. A post-run
descriptive shape check gives Spearman correlations of 0.532 for Layer-1 and
0.995 for Layer-16; even independently fitted scalar multipliers leave RMSEs
of 0.171 and 0.369. Those fitted scalars are diagnostic only and are not
validation evidence.

The paper states that FFT is applied to Q/K/V along the sequence dimension, but
does not disclose the prompt or corpus, token count, feature/head selection,
number of samples, centering rule, magnitude-versus-squared-magnitude energy,
frequency grouping, or cross-layer normalization. In particular, averaging
all 4096 features over 32 unrelated text windows smooths the sharp Layer-1
peaks visible in the raster, while squared magnitude makes the Layer-16 decay
steeper. The residuals cannot identify which combination the authors used.

No prompt, window count, feature selection, spectral statistic, grouping, or
normalization is changed after observing run019. This result rejects the
explicit H16 reconstruction; it does not reject the general semantic-frequency
locality claim or establish that the paper used the same undisclosed protocol.
Exact numerical reproduction now requires the authors' spectrum script or a
complete activation-analysis recipe.
