# MLX FFT compression implementation source map

Source: user-provided local Markdown extraction of *MLX: Multi-Layer Execution
for Structured LLM Workload Acceleration on Spatial Architectures*. Layout page
anchors are unavailable; stable line/block anchors below refer to the frozen
85,855-byte Markdown source.

## Terminology ledger

| Canonical term | 中文 | Decision |
|---|---|---|
| semantic-aware Fourier compression | 语义感知傅里叶压缩 | Preserve the paper's method name |
| per-layer chunk length `L_l` | 逐层分块长度 `L_l` | Do not replace with a uniform L |
| compression ratio `s` | 压缩比例 `s` | Compressed chunk length is `sL` |
| leading frequency coefficients | 前部频率系数 | Exact real/complex convention remains unspecified |
| symmetric decompression | 对称解压 | Figure 7 omits the drawn reverse path |

<a id="S001"></a>
**Source:** Section III-A, Markdown lines 117-123

**Original:** “We confirm this by applying FFT to the Q, K, V vectors of
Llama2-7B across transformer layers ... Although Q/K/V are intermediate
representations, their frequency profiles reflect how each layer aggregates
semantic information along the sequence dimension.”

**中文:** 论文明确对各 Transformer 层的 Q、K、V 向量沿序列维执行 FFT；Q/K/V
虽然是中间表示，但其频谱用于反映各层聚合语义信息的方式。

**Implementation consequence:** H15's Q/K/V transform scope agrees with the
paper. A K/V-only or attention-output-only reinterpretation is not supported.

<a id="S002"></a>
**Source:** Section III-A, Markdown lines 123-127, Eq. (1)

**Original:** “we define a per-layer chunk length `L_l` as the sequence interval
that matches the shortest prominent variation scale of layer l ...
`L~=N/f~_H`, `L=Pow2Round(L~)`.”

**中文:** 论文把每一层的分块长度 `L_l` 定义为该层最短显著变化尺度对应的序列区间；
先由超过相对能量阈值的最高频谱峰得到 `N/f_H`，再量化到 2 的幂。

**Implementation consequence:** The paper requires layer-dependent L. H15/H38's
uniform `L=32` is an explicit inferred simplification, not a faithful reading.
The paper does not publish the threshold, aggregation protocol, or resulting
per-layer values.

<a id="S003"></a>
**Source:** Section III-A, Markdown line 128

**Original:** “for each matrix of `Q, K, V in R^(N x D)` after projection: (1)
reshape into `N/L` chunks and perform `N/L` independent L-point FFTs per feature
dimension; (2) truncate the last `(1-s)` fraction of high-frequency
coefficients ... keep leading informative `sL` components; (3) apply an
`sL`-point iFFT ... re-generating a shortened token representation.”

**中文:** 对投影后的每个 Q/K/V 矩阵，先按 `L` 分块并逐特征执行独立 L 点 FFT；再删除
末尾 `(1-s)L` 个高频系数、保留前部 `sL` 个系数；最后执行 `sL` 点 iFFT，得到缩短的
时域 token 表示。

**Implementation consequence:** Chunking, all-QKV scope, and shortened `sL`
attention are explicit. The text does not say how a real input preserves
conjugate symmetry after retaining a literal full-FFT prefix, nor whether the
implementation takes a real part, magnitude, or one-sided spectrum. H29's
real/complex ambiguity remains.

<a id="F001"></a>
**Source:** Fig. 7 (`_page_3_Figure_11.jpeg`)

![Fig. 7](../MLX%20Multi-Layer%20Execution%20for%20Structured%20LLM%20Workload%20Acceleration%20on%20Spatial%20Architectures/_page_3_Figure_11.jpeg)

**Original caption:** “Our approach: hybridizing structured sparsity and FFT
(Decompression is symmetric and omitted).”

**中文图注:** “本文方法：混合结构化稀疏与 FFT（对称解压过程省略未画）。”

**Reading note:** The diagram visibly shows an L-token time-domain block, an
L-coefficient frequency-domain block, removal of the rightmost `(1-s)L`
frequency region, and an `sL` time-domain result. It supplies no missing
real/complex convention.

<a id="S004"></a>
**Source:** Section III-A, Markdown line 139

**Original:** “In prefill, semantic-FFT is applied to the prompt in fixed-size
L-token chunks. In decode ... Completed chunks reuse cached compressed blocks,
while new tokens accumulate in a local buffer. Once the buffer reaches L, we
trigger FFT compression and append a new block.”

**中文:** prefill 对 prompt 按固定 L-token 块执行语义 FFT；decode 时复用已完成块的压缩
缓存，新 token 在局部 buffer 中累积，达到 L 后压缩并追加新块。

**Implementation consequence:** This supports independent chunks and append-only
cache behavior. It does not define autoregressive training likelihood or the
buffer's mixed compressed/uncompressed attention graph.

<a id="S005"></a>
**Source:** Section VII-A, Markdown line 350

**Original:** “BERT is also small enough for retraining, allowing us to apply
layer-wise FFT compression using the semantic interval length L (Eq. 1)
together with hierarchical BSMM sparsity ... five cases ... last k layers with
`s=0.5`.”

**中文:** BERT 可进行重新训练，因此论文把 Eq. (1) 的逐层语义区间 L 与 hierarchical
BSMM 同时应用，并在 `s=0.5` 下测试最后 k 层的五种设置。

**Implementation consequence:** The BERT experiment explicitly inherits the
layer-wise Eq. (1) rule. Uniform L is not justified by the evaluation text.

## Grounded conclusion

The supplied primary source supports three H15 choices: compress Q/K/V after
projection, form independent L-token chunks, and symmetrically decompress the
shortened attention result. It contradicts one H15 simplification: all layers
using L=32. It leaves two critical fields unresolved: the spectrum input /
aggregation / relative threshold that chooses each `L_l`, and the real/complex
coefficient convention.
