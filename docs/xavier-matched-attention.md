# Matched Xavier Attention components

H84 replaces the old single-FFT Xavier Attention proxy with four
execution-driven CUDA families under the frozen eight-SM Xavier configuration:
variable-depth FFT-CMP, QK, softmax statistics, and SV/FDIV.

All 32 GPGPU-Sim runs complete in detailed PTX mode with normal exits and
checksum error below `1e-5`. Full logical FU work exactly matches H79 for both
N=256 and N=8192.

The registered small-anchor affine models are rejected:

| Shape/family | First holdout error | Second holdout error |
|---|---:|---:|
| N=256 FFT-CMP | 12.41% | 35.81% |
| N=256 QK | 3.17% | 7.58% |
| N=256 softmax | 0.00% | 17.74% |
| N=256 SV | 0.067% | 0.127% |
| N=8192 FFT-CMP | 12.10% | 35.19% |
| N=8192 QK | 3.17% | 7.58% |
| N=8192 softmax | 12.27% | 53.17% |
| N=8192 SV | 2.23% | 6.51% |

Only 6/16 holdouts pass. MAPE is 13.07% and maximum error is 53.17%. The
failure is caused by CTA/SM occupancy transitions: QK cycles are nearly flat
from 128 to 1024 threads while FFT-CMP and softmax slopes change as additional
CTAs and launches become active.

No full-size Xavier cycle sum from these failed models is eligible for Figure
20. A follow-up must use independently registered saturated anchors and new
larger holdouts; paper residuals remain forbidden.

The immutable result is
`artifacts/results/xavier-matched-attention-run089.json`.

The saturation follow-up is reported in
[`xavier-saturated-attention.md`](xavier-saturated-attention.md). Four of six
cycle models pass, but short SV and long softmax fail and one long FFT checksum
exceeds the frozen limit; a full Xavier sum remains disallowed.
