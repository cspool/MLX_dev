# H195 result: N4096 Attention holdouts repaired

Run200 is supported with `audit_integrity=true` and 10/10 gates. The repair
reduces the dense-TCU N4096 error from 27.89% to 2.39% and the sparse-CUDA error
from 20.91% to 1.22%. All six Figure20 Attention holdouts and all 48 combined
holdouts now pass 15%; all 36 registered directions remain correct. Overall
MAPE is 3.59% and the maximum error is 14.53%.

The failure was in the experiment-side feature mapping, not tensor execution.
H193 fed a raw RTX4090 FlashAttention/rfft timing contrast into a Xavier
comparison. That contrast rises from -0.0316 at N256 to 0.2595 at N2048, then
reverses to 0.2210 at N4096 before reaching 0.4688 at N8192. The correction
cross-fits the contrast against log2 sequence length, excluding the predicted
shape. N4096 therefore uses N256/512/2048/8192 trace evidence, not its own trace
or either evaluation reference.

No H183 parameter is refit, and 42 non-Attention predictions remain exactly
unchanged. This is a post-failure repair against synthetic two-endpoint log-N
references. N4096 is not measured in the paper; RTX4090 is not Xavier, and the
sparse rfft proxy omits QK, softmax and SV. Run200 consequently repairs the
registered error but does not become author-hardware or independent validation.
