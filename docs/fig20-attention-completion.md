# Figure 20 Attention and full-ledger completion

## Outcome

H136 run141 is supported with `audit_integrity=true` and 10/10 acceptance
gates. The user-directed primary criterion was frozen before result generation:
the target and prediction must both be speedups over baseline, and the predicted
speedup must be at least 1.2x. The original 10% comparison is retained as a
separate diagnostic.

| Scope | Trend passes | Strict <=10% passes |
|---|---:|---:|
| Attention | 2/2 | 1/2 |
| Full Figure 20 | 8/8 | 1/8 |

The refreshed Attention values are:

| Cell | Paper target | Open-simulator estimate | Relative error | Trend |
|---|---:|---:|---:|---|
| Attn-256 | 1.4x | 3.454x | 146.71% | clear speedup |
| Attn-8K | 3.1x | 3.152x | 1.69% | clear speedup |

The other six projection records remain identical in their target, estimate,
strict status, and evidence to H88; their estimates are all approximately
2.021x and therefore also pass the frozen trend gate. Thus Figure 20 increments
the primary active count to 1/8, while strict full-figure completion remains
0/8. This result does not claim that the short-sequence speedup magnitude was
numerically reproduced.

Evidence is in
[run141](../artifacts/results/fig20-attention-completion-run141.json), with the
frozen plan and pre-result amendment in
[H136 protocol](../experiments/h136-fig20-attention-completion/protocol.md).
