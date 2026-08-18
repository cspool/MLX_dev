# Frozen current-coupled Figure 19 transfer

## Outcome

H130 run135 is rejected with `audit_integrity=true`. It applies the frozen H99
composition—24 serialized layers at 1 GHz—to H129's 12 current-coupled component
cycles and compares every attention, FFN and total point.

Zero of 12 points pass. Global MAPE is 180.27% and maximum error is 232.69%.

| Series | Passing | MAPE | Maximum error |
|---|---:|---:|---:|
| Attention | 0/4 | 171.55% | 201.36% |
| FFN | 0/4 | 187.11% | 232.69% |
| Total | 0/4 | 182.15% | 223.13% |

Predicted totals are 5.28/10.55/21.30/39.81 ms versus targets
2.23/3.35/6.59/15.64 ms. H128/H129 reduce H99's 724% MAPE substantially, but
all components remain uniformly high.

## Stopping rule

The target table's attention and FFN components sum exactly to total, so adding
cross-component overlap would violate the published decomposition. The frozen
FABNet-Large contract independently fixes 24 layers and 1 GHz. No layer,
frequency, overlap or component factor is fitted from H130 residuals.

Figure 19 remains incomplete and active completion stays 0/8. Reopen only with
an author schedule/timing trace that changes the per-component simulator
contract. Work moves to Figure 18's target-free workload/performance identity.

Evidence is in
[run135](../artifacts/results/fig19-coupled-transfer-run135.json), with the
frozen plan in
[H130 protocol](../experiments/h130-fig19-coupled-transfer/protocol.md).
