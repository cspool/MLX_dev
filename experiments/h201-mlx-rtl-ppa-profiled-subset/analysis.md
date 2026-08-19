# H201 result: profiled subset narrows area error, power rejected

Run206 is rejected with `audit_integrity=true` and 7/10 gates.

The independently profiled reduced topology and full-only FP32 sidecar improve
area substantially. Config and RF pass at 5.41% and 8.98%; tag/FU are
20.32%/19.26%; data/control remain 47.30%/46.21%; reduced area is 33.21% high.
Area MAPE falls to 20.08% from H200's 76.78%.

Power remains structurally inconsistent. Config/data/control/tag/RF consume
clock internal power every cycle, while most FU area is combinational. Data is
282.59% high and FU is 86.38% low after the single aggregate power scale. The
next implementation must add domain clock gating and rebalance the remaining
full/reduced dimensions; simply increasing arithmetic area is insufficient.
