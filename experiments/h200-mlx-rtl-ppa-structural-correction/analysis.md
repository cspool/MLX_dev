# H200 result: structural correction improves area but is rejected

Run205 is rejected with `audit_integrity=true` and 6/10 gates. All four
128-repetition activity runs, twelve synthesis records and twenty component
power analyses are valid.

Area improves materially: FU passes at 6.53%; data network reaches 19.94%, tag
buffer 16.68%, and RF 35.40%. Config and control remain 159.93% high and 88.35%
low. The reduced design worsens to 364.23% because it still reuses the full
config/data/control/tag structures even though the paper describes a profiled
specialized subset.

Steady-state activity exposes a second issue. Driving all six buffered network
ingress paths together makes data-network power 235.21% high, while FU power is
75.36% low because the test still issues only the scalar instruction stream
instead of the paper's approximately 90% lane utilization. No component power
row passes. H201 therefore needs separate reduced non-compute parameters and a
paper-utilization activity contract; it must not add per-row scale factors.
