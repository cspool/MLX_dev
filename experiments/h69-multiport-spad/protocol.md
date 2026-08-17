# H69 protocol: diagram-derived multi-port scratchpad

## Candidate fixed before execution

Fig. 9(a) visibly attaches scratch memory once per array column at the top and
once per row at the right. Fig. 11(a) states that BSMM uses column-wise SRAM
access. H69 therefore assigns BSMM requests to one independent validated H66
scratchpad port per mesh column: four ports on 4x4 and eight ports on 8x8,
selected by PE x-coordinate.

Each port retains H66's exact eight-bank, four-entry ordered DSAGEN pipeline.
Port replication is an explicit reconstruction hypothesis; the paper does not
disclose its queue implementation. No port count, latency, or buffer value is
derived from Figure 23 targets.

## Gates

- one-port mode remains byte-identical to H66;
- all 20 frozen H64 configs change only to adapter memory and execute twice;
- instruction/event/route/request/lane work remains exact;
- per-port request sums equal global requests and every replay is identical;
- compiler/runner read no paper target.

H69 performs no Figure 23 numerical comparison. The immutable output is
`artifacts/results/multiport-spad-run074.json`.
