# H83 protocol: combined SIMD32 Attention with real DSAGEN SRAM

H83 compiles the first target-free, full-design Figure 20 Attention schedule.
H81/H82 remain SIMD8 mechanism anchors; H83 remaps their exact scalar work to
the paper's 4x4 SIMD32 full design and must not add their cycle totals.

The graph combines variable-depth FFT-CMP and grouped compressed Attention in
one paper-static tagged schedule. Each complex butterfly pair consumes two
64-byte SIMD32 packets and produces two packets. The truncation stage emits one
retained packet. FFT Q/K/V output packets stream directly to QK/SV through NoC
events without a scratchpad round trip. Only original Q/K/V input and final
Attention output use the H66-validated DSAGEN SRAM mechanism through H69's
diagram-derived four independent column ports.

Different input events require different reuse rates. H83 therefore adds an
optional `blocks[].wait_event_periods` object mapping each named wait event to
its positive period, falling back to H82's block-wide period. Defaults must
preserve the frozen H82 summary exactly.

Pre-execution structural review adds the complementary
`blocks[].wait_event_multiplicities` object. It records how many tokens from a
named event are required per authorized iteration group. Readiness for event e
is `(floor(iteration / period[e]) + 1) * multiplicity[e]` tokens. This is
necessary because an inverse butterfly consumes two retained packets per
iteration, whereas QK/SV reuse one packet over many iterations. Missing entries
default to one and remain backward compatible.

For combined scale u:

- N=256/R=128: FFT q=16u, Attention q=u, full u=128;
- N=8192/R=4096: FFT q=u, Attention q=2u, full u=65,536.

QK and SV reuse each pair of Q/K/V packet events for N local FMA iterations;
SV also reuses each weight event for D=4096 iterations and emits an output
vector every R iterations. Initial loads and final stores are exactly 64-byte
packets. Full off-chip bytes must be 7,340,032 and 234,881,024; boundary NoC
bytes must be 3,145,728 and 100,663,296. All H79 FU work must match exactly.

The active-tag window is two, independently fixed by the paper's 32-entry PE
instruction store: two simultaneously resident branch blocks must stay within
32 static instructions per PE. u=4/8 fit cycles and u=16/32 are held out at
5%. All configs run twice through four column SRAM ports.

No Figure 20 performance target is read. The immutable output is
`artifacts/results/combined-attention-memory-run088.json`.
