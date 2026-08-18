# H98 protocol: Figure 19 source-integrated paths

H98 implements the H97 workload without Figure 19 targets.

For each N, one combined two-axis FFT graph executes ten hidden-axis stages
followed by log2(N) token-axis stages. Per lane and scale, one iteration
represents 32 radix pairs, uses four FMA plus six ADD operations, two 64-byte
real input loads, four 64-byte complex intermediate transfers, and two 64-byte
real output stores. Full scale is 4N, exactly covering 512N pairs per stage.

The two global FFN paths retain H97's analytical FMA-equivalent convention and
use 10/12 tags for B1024/B4096. Each shape/path independently gcd-normalizes
full per-lane FMA and 64-byte load/store work, preserving constant weights and
N-scaled activations.

All paths use SIMD32, grouped adjacent events, active window two, and four H69
column SRAM ports. q=4/8 fit cycles and q=16/32 are held out for all 12 paths.

Support requires all 24 holdouts within 5%, exact source-derived FFT FU/packet
work, exact H97 global-FFN operations/bytes, and byte-identical double runs.
The 14-FLOP executable FFT pair mix is reported separately from H97's
conventional 10-FLOP analytical count.

The immutable output is
`artifacts/results/fig19-source-paths-run103.json`.
