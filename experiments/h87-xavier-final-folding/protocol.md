# H87 protocol: final target-free Xavier folding gate

H87 performs one final independently justified steady-state check after H86
resolved numerical correctness but found 2048/4096 anchors slightly below the
5% cycle boundary.

The stable FFT source and attention source are byte frozen. For N=256 and
N=8192 FFT-CMP, count 4096/8192 anchors fit cycles and a new 16,384-pair run is
held out. For N=256 SV, count 4096/8192 anchors fit and a new 16,384-thread run
is held out. No other kernel is rerun.

If all three new runs pass checksum/normal-exit gates and all three cycle
errors are at most 5%, H87 may combine these models with H85's already passing
shared-QK and N=8192-SV models plus the directly executed complete softmax
measurements. Full counts remain exactly those frozen in H84/H86.

No further anchor range is permitted from this trajectory. No Figure 20 target
is read. The immutable output is
`artifacts/results/xavier-final-attention-run092.json`.
