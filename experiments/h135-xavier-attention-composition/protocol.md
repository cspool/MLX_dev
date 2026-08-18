# H135 protocol: target-free Xavier/MLX Attention composition

## Hypothesis

H133 FFT plus H134 QK/softmax/SV form complete finite Xavier Attention totals
for N256/N8192, enabling target-free speedups against H83 MLX cycles under fixed
1.377-GHz/1-GHz clocks.

## Composition

For each shape, sum serialized Xavier component cycles, divide by 1.377 GHz,
divide H83 full cycles by 1 GHz, then compute Xavier seconds / MLX seconds. Do
not read Figure 20 targets or introduce overlap/factors.

## Acceptance gates

1. H133/H134/H83 qualify and are supported with integrity.
2. Both shapes contain eligible FFT/QK/softmax/SV components exactly once.
3. Xavier total cycles equal the exact four-component sum.
4. H83 MLX cycles are copied exactly from its passing models.
5. Fixed device clocks match parent evidence and are positive.
6. Both total times and speedups are finite and positive.
7. Component/provenance records are retained in output.
8. Auditor/test contain no Figure 20 target, fit, factor or overlap path.
9. Results remain labeled transparent Xavier proxies, not author CUDA timing.
10. H135 changes no active 0/8 count; H136 separately joins targets.

Support requires all ten gates. The immutable result will be
`artifacts/results/xavier-attention-composition-run140.json`.
