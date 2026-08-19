# H168 protocol: target-free GPU baseline mapping audit

## Hypothesis

A complete, source-qualified mapping can be built for all four GPU devices used
by MLX while correctly rejecting every current denominator as strict validation
evidence. Accel-Sim/GPGPU-Sim remains the preferred Volta/Ampere substrate;
FlashGPU-Sim is the preferred Hopper candidate. Device facts, timing templates,
workload mappings and missing native evidence must remain separate.

## Frozen evidence

- H51/H54/H56 provide target-free detailed PTX executions for resource-edited
  RTX3090, Orin and Xavier proxies.
- H123 proves a 6.149% Orin cycle spread from CTA shape alone at equal work.
- H143 finds zero qualified dense-Xavier family rows for Figure 21 before later
  compute-only synthetic services.
- The source note records that the earlier H100 cross-figure analytical path
  was rejected, but H168 does not read its target-exposed result file.
- The 2026-08-19 source note freezes NVIDIA facts, Accel-Sim/GPGPU-Sim pins and
  FlashGPU-Sim SM90 pin `f3d4bba`.

No paper performance bar or target-exposed artifact is consumed by this audit.

## Device classification

For Xavier, Orin, RTX3090 and H100, emit:

1. paper figure and workload role;
2. target ISA and vendor resource identity;
3. preferred open simulator;
4. current local timing template and cross-device substitutions;
5. executable proxy status;
6. native tuner/config, application trace and correlation status;
7. strict validation eligibility and the exact blocking gaps.

An executable proxy is not validation-eligible merely because its functional
checksum passes. ISA/timing identity, exact application schedule and native
correlation are all required.

## Acceptance gates

1. All source and local evidence files pass byte/hash qualification.
2. H51/H54/H56 remain supported, target-free detailed proxy executions.
3. Exactly four device records cover every MLX GPU figure role.
4. Vendor identity and an open simulator candidate exist for all four devices.
5. Exactly three local executable proxies exist; H100 has no local execution.
6. Zero devices have a native target-tuned config and exact application traces.
7. Zero devices are promoted to strict validation eligibility.
8. Xavier dense-family, Orin schedule and H100 analytical failures are retained.
9. RTX3090 is labeled the closest current ISA-family proxy and FlashGPU-Sim the
   preferred H100 candidate; neither is mislabeled as validated.
10. The result claims mapping/gap completion only and consumes no performance
    target.

The immutable result will be
`artifacts/results/gpu-baseline-mapping-run173.json`.
