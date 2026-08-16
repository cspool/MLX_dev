# H1 analysis

## Outcome

**Supported with an explicit lineage caveat.** The target paper directly says its reduced 256-GOp/s design was tuned in a cycle-accurate simulator and cites SimICT. Initial searches found the SimICT paper and later ICT work describing it as an internally developed simulation framework, but no official public source repository. Therefore SimICT is the strongest evidence for the original simulation host, but it cannot be deployed here.

DSAGEN is the closest open spatial surrogate located:

- Its official stack includes a RISC-V host/ISA extension, LLVM path, spatial scheduler, gem5-integrated simulator, architecture description graph, and RTL generation.
- MLX uses the same broad decoupled-spatial vocabulary and includes a DSAGEN coauthor.
- The official DSAGEN instructions require all submodules and recommend a Docker/prebuilt environment; the documented prebuilt image is about 70 GB.

These facts justify using DSAGEN as an interface and mechanism reference. Shared concepts and authorship do **not** prove direct source-code reuse.

Accel-Sim v1.3.0 is the best open GPU-side reference found. It consumes SASS traces, uses GPGPU-Sim as the detailed timing model, and integrates AccelWattch. Its documentation includes Volta and an Ampere RTX 3060 example. The target paper spans Xavier/Volta, RTX 3090 and Orin/Ampere, and H100/Hopper; the latter is not established as a validated stock configuration, so H100 reproduction must use either a new tuned configuration or a clearly labeled roofline/native-measurement surrogate.

Timeloop is useful for analytical tensor mapping and energy cross-checks, but its stock abstraction cannot represent tag-priority layer folding and skip-hop contention, so it is not sufficient as the main simulator.

## Capability matrix

| Requirement | SimICT | DSAGEN | Accel-Sim | Timeloop | Local MLX model |
|---|---:|---:|---:|---:|---:|
| Open source located | No | Yes | Yes | Yes | Yes |
| Cycle/event timing | Reported | Yes | Yes | Analytical | Required |
| RISC-V host path | Unknown | Yes | No | No | Command-level |
| Spatial PE/NoC | Component-model capable | Yes | No | Abstract | Required |
| Tag-folded CDC scheduling | MLX extension | No stock support | No | No | Required |
| NVIDIA GPU traces | No | No | Yes | No | Roofline adapter |
| Power model | Integrates models | RTL-dependent | AccelWattch | Accelergy-compatible | Paper/RTL calibrated |

## Checked revisions

- DSAGEN meta repository: `273e141a519d12138ee0fbc9743059d13e9b5a64`; submodule pins include `dsa-gem5@1e5d2c3` and `dsa-scheduler@d0dd816`.
- Accel-Sim release: `v1.3.0` at `c5296df152c99a28dd64e5d9560bd58a8fd2e774`.
- Timeloop HEAD inspected on 2026-08-16: `32370826fdf1aa3c8deb0c93e6b2a2fc7cf053aa`.

## Decision

Implement MLX-specific timing locally, validate its abstractions against DSAGEN/Timeloop where meaningful, and make Accel-Sim an optional GPU backend. This avoids a 70-GB mandatory stack and permits deterministic CI on the available CPU host while retaining a path to detailed GPU traces.

