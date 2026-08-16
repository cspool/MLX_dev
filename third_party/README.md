# Optional third-party backends

Large upstream repositories are intentionally not vendored into Git. They are validation/reference backends; the default MLX simulator remains CPU-only and self-contained.

| Backend | Upstream | Pin | License | Role | Default? |
|---|---|---|---|---|---|
| DSAGEN | https://github.com/PolyArch/dsa-framework | `273e141a519d12138ee0fbc9743059d13e9b5a64` | BSD-2-Clause plus subproject licenses | Spatial ISA/compiler/gem5/RTL reference | No; official prebuilt stack is about 70 GB |
| Accel-Sim | https://github.com/accel-sim/accel-sim-framework | `v1.3.0` / `c5296df152c99a28dd64e5d9560bd58a8fd2e774` | See upstream | Trace-driven NVIDIA GPU and AccelWattch reference | No; requires CUDA and traces from real GPU |
| Timeloop | https://github.com/NVlabs/timeloop | `32370826fdf1aa3c8deb0c93e6b2a2fc7cf053aa` | BSD-3-Clause | Analytical tensor-mapping cross-check | No |

The pins above are evidence of the inspected versions, not a claim that the unpublished MLX source used any of them directly.
