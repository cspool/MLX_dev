# H106 historical DPU memory evidence

H106 narrows the open reconstruction to mechanisms stated in primary papers by
the MLX hardware-author line. It does not claim access to SimICT source code.

The 2018 non-stop-buffer paper provides the controlling state machine: two SPM
halves alternate by `tile_idx % 2`; each half has explicit DMA-versus-PE
ownership; addresses are relative to one half; results are copied out before a
new tile is copied in; and all tiles using one DFG pass through a continuously
running array with only one fill and one drain. Its evaluation configuration is
8 MiB SPM, 32 banks, 256 bits per bank, 64 GB/s at 1 GHz.

The 2019 operand-conflict paper confirms the host/DMA boundary and separates PE
operand-RAM slices from the array scratchpad. Operand replication is therefore
not modeled as extra SPM ports in H106.

The 2022 look-ahead paper confirms separate data/instruction SPM organizations,
four physical meshes, and the SimICT-to-gem5 calibration method. Its pre-fire
queue and PE input-buffer rules concern inter-PE context flow control and are
deferred rather than folded into scratchpad latency.

None of these sources discloses a DRAM timing tuple, DMA setup latency, or SPM
response latency. H106 uses the reported DMA bandwidth and the already frozen
H66 DSAGEN scratchpad timing, labels both provenance paths, and forbids tuning
either from an MLX figure residual.

