# MLX off-chip memory path in the open hybrid simulator

The simulator now models MLX as a spatial overlay whose PEs retain local,
programmable load, store, compute, and transfer pipelines. DSAGEN/dsa-gem5
owns tag scheduling, the PE mesh, skip-hop routing, scratchpads, and the clock.
The PE follows the paper's static tagged-block control. Accel-Sim/GPGPU-Sim is
not required by this path and is used only for separate GPU baselines.

For `memory_backend=dsagen_dma`, an overlay memory instruction enters a
dedicated MinorCPU LSQ transfer queue. Loads pass through address translation,
L1D, L2, and the configured memory controller. Stores use Minor's existing
store buffer and are acknowledged to the overlay only after the real response,
not at store-buffer insertion. A distinct `cpu.mlx_dma` requestor ID makes the
cache and DRAM evidence independently auditable.

H47's cold-memory microtrace uses 16 PE-local blocks with four iterations of
load, integer add, and store: 64 operations of each kind. The fixed control and
DMA run use byte-identical schedules except for the backend. The guest resolves
addresses from aligned ELF symbols, initializes the read targets, conditions
the 512-KiB L2 with a separate 2-MiB region, and verifies the write result.

The H47 result is a mechanism result, not a paper-accuracy result. It establishes
that subsequent full BSMM/FFT/attention/norm schedules can use a real open
cache/DDR path. It does not establish equivalence to the authors' unpublished
simulator or validate an MLX figure by itself.
