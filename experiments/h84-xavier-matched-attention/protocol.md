# H84 protocol: matched Xavier two-component Attention

H84 builds the Xavier side only after H83's MLX cycles are immutable. It uses
the H56 source-derived eight-SM Volta Xavier GPGPU-Sim configuration and a new
CUDA 11.8 compute_70 program with four separately timed launch families:

1. FFT-CMP: one thread computes one radix-2 pair with four FMA and six ADD
   operations, writing both outputs; F forward launches, one truncation launch,
   and I inverse launches at half pair count;
2. QK: one thread computes one score with D=4096 FMA iterations;
3. softmax statistics: one thread computes one row with R FMAX then R FEXP+ADD;
4. SV: one thread computes one output with R FMA iterations and one FDIV.

This exactly separates the H79 FU/stage work instead of representing both
components by one FFT proxy. For each shape and family, two outer-count anchors
fit an affine cycle model and two larger counts are held out:

- FFT pair counts 512/1024 fit, 2048/4096 hold out;
- QK and SV thread counts 128/256 fit, 512/1024 hold out;
- softmax row counts 8/16 fit, 32/64 hold out.

Every one of the 16 holdouts must be within 5%, every CUDA checksum within
1e-5, and every run must be detailed execution-driven PTX with normal exit.
Full counts are fixed from N/R/D: FFT forward pairs 1,572,864/50,331,648,
QK threads 16,384/16,777,216, softmax rows 128/4096, and SV threads
524,288/16,777,216. Full Xavier cycles are the sum of the four independently
validated models.

No Figure 20 performance target or H83-to-target residual is read. The
immutable output is
`artifacts/results/xavier-matched-attention-run089.json`.
