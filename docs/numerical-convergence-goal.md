# Figure 19/20/23 numerical-convergence goal

Adjust the simulator and experiment implementation so that every registered
Figure23, Figure19 and Figure20 performance value is within 10--15% of the MLX
paper and retains the same baseline-relative direction. Native RTX4090 traces
may supply workload and scale-regime features.

The implementation must use shared mechanism parameters rather than a
coefficient per paper point. It must also provide a documented, replayable path
from model/operator graphs to native MLX overlay/DPU-memory JSON or analytical
KernelProfile JSON, simulator execution and audit results.

Paper-target calibration must be disclosed. Independent validation, the
authors' unpublished LLVM/spatial assembler, RTL, power and area are not
claimed.
