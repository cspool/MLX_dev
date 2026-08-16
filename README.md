# MLX paper reproduction

This repository is a transparent, open surrogate for the unpublished simulator used by *MLX: Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures* (ISCA 2026). It does **not** claim to contain the authors' simulator or RTL.

The default path is a CPU-only tag/block discrete-event simulator. Optional pinned DSAGEN, Accel-Sim, and Timeloop integrations provide spatial, GPU, and analytical cross-checks. Every comparison distinguishes paper-reported, raster-digitized, measured, inferred, calibrated, and locally simulated data.

## Quick start

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
mlxsim simulate --hardware configs/hardware/mlx_reduced.yaml \
  --calibration configs/calibration/paper_v1.yaml \
  --kernel bsmm --n 1024 --d 512 --block-size 32
mlxsim reproduce --figure 22 --output artifacts/results/fig22.json
mlxsim reproduce --figure 23 --output artifacts/results/fig23.json
mlxsim reproduce --figure 20 --output artifacts/results/fig20.json
mlxsim reproduce --figure 21 --output artifacts/results/fig21.json
mlxsim reproduce --figure 18 --output artifacts/results/fig18-consistency.json
mlxsim reproduce --figure tables --output artifacts/results/tables.json
mlxsim reproduce --figure 24 --output artifacts/results/fig24.json
mlxsim reproduce --figure 25 --output artifacts/results/fig25.json
python scripts/audit_empirical_surface_fits.py
pytest
```

The full-paper completion checklist is in `docs/experiment_inventory.md`. Research decisions and current limitations are in `findings.md` and `research-state.yaml`.

## Evidence policy

- Paper target values live only under `artifacts/targets/`.
- Simulator code may read hardware/workload configuration, but never paper result bars as timing inputs.
- Calibrated parameters are named and versioned under `configs/calibration/`.
- Fig. 24/25 empirical surfaces are saturated calibration replays and explicitly export `validation_eligible: false`.
- A result is accepted only by a point-wise audit; matching a headline geometric mean is insufficient.
