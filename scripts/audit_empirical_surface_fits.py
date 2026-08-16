#!/usr/bin/env python3
"""Re-fit the saturated Fig. 24/25 empirical surfaces without modifying files.

This audit makes the provenance of ``roofline_v1.yaml`` inspectable.  It prints
fresh coefficients and their maximum difference from the checked-in values.
An exact replay is expected and must not be interpreted as held-out validation.
"""

from __future__ import annotations

import json
import math

import numpy as np

from mlxsim.experiments import (
    PAPER_CALIBRATION,
    PROJECT_ROOT,
    ROOFLINE_CALIBRATION,
    _fig24_workload,
    load_targets,
)
from mlxsim.roofline import RooflineCalibration
from mlxsim.schema import CalibrationConfig, HardwareConfig
from mlxsim.simulator import MLXSimulator


def _fig25_design() -> np.ndarray:
    cases = ((512, 1024), (1024, 4096), (4096, 4096), (8192, 1024))
    rows = []
    for n, d in cases:
        log_n = math.log2(n / 512)
        log_d = math.log2(d / 1024)
        rows.append((1.0, log_n, log_d, log_n * log_d))
    return np.asarray(rows)


def _fig24_design() -> np.ndarray:
    cases = (
        ("BERT", 512),
        ("BERT", 8192),
        ("Llama2", 512),
        ("Llama2", 1024),
        ("Llama2", 4096),
        ("InternLM2", 2048),
        ("InternLM2", 8192),
    )
    rows = []
    for family, n in cases:
        log_n = math.log2(n / 512)
        llama = float(family.startswith("Llama"))
        intern = float(family.startswith("Intern"))
        rows.append((1.0, log_n, log_n * log_n, llama, intern, log_n * intern, log_n * llama))
    return np.asarray(rows)


def main() -> int:
    targets = load_targets()
    checked_in = RooflineCalibration.from_yaml(ROOFLINE_CALIBRATION)
    report: dict[str, object] = {
        "classification": "saturated-calibration-fit-audit",
        "validation_eligible": False,
    }

    design25 = _fig25_design()
    fitted25: dict[str, dict[str, list[float]]] = {}
    differences: list[float] = []
    for system, matrix in targets["fig25_roofline_utilization"]["heatmap"].items():
        fitted25[system] = {}
        for operator, values in zip(
            targets["fig25_roofline_utilization"]["operators"], matrix, strict=True
        ):
            coefficients = np.linalg.solve(design25, np.asarray(values))
            fitted25[system][operator] = coefficients.tolist()
            expected = np.asarray(checked_in.efficiency_coefficients[system][operator])
            differences.extend(np.abs(coefficients - expected).tolist())

    full_hardware = HardwareConfig.from_yaml(PROJECT_ROOT / "configs/hardware/mlx_full.yaml")
    event_calibration = CalibrationConfig.from_yaml(PAPER_CALIBRATION)
    simulator = MLXSimulator(full_hardware, event_calibration)
    cases24 = (
        ("BERT", 512, 1024),
        ("BERT", 8192, 1024),
        ("Llama2", 512, 4096),
        ("Llama2", 1024, 4096),
        ("Llama2", 4096, 4096),
        ("InternLM2", 2048, 4096),
        ("InternLM2", 8192, 4096),
    )
    design24 = _fig24_design()
    fitted24: dict[str, list[float]] = {}
    for operator, ratios in targets["fig24_structured_sweep"]["mlx_over_orin"].items():
        required_orin_gops = []
        for (_, n, d), ratio in zip(cases24, ratios, strict=True):
            mlx_gops = simulator.simulate(_fig24_workload(operator, n, d)).achieved_gops
            required_orin_gops.append(mlx_gops / ratio)
        coefficients = np.linalg.solve(design24, np.log(required_orin_gops))
        fitted24[operator] = coefficients.tolist()
        expected = np.asarray(checked_in.baseline_throughput_coefficients["orin"][operator])
        differences.extend(np.abs(coefficients - expected).tolist())

    report["fig25_efficiency_coefficients"] = fitted25
    report["fig24_orin_log_throughput_coefficients"] = fitted24
    report["maximum_checked_in_coefficient_difference"] = max(differences)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
