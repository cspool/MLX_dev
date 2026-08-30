#!/usr/bin/env python3
"""Build/run MLX bare-metal ELFs on the Chipyard cycle and RTL configs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/system/mlx_riscv_system_v1.yaml"
BACKEND_CONFIGS = {
    "cycle": "MLXCycleRocketConfig",
    "rtl": "MLXRTLRocketConfig",
}
CHIPYARD_RESOURCES = [
    "mlx_fp16.sv",
    "mlx_fu.sv",
    "mlx_register_file.sv",
    "mlx_tag_buffer.sv",
    "mlx_config_network.sv",
    "mlx_data_network.sv",
    "mlx_control_logic.sv",
    "mlx_pe_top.sv",
    "mlx_array_pe_tile.sv",
    "mlx_array_4x4_distributed.sv",
    "mlx_array_4x4.sv",
    "mlx_cycle_model.sv",
    "mlx_rocc_controller.sv",
]


def digest(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    try:
        name = str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        name = str(path)
    return {
        "path": name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def run(
    command: list[str],
    *,
    log: Path | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(result.stdout + result.stderr)
    return result


def parse_elf_pass(text: str) -> dict[str, int | str]:
    line = next((line for line in text.splitlines() if "MLX_ELF_PASS " in line), None)
    if line is None:
        raise ValueError("Chipyard log has no MLX_ELF_PASS line")
    line = line[line.index("MLX_ELF_PASS ") :]
    values: dict[str, int | str] = {}
    for key, value in re.findall(r"(\w+)=([^\s]+)", line):
        values[key] = value if key in {"workload", "backend"} else int(value)
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--backend", choices=["all", *BACKEND_CONFIGS], default="all")
    parser.add_argument("--build", action="store_true")
    parser.add_argument(
        "--reuse-logs",
        action="store_true",
        help="rebuild the manifest from existing successful simulator logs",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    chipyard = Path(config["chipyard"]["root"])
    selected = list(BACKEND_CONFIGS) if args.backend == "all" else [args.backend]
    output = PROJECT_ROOT / "artifacts/environment/h207"
    output.mkdir(parents=True, exist_ok=True)

    setup_records: dict[str, int] = {}
    if args.build:
        install = run(["bash", "scripts/install_mlx_chipyard.sh", str(chipyard)])
        setup_records["install_returncode"] = install.returncode
        workloads = run(
            [sys.executable, "-m", "scripts.build_mlx_system_workloads"]
        )
        setup_records["workload_returncode"] = workloads.returncode
        software = run(["make", "-C", "system_sim/software", "-j4", "all"])
        setup_records["software_returncode"] = software.returncode
        for backend in selected:
            build = run(
                [
                    "bash",
                    "-lc",
                    (
                        f"source {chipyard}/env.sh && make -C {chipyard}/sims/verilator "
                        f"CONFIG={BACKEND_CONFIGS[backend]} -j4"
                    ),
                ],
                log=output / f"build-{backend}.log",
                timeout=1800,
            )
            setup_records[f"build_{backend}_returncode"] = build.returncode

    workload_manifest = json.loads(
        (PROJECT_ROOT / config["manifest"]).read_text()
    )
    standalone = json.loads(
        (PROJECT_ROOT / "artifacts/results/mlx-system-backends-run210.json").read_text()
    )
    standalone_by_name = {item["workload"]: item for item in standalone["records"]}
    installed_root = chipyard / "generators/chipyard/src/main"
    installed_pairs = [
        (
            PROJECT_ROOT / "system_sim/chipyard/MLXRoCC.scala",
            installed_root / "scala/MLXRoCC.scala",
        ),
        *(
            (
                PROJECT_ROOT / f"rtl/mlx/{name}",
                installed_root / f"resources/vsrc/{name}",
            )
            for name in CHIPYARD_RESOURCES
        ),
    ]
    preflight_checks = {
        "installed_sources_match": all(
            target.is_file() and source.read_bytes() == target.read_bytes()
            for source, target in installed_pairs
        ),
        "simulators_present": all(
            (
                chipyard
                / f"sims/verilator/simulator-chipyard-{BACKEND_CONFIGS[backend]}"
            ).is_file()
            for backend in selected
        ),
        "elfs_present": all(
            (PROJECT_ROOT / f"system_sim/build/software/mlx-{item['name']}.riscv").is_file()
            for item in workload_manifest["workloads"]
        ),
        "build_logs_present": all(
            (output / f"build-{backend}.log").is_file()
            and (output / f"build-{backend}.log").stat().st_size > 0
            for backend in selected
        ),
    }
    records = []
    for backend in selected:
        simulator = chipyard / f"sims/verilator/simulator-chipyard-{BACKEND_CONFIGS[backend]}"
        if not simulator.is_file():
            raise FileNotFoundError(f"missing Chipyard simulator: {simulator}")
        for workload in workload_manifest["workloads"]:
            name = workload["name"]
            elf = PROJECT_ROOT / f"system_sim/build/software/mlx-{name}.riscv"
            log = output / backend / f"{name}.log"
            reused = args.reuse_logs and log.is_file()
            if reused:
                execution_returncode = 0
            else:
                execution_returncode = run(
                    [str(simulator), str(elf)], log=log, timeout=600
                ).returncode
            summary = parse_elf_pass(log.read_text())
            summary["host_total"] = int(summary["host_config"]) + int(
                summary["host_launch_wait"]
            )
            expected_dma_bytes = 64 * (
                int(workload["input_vectors"]) + int(workload["output_vectors"])
            )
            expected_kernel = standalone_by_name[name][backend]["summary"]
            records.append(
                {
                    "backend": backend,
                    "config": BACKEND_CONFIGS[backend],
                    "workload": name,
                    "returncode": execution_returncode,
                    "execution_reused": reused,
                    "summary": summary,
                    "checks": {
                        "pass": execution_returncode == 0,
                        "identity": summary["backend"] == backend
                        and summary["workload"] == name,
                        "kernel_cycles": summary["kernel"] == expected_kernel["cycles"],
                        "instruction_count": summary["instructions"]
                        == workload["instruction_count"],
                        "system_overhead_included": summary["system"]
                        > summary["kernel"]
                        and summary["dma"] > 0,
                        "system_cycle_accounting": summary["system"]
                        == summary["dma"] + summary["kernel"] + 2,
                        "host_total_accounting": summary["host_total"]
                        == summary["host_config"] + summary["host_launch_wait"]
                        and summary["host_total"] > summary["system"],
                        "dma_bytes": summary["dma_bytes"] == expected_dma_bytes,
                        "operation_accounting": summary["instructions"]
                        == summary["load"]
                        + summary["store"]
                        + summary["compute"]
                        + summary["xfer"],
                    },
                    "artifacts": {
                        "simulator": digest(simulator),
                        "elf": digest(elf),
                        "log": digest(log),
                    },
                }
            )

    checks = {
        "setup": all(value == 0 for value in setup_records.values()),
        "preflight": all(preflight_checks.values()),
        "expected_runs": len(records) == 4 * len(selected),
        "all_elf_runs": all(
            all(item["checks"].values()) for item in records
        ),
        "both_backends": args.backend != "all" or {item["backend"] for item in records}
        == {"cycle", "rtl"},
        "chipyard_commit": run(
            ["git", "-C", str(chipyard), "rev-parse", "HEAD"]
        ).stdout.strip()
        == config["chipyard"]["commit"],
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": "H207",
        "run_id": "run212",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classification": "measured_chipyard_verilator_bare_metal_system_simulation",
        "paper_performance_targets_consumed": False,
        "chipyard": config["chipyard"],
        "command_interface": config["interface"],
        "setup": setup_records,
        "preflight": preflight_checks,
        "build_logs": {
            backend: digest(output / f"build-{backend}.log")
            for backend in selected
        },
        "execution_mode": "reused_logs" if args.reuse_logs else "fresh_simulation",
        "records": records,
        "checks": checks,
        "provenance": {
            "host_and_total_cycles": "measured by RISC-V rdcycle in bare-metal ELF",
            "system_dma_cycles": "measured by MLX RoCC controller",
            "kernel_breakdown": "measured by selected MLX backend",
            "goldens": "generated software FP16 architectural reference",
        },
    }
    manifest_path = output / "mlx-chipyard-run-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    result = {
        "schema_version": 1,
        "experiment_id": "H207",
        "run_id": "run212",
        "status": "supported" if all(checks.values()) else "rejected",
        "claim": "real RISC-V bare-metal ELF config/launch/wait/status closure on Chipyard",
        "records": records,
        "checks": checks,
        "manifest": digest(manifest_path),
    }
    result_path = PROJECT_ROOT / "artifacts/results/mlx-chipyard-system-run212.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
