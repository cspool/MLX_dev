#!/usr/bin/env python3
"""Audit H104's author-centric MLX simulator lineage survey."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/mlx_author_simulator_lineage_v1.yaml"


LINEAGE_RECORDS: list[dict[str, Any]] = [
    {
        "id": "simict_2013",
        "year": 2013,
        "title": "SimICT: A Fast and Flexible Framework for Performance and Power Evaluation of Large-Scale Architecture",
        "role": "named_simulation_framework",
        "sources": ["openalex_simict", "jcst_2022_lookahead_pdf"],
        "simulator": "Component-based parallel framework; later team papers explicitly use it for cycle-accurate DPU models.",
        "implementation": "Performance/power models are integrated as configurable components; source repository not found.",
        "relationship": "MLX's simulator citation [36] resolves to SimICT; this supports framework ancestry, not code identity.",
    },
    {
        "id": "bdsim_2015",
        "year": 2015,
        "title": "BDSim: A Component-Based Highly Configurable Parallel Simulation Framework for Big-Data Application Evaluation",
        "role": "team_parallel_simulation_precedent",
        "sources": ["ucas_wenming"],
        "simulator": "Same-team componentized parallel simulation publication for many-core/big-data evaluation.",
        "implementation": "Bibliographic primary evidence only; no public code or MLX citation identified.",
        "relationship": "Possible infrastructure precedent, weaker than the explicit SimICT citation.",
    },
    {
        "id": "dpu_noc_2017",
        "year": 2017,
        "title": "An Efficient Network-on-Chip Router for Dataflow Architecture",
        "role": "dpu_microarchitecture_precedent",
        "sources": ["jcst_2017_noc_pdf"],
        "simulator": "SimICT-based dataflow PE/NoC evaluation.",
        "implementation": "Multiple-destination, high-injection, latency-sensitive non-flit NoC design.",
        "relationship": "Establishes the ICT DPU PE-array/NoC line that predates MLX.",
    },
    {
        "id": "dpu_loop_2018",
        "year": 2018,
        "title": "A Pipelining Loop Optimization Method for Dataflow Architecture",
        "role": "dpu_execution_precedent",
        "sources": ["jcst_2018_loop_pdf"],
        "simulator": "SimICT dataflow accelerator model with an 8x8 PE array and SPM.",
        "implementation": "Loop controller/instructions raise ready work and pipeline utilization.",
        "relationship": "Precedent for MLX loop-driven tagged templates, without MLX tags or CDCs.",
    },
    {
        "id": "dpu_buffer_2018",
        "year": 2018,
        "title": "A Non-Stop Double Buffering Mechanism for Dataflow Architecture",
        "role": "dpu_memory_precedent",
        "sources": ["jcst_2018_buffer_pdf"],
        "simulator": "SimICT model exposes instruction buffers and multiple blocks in flight.",
        "implementation": "Non-stop double buffering avoids repeated array fill/drain across tiles.",
        "relationship": "Directly relevant to MLX data-supply and folded execution timing.",
    },
    {
        "id": "dpu_4x4_2019",
        "year": 2019,
        "title": "Optimum Research on Inner-Instruction Memory Access Conflict for Dataflow Architecture",
        "role": "closest_public_reduced_dpu_configuration",
        "sources": ["crad_2019_memory_conflict"],
        "simulator": "SimICT hosts a 1 GHz 4x4 dataflow array with ARM host, SPM, operand RAM and two 64-bit XY NoCs.",
        "implementation": "A matching Verilog implementation is synthesized in a foundry process for area/power.",
        "relationship": "Strong public predecessor for MLX's compact reduced design, but not an explicit parent statement.",
    },
    {
        "id": "dpu_transfer_2022",
        "year": 2022,
        "title": "Accelerating Data Transfer in Dataflow Architectures Through a Look-Ahead Acknowledgment Mechanism",
        "role": "dpu_validation_and_transfer_precedent",
        "sources": ["jcst_2022_lookahead_pdf"],
        "simulator": "Cycle-accurate SimICT model calibrated against gem5; host derives from SimpleScalar.",
        "implementation": "4x4 array, heterogeneous FUs, instruction buffers, SPM and multiple mesh networks.",
        "relationship": "Strong evidence for the simulator methodology and transfer subsystem inherited by later DFUs.",
    },
    {
        "id": "multi_batch_dfu_2024",
        "year": 2024,
        "title": "Improving Utilization of Dataflow Unit for Multi-Batch Processing",
        "doi": "10.1145/3637906",
        "role": "m2_dfu_architecture_predecessor",
        "sources": ["openalex_multi_batch_dfu", "ucas_wenming"],
        "simulator": "Public metadata does not name the simulator; it belongs to the same DPU evaluation line.",
        "implementation": "Unified scale-vector modes, reconfigurable clusters, DFG-node pipeline stages and task model.",
        "relationship": "High-specificity conceptual predecessor to M2-DFU's multi-mode architecture.",
    },
    {
        "id": "dfu_e_2025",
        "year": 2025,
        "title": "DFU-E: A Dataflow Architecture for Edge DSP and AI Applications",
        "doi": "10.1109/TPDS.2025.3555329",
        "role": "parent_family_candidate",
        "sources": ["openalex_dfu_e", "ucas_wenming", "ict_xiaochun"],
        "simulator": "Full text unavailable in the frozen sources; exact simulator configuration remains unreported.",
        "implementation": "Multi-layer task/block/instruction/data parallelism, customized PE/memory/NoC and software stack.",
        "relationship": "Very strong architecture-family candidate with extensive MLX author overlap; no explicit derivation link.",
    },
    {
        "id": "panda_2025",
        "year": 2025,
        "title": "PANDA: Adaptive Prefetching and Decentralized Scheduling for Dataflow Architectures",
        "doi": "10.1145/3721288",
        "role": "memory_scheduler_precedent",
        "sources": ["openalex_panda", "zhihua_home"],
        "simulator": "Simulator name is not present in retained primary metadata.",
        "implementation": "Adaptive prefetch/on-chip memory plus decentralized PE scheduling.",
        "relationship": "Likely relevant to MLX memory backpressure; not a parent claim.",
    },
    {
        "id": "dfgas_2025",
        "year": 2025,
        "title": "DFGAS: Exploring the Balance of HW-SW Scheduling through the DFG-Aware Scheme",
        "doi": "10.1145/3773768",
        "role": "scheduler_precedent",
        "sources": ["openalex_dfgas", "zhihua_home"],
        "simulator": "Publisher full text is unavailable; no source-qualified simulator name recovered.",
        "implementation": "DFG-aware hardware/software scheduling in the same DFU line.",
        "relationship": "Likely scheduling ancestry, but not an explicit MLX fork or parent.",
    },
    {
        "id": "m2_dfu_2026",
        "year": 2026,
        "title": "M2-DFU: Multi-Mode Dataflow Architecture for Adaptive and High-Efficiency Data Processing",
        "role": "highest_ranked_general_parent_candidate",
        "sources": ["ucas_wenming", "zhihua_home"],
        "simulator": "Just-accepted record has no accessible full text or simulator disclosure.",
        "implementation": "Official author records identify a general multi-mode DFU with the core MLX hardware authors.",
        "relationship": "Best title/chronology/author match for MLX's unnamed general-purpose design, but explicit lineage remains absent.",
    },
    {
        "id": "smarco_internal_stack",
        "year": 2025,
        "title": "SmarCo Processor Architecture Group engineering disclosure",
        "role": "internal_engineering_stack_evidence",
        "sources": ["shantian_home", "ucas_wenming_chip"],
        "simulator": "Official personal page names PE, on-chip memory and data-transfer simulator optimization.",
        "implementation": "Same group reports SPM/cache RTL and runtime scheduling; DPU-s/HTC-3000/HTC-3500 chips are listed.",
        "relationship": "Strong evidence that the current DFU simulator/RTL/runtime stack is internal to ICT/Ricore/SmarCo.",
    },
    {
        "id": "debug_patent_2020",
        "year": 2020,
        "title": "Debugging Method and Device for a Coarse-Grained Dataflow Execution Array",
        "role": "block_instance_execution_evidence",
        "sources": ["patent_debug_array"],
        "simulator": "Patent is hardware debug rather than performance simulation.",
        "implementation": "PE array executes program blocks identified by task ID, block ID and repeated instance.",
        "relationship": "Precedent for MLX block/tag/instance state, without proving shared RTL.",
    },
    {
        "id": "jian_weng_open_stack",
        "year": 2025,
        "title": "DSAGEN / OverGen / Assassyn full-stack spatial-accelerator line",
        "role": "open_engineering_precedent",
        "sources": ["kaust_weng", "weng_home"],
        "simulator": "DSAGEN supplies gem5 timing; Assassyn unifies asynchronous simulation and RTL.",
        "implementation": "Public compiler, generation and simulation methods from MLX coauthor Jian Weng.",
        "relationship": "Best open second-development reference, but no primary evidence that ICT's MLX simulator forks it.",
    },
]


def digest(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    data = path.read_bytes() if exists else b""
    return {
        "path": str(path.relative_to(PROJECT_ROOT)) if exists else str(path),
        "bytes": len(data) if exists else None,
        "sha256": hashlib.sha256(data).hexdigest() if exists else None,
        "is_file": exists,
    }


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    qualified = {}
    for name, spec in config["frozen_inputs"].items():
        item = digest(PROJECT_ROOT / spec["path"])
        checks = {
            "is_file": item["is_file"],
            "sha256": item["sha256"] == spec["sha256"],
        }
        if "bytes" in spec:
            checks["bytes"] = item["bytes"] == int(spec["bytes"])
        item.update({"checks": checks, "pass": all(checks.values())})
        qualified[name] = item
    snapshot = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["source_snapshot"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    records = snapshot["records"]
    source_checks = {}
    for lineage in LINEAGE_RECORDS:
        checks = []
        for source in lineage["sources"]:
            checks.append(source in records and records[source]["transport_success"])
        source_checks[lineage["id"]] = all(checks)

    manuscript = (PROJECT_ROOT / config["frozen_inputs"]["manuscript"]["path"]).read_text(
        encoding="utf-8"
    )
    author_coverage = {
        item["name"]: item["name"] in manuscript for item in config["authors"]
    }
    manuscript_folded = manuscript.casefold()
    paper_simict = (
        "tuned in our simulator [36]" in manuscript_folded
        and "simict: a fast and flexible framework" in manuscript_folded
    )
    candidate_ranking = [
        {
            "rank": 1,
            "candidate": "ICT/Ricore DPU -> DFU-E -> M2-DFU family",
            "classification": "high_confidence_family_candidate",
            "basis": [
                "same institution and commercial chip line",
                "4x4/8x8 PE-SPM-multi-NoC SimICT models",
                "multi-layer and multi-mode DFU mechanisms",
                "matching simulator/RTL/runtime engineering disclosures",
            ],
            "explicit_mlx_derivation": False,
        },
        {
            "rank": 2,
            "candidate": "SimICT",
            "classification": "supported_simulation_framework_at_citation_level",
            "basis": [
                "MLX simulator citation resolves to SimICT",
                "team DPU papers repeatedly use cycle-accurate SimICT",
                "historical calibration against gem5 and RTL",
            ],
            "source_code_reuse_supported": False,
        },
        {
            "rank": 3,
            "candidate": "BDSim",
            "classification": "same_team_parallel_simulation_precedent",
            "basis": ["component-based configurable PDES publication"],
            "mlx_citation_or_derivation": False,
        },
        {
            "rank": 4,
            "candidate": "DSAGEN / Assassyn",
            "classification": "open_engineering_precedent",
            "basis": ["MLX coauthor's public spatial compiler/simulation stack"],
            "ict_mlx_fork_supported": False,
        },
    ]
    conclusions = {
        "exact_parent_chip": "unresolved",
        "explicit_architecture_derivation": False,
        "highest_ranked_parent_family": "ICT/Ricore DPU -> DFU-E -> M2-DFU",
        "parent_family_confidence": "high_candidate_not_proven",
        "simulator_framework": "SimICT",
        "simulator_framework_evidence": "supported_at_citation_and_historical_method_level",
        "simulator_source_code_reuse": "not_supported",
        "likely_recent_implementation_state": "closed_internal_SmarCo_DFU_simulator_RTL_runtime_stack",
        "best_open_second_development_substrate": "gem5 component model plus DSAGEN/Assassyn mechanisms and Accel-Sim GPU baselines",
        "bdsim_role": "same-team simulation precedent, not MLX-cited",
    }
    integrity_checks = {
        "frozen_inputs": all(item["pass"] for item in qualified.values()),
        "endpoint_count": snapshot["summary"]["endpoint_count"] == 25,
        "required_sources": snapshot["summary"]["required_transport_successes"]
        == snapshot["summary"]["required_count"]
        == 12,
        "minimum_primary_sources": snapshot["summary"]["primary_successes"]
        >= int(config["decision_gates"]["minimum_t1_primary_successes"]),
        "all_lineage_records_sourced": all(source_checks.values()),
        "all_authors_covered": len(author_coverage)
        == int(config["decision_gates"]["required_author_coverage"])
        and all(author_coverage.values()),
        "simict_citation": paper_simict,
        "explicit_lineage_not_inferred": conclusions["explicit_architecture_derivation"]
        is False,
        "source_code_reuse_not_inferred": conclusions["simulator_source_code_reuse"]
        == "not_supported",
        "exact_parent_not_inferred": conclusions["exact_parent_chip"] == "unresolved",
    }
    integrity = all(integrity_checks.values())
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "validation_eligible": config["validation_eligible"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_commit(),
        "hypothesis_status": "supported" if integrity else "inconclusive",
        "audit_integrity": integrity,
        "frozen_inputs": qualified,
        "source_snapshot_summary": snapshot["summary"],
        "author_coverage": author_coverage,
        "lineage_records": LINEAGE_RECORDS,
        "lineage_source_checks": source_checks,
        "candidate_ranking": candidate_ranking,
        "conclusions": conclusions,
        "integrity_checks": integrity_checks,
        "paper_performance_targets_consumed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = build_audit(config)
    output = PROJECT_ROOT / config["result_path"]
    if args.verify_existing:
        existing = json.loads(output.read_text(encoding="utf-8"))
        keys = (
            "hypothesis_status",
            "audit_integrity",
            "candidate_ranking",
            "conclusions",
            "integrity_checks",
        )
        matches = all(existing.get(key) == report.get(key) for key in keys)
        print(json.dumps({"existing_matches": matches, **report}, indent=2))
        return 0 if matches else 1
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["hypothesis_status"],
                "integrity": report["audit_integrity"],
                "records": len(report["lineage_records"]),
                "conclusions": report["conclusions"],
            },
            indent=2,
        )
    )
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
