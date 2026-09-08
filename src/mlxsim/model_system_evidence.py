"""Audit completed Rocket/C++ task execution without advancing the simulator.

These checks are necessary, not sufficient, for model acceptance. The caller
must also bind the executable, compiler input, initialization and CPU checks.
"""

from collections import Counter
import math


WIDTH = {"f16": 2, "f32": 4, "i64": 8, "bool": 1}
WIRE_BYTES = {"matrix": 1088, "vector": 4288, "memory": 15872, "control": 640}
ADAPTER_CLASS = "clocked_cpp_rocc_requestor_adapter_execution_scope_requires_external_evidence"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def count(value, name):
    require(type(value) is int and 0 <= value < 2**64, f"invalid counter: {name}")
    return value


def terminal_execution(state, log):
    require(state.get("status") == "exited" and type(state.get("exit_code")) is int
            and state["exit_code"] == 0, "actual system process has no successful terminal result")
    require(state.get("validation") == "registered_graph_checks_passed",
            "runner has not validated the terminal graph")
    require(log.splitlines().count("MLX_CLOCKED_CHAIN_PASS") == 1,
            "CPU completion/output checks lack a unique terminal marker")


def matrix_work(node, values):
    shape = node["output"]["shape"]
    require(node["kind"] in {"linear", "matmul"} and len(shape) >= 2,
            "unregistered matrix execution geometry")
    k = values[node["args"][0]["value"]]["shape"][-1]
    m = math.prod(shape[:-1]) if node["kind"] == "linear" else shape[-2]
    batches = 1 if node["kind"] == "linear" else math.prod(shape[:-2])
    require(batches > 0, "empty matrix batches cannot complete a source")
    return m, shape[-1], k, batches


def task_coverage(program, plan):
    nodes = program["nodes"]
    require(nodes and len({n["id"] for n in nodes}) == len(nodes)
            and len({n["source_operator_id"] for n in nodes}) == len(nodes),
            "source/node IDs are empty or duplicated")
    require(plan["source_calls"] == len(nodes), "source completion scope is incomplete")
    values = {name: asset for name, asset in program["assets"].items()}
    values.update({node["id"]: node["output"] for node in nodes})
    tasks, sources, counts = [], [], {name: 0 for name in (*WIRE_BYTES, "view")}
    offset = 0
    for ordinal, node in enumerate(nodes):
        routes = [family for family in WIRE_BYTES if family + "_program" in node]
        require(len(routes) == 1, "source has missing or ambiguous backend lowering")
        family = routes[0]
        view = family == "memory" and node["memory_program"]["mode"] == "view"
        batches = matrix_work(node, values)[3] if family == "matrix" else 1
        counts[family] += 1
        counts["view"] += int(view)
        sources.append({"source_ordinal": ordinal, "source_operator_id": node["source_operator_id"],
                        "kind": node["kind"], "family": family, "view_elided": view,
                        "task_count": batches, "forward_id": node["forward_id"],
                        "layer_idx": node["layer_idx"]})
        for batch in range(batches):
            size = 0 if view else WIRE_BYTES[family]
            tasks.append({"kind": 0 if view else 1 if family == "control" else 2,
                          "source_ordinal": ordinal, "source_id": node["source_operator_id"],
                          "batch_index": batch, "batch_count": batches,
                          "command_offset": offset if size else 0, "bytes": size, "family": family})
            offset += size
    require(plan["tasks"] == tasks and plan["sources"] == sources,
            "compiled source/batch routing is incomplete, duplicated or reordered")
    require(plan["family_source_calls"] == counts and plan["task_count"] == len(tasks)
            and plan["command_bytes"] == offset, "task/source/command accounting differs")
    outputs = [(o["forward_id"], role, o[role]) for o in program["outputs"] for role in ("logits", "token")]
    require([(o["forward_id"], o["role"], o["value"]) for o in plan["outputs"]] == outputs,
            "compiled final outputs are missing or duplicated")
    return [t for t in tasks if t["kind"] == 2], values, counts


def check_kernel(node, task, kernel, profile, values):
    family = task["family"]
    require(kernel["done"] is True and kernel["external_memory_port"] is True,
            "backend did not finish through actual external responses")
    requests = count(kernel["dma_requests"], "kernel requests")
    require(requests == kernel["dma_responses"], "backend requests have not drained")
    numeric = kernel["numeric_instructions"]
    require(numeric["calls"] == 1, "window did not execute exactly one backend call")
    elements = math.prod(node["output"]["shape"]) // task["batch_count"]
    expected_bytes = elements * WIDTH[node["output"]["dtype"]]
    writes = numeric["write_bytes" if family == "memory" else "global_write_bytes"]
    reads = numeric["read_bytes" if family == "memory" else "global_read_bytes"]
    count(reads, "numeric read bytes")
    require(writes == expected_bytes, "backend output stores do not cover the actual source/batch")
    macs = 0
    if family in {"matrix", "vector"}:
        require(kernel["admitted"] == kernel["retired"], "resident contexts did not retire")
        for key in ("pending_dma", "pending_spm", "active_contexts", "allocated_spm_vectors",
                    "pending_compute" if family == "matrix" else "pending_fu"):
            require(kernel[key] == 0, f"backend still owns {key}")
        options = profile[family + "_options"]
        require(0 <= kernel["peak_contexts"] <= options["rows"] * options["columns"] * options["contexts"],
                "backend exceeded physical context capacity")
        require(kernel["spm_capacity_vectors"] == 128 and 0 <= kernel["peak_spm_vectors"] <= 128,
                "backend changed or exceeded SPM capacity")
        require(kernel["rf_frame_vectors"] == (6 if family == "matrix" else 8)
                and 0 <= numeric["max_rom_words"] <= 32, "RF/ROM budget changed")
    if family == "matrix":
        m, n, k, _ = matrix_work(node, values)
        p = node["matrix_program"]
        tiles = ((m + p["tile_m"] - 1) // p["tile_m"]) * ((n + p["tile_n"] - 1) // p["tile_n"])
        macs = m * n * k
        require(kernel["admitted"] == numeric["output_tiles"] == tiles,
                "matrix did not execute every output tile")
        require(numeric["mul_active_lanes"] == macs and kernel["blas_calls"] == 0,
                "matrix MAC work is missing or used BLAS")
        require(kernel["dma_read_bytes"] == reads and kernel["dma_write_bytes"] == writes,
                "matrix numeric and physical byte counters disagree")
    elif family == "vector":
        require(kernel["spm_frame_vectors"] == 5 and numeric["rf_vectors_used"] == 8
                and numeric["spm_bytes_used"] <= 320, "vector local storage budget changed")
    else:
        require(family == "memory" and not kernel["view_elided"] and not kernel["inflight_transactions"],
                "device memory task was elided or still owns a transaction")
        require(numeric["allocations"] == 1 and numeric["view_elisions"] == 0
                and numeric["staging_bytes"] == 128 and numeric["register_bytes_total"] == 32,
                "memory transfer work/storage accounting differs")
        require(kernel["dma_read_bytes"] == reads and kernel["dma_write_bytes"] == writes,
                "memory numeric and physical byte counters disagree")
    return {"requests": requests, "read_bytes": reads, "write_bytes": writes, "matrix_mac_lanes": macs}


def verify_system_execution(program, plan, device, memory, profile):
    tasks, values, source_counts = task_coverage(program, plan)
    require(tasks, "system execution has no device tasks")
    require(device["classification"] == ADAPTER_CLASS
            and device["source_id_basis"] == "transport_launch_ordinal_not_model_operator_identity",
            "unknown system backend or ambiguous source identity")
    require(not device["frontend_error"] and not device["cache_request_owned"]
            and not device["cpu_response_pending"], "RoCC front end has errors or pending work")
    require(device["effective_profile"] == profile, "executed resource profile differs")
    require(device["launches"] == len(tasks) == len(device["windows"]),
            "not all compiled device windows actually executed")
    require(device["requests"] == device["responses"], "cache requests have not drained")
    observer = device.get("progress_observer")
    require(observer is None or not observer["failed"], "system observer reported an error")
    totals, per_source = Counter(), []
    previous_cycle, previous_requests = 0, 0
    for ordinal, (task, window) in enumerate(zip(tasks, device["windows"], strict=True)):
        require(window["source_id"] == ordinal and window["launches"] == ordinal + 1
                and window["backend"] == task["family"], "executed launch/source/backend mapping differs")
        require(window["done"] is True and window["complete"] is True
                and not window["busy"] and not window["error"], "device window is not successfully terminal")
        require(window["descriptor_bytes_fetched"] == task["bytes"], "descriptor was not fully fetched")
        start, end = count(window["launch_start_cycle"], "launch start"), count(window["cycle"], "window end")
        fetch, run, drain = [count(window[k + "_cycles"], k) for k in ("fetch", "run", "drain")]
        require(start >= previous_cycle and end == start + fetch + run + drain and not drain,
                "window clocks overlap, lose edges or include a failed drain")
        require(run == max(1, count(window["kernel"]["cycles"], "backend cycles")),
                "backend and external execution edges disagree")
        transport = window["transport"]
        require(transport["idle"] and not any(transport[k] for k in ("inflight", "queued", "response_pending")),
                "window transport is not idle")
        submitted = count(transport["submitted"], "transport requests")
        require(all(transport[k] == submitted for k in ("accepted", "responses", "consumed"))
                and transport["nacks"] == 0, "transport conservation or cache replay ownership differs")
        node = program["nodes"][task["source_ordinal"]]
        work = check_kernel(node, task, window["kernel"], profile, values)
        require(submitted - previous_requests == task["bytes"] // 8 + work["requests"],
                "descriptor/data request accounting omits or duplicates traffic")
        totals.update(work)
        totals.update(fetch_cycles=fetch, run_cycles=run, drain_cycles=drain)
        per_source.append({"launch_ordinal": ordinal, "source_operator_id": node["source_operator_id"],
                           "source_ordinal": task["source_ordinal"], "batch_index": task["batch_index"],
                           "batch_count": task["batch_count"], "backend": task["family"],
                           "forward_id": node["forward_id"], "layer_idx": node["layer_idx"],
                           "phase": node.get("phase"), "work": work})
        previous_cycle, previous_requests = end, submitted
    final = device["device"]
    require({k: v for k, v in final.items() if k != "cycle"}
            == {k: v for k, v in device["windows"][-1].items() if k != "cycle"},
            "final controller state differs from the last completed window")
    require(final["cycle"] >= previous_cycle and previous_requests == device["requests"],
            "final clock/request accounting differs")
    require(memory["classification"] == "wide_checked_axi_magic_memory_not_calibrated_dram_timing"
            and memory["base"] == 0x80000000 and memory["bytes"] == 16 * 2**30,
            "actual system RAM is not the registered 64-bit 16GiB mapping")
    require(memory["idle"] and memory["aw_requests"] == memory["write_responses"]
            and memory["cycle"] == final["cycle"], "system memory clock or outstanding writes differ")
    require(memory["max_read_address"] >= plan["device_base"], "graph addresses did not reach actual RAM")
    return {"classification": "checked_system_task_execution_not_standalone_model_certificate",
            "source_calls": len(program["nodes"]), "source_counts": source_counts,
            "device_windows": len(tasks), "system_requests": previous_requests,
            "system_clock_edges_at_exit": final["cycle"], "backend_totals": dict(totals),
            "executed_device_routes": per_source, "full_model_execution_verified": False,
            "mlx_system_verified": False, "inference_performance_eligible": False}
