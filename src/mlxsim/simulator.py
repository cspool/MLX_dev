from __future__ import annotations

import heapq
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from .schema import (
    CalibrationConfig,
    HardwareConfig,
    KernelProfile,
    SimulationResult,
    StageSpec,
    Workload,
)
from .workloads import compile_workload


@dataclass
class _Operation:
    operation_id: int
    resource: str
    duration: float
    tag: int
    wave: int
    name: str
    dependencies: int = 0
    dependents: list[int] = field(default_factory=list)


@dataclass
class _Schedule:
    cycles: float
    busy: dict[str, float]
    event_count: int
    trace: list[dict[str, Any]]


class MLXSimulator:
    """Cycle/event surrogate for MLX's tagged, decoupled PE pipelines.

    The model is deliberately block-level: each operation represents an aggregate
    set of identical CDC instances. This preserves dependencies, resource
    contention, fill/drain effects, and tag arbitration without simulating every
    SIMD lane value.
    """

    def __init__(
        self,
        hardware: HardwareConfig,
        calibration: CalibrationConfig | None = None,
        *,
        trace_limit: int = 0,
    ):
        self.hardware = hardware
        self.calibration = calibration or CalibrationConfig()
        self.trace_limit = max(0, trace_limit)

    def simulate(self, workload: Workload) -> SimulationResult:
        profile = compile_workload(workload)
        return self.simulate_profile(workload, profile)

    def simulate_profile(
        self, workload: Workload, profile: KernelProfile
    ) -> SimulationResult:
        """Simulate a precompiled profile using the normal scheduler and accounting."""

        waves = self._wave_count(profile)
        operations = self._build_operations(profile, workload, waves)
        schedule = self._run_scheduler(operations)
        cycles = schedule.cycles + self.hardware.launch_cycles

        busy = dict(schedule.busy)
        utilization = {name: value / cycles for name, value in sorted(busy.items())}
        compute_names = [name for name in busy if name.startswith("compute")]
        if "unified" in busy:
            compute_active_fraction = min(1.0, busy["unified"] / cycles)
        elif compute_names:
            compute_active_fraction = sum(busy[name] for name in compute_names) / (
                cycles * len(compute_names)
            )
        else:
            compute_active_fraction = 0.0
        compute_utilization = min(
            1.0, profile.operations / (cycles * self.hardware.peak_ops_per_cycle)
        )

        data_names = [name for name in ("load", "store", "xfer") if name in busy]
        data_utilization = (
            sum(busy[name] for name in data_names) / (cycles * len(data_names))
            if data_names
            else 0.0
        )
        activity = min(1.0, 0.8 * compute_active_fraction + 0.2 * data_utilization)
        idle = self.hardware.idle_power_fraction
        core_power = self.hardware.core_power_w * (idle + (1.0 - idle) * activity)
        memory_power = self.hardware.memory_power_w * (idle + (1.0 - idle) * data_utilization)
        average_power = core_power + memory_power

        latency_us = cycles / (self.hardware.frequency_ghz * 1000.0)
        seconds = latency_us * 1e-6
        achieved_gops = profile.operations / seconds / 1e9 if seconds else math.inf
        energy_mj = average_power * seconds * 1000.0

        return SimulationResult(
            hardware=self.hardware.name,
            workload=workload.label,
            cycles=cycles,
            latency_us=latency_us,
            operations=profile.operations,
            offchip_bytes=profile.offchip_bytes,
            achieved_gops=achieved_gops,
            average_power_w=average_power,
            energy_mj=energy_mj,
            resource_busy_cycles=busy,
            resource_utilization=utilization,
            compute_utilization=compute_utilization,
            waves=waves,
            event_count=schedule.event_count,
            metadata={
                **profile.metadata,
                "hardware_peak_tops": self.hardware.peak_tops,
                "operational_intensity": profile.operations / max(profile.offchip_bytes, 1.0),
                "launch_cycles": self.hardware.launch_cycles,
                "compute_active_fraction": compute_active_fraction,
                "calibration": self.calibration.to_dict(),
                "trace": schedule.trace,
            },
        )

    def _wave_count(self, profile: KernelProfile) -> int:
        elements_per_wave = self.hardware.pe_count * self.hardware.simd_width
        raw = max(1, math.ceil(profile.output_elements / elements_per_wave))
        return min(raw, self.hardware.max_event_waves)

    def _resource_name(self, name: str) -> str:
        if not self.hardware.decoupled_pipelines:
            return "unified"
        if name.startswith("compute") and not self.hardware.heterogeneous_compute:
            return "compute"
        return name

    def _compute_capacity(self, resource: str, stage: StageSpec, workload: Workload) -> float:
        base = self.hardware.peak_ops_per_cycle * self.hardware.compute_issue_efficiency
        if resource == "compute_fmax":
            base /= self.hardware.ops_per_lane_cycle
        elif resource == "compute_fexp":
            base /= 4.0 * self.hardware.ops_per_lane_cycle
        return (
            base
            * self.calibration.issue_scale(stage.kernel_class)
            * self.calibration.mesh_efficiency(self.hardware, workload.n)
        )

    def _route_hops(self, stage: StageSpec) -> int:
        if stage.route_distance <= 0:
            return 0
        if self.hardware.skip_distance <= 1:
            return min(stage.route_distance, max(1, self.hardware.max_skip_hops))
        hops = math.ceil(stage.route_distance / self.hardware.skip_distance)
        return max(1, min(hops, self.hardware.max_skip_hops))

    def _durations(
        self, stage: StageSpec, workload: Workload, waves: int
    ) -> list[tuple[str, float, str]]:
        result: list[tuple[str, float, str]] = []
        if stage.load_bytes > 0:
            duration = self.hardware.load_latency_cycles + (
                stage.load_bytes / waves / self.hardware.spm_bytes_per_cycle
            )
            result.append((self._resource_name("load"), duration, f"{stage.name}:load"))

        capacity = self._compute_capacity(stage.compute_resource, stage, workload)
        setup_cycles = self.calibration.compute_setup_cycles(
            stage.kernel_class, self.hardware.compute_latency_cycles
        )
        compute_duration = setup_cycles + stage.operations / waves / capacity
        result.append(
            (
                self._resource_name(stage.compute_resource),
                compute_duration,
                f"{stage.name}:compute",
            )
        )

        if stage.transfer_bytes > 0:
            hops = self._route_hops(stage)
            duration = self.hardware.xfer_latency_cycles * hops + (
                stage.transfer_bytes / waves / self.hardware.noc_bytes_per_cycle
            )
            result.append((self._resource_name("xfer"), duration, f"{stage.name}:xfer"))

        if stage.store_bytes > 0:
            duration = self.hardware.store_latency_cycles + (
                stage.store_bytes / waves / self.hardware.spm_bytes_per_cycle
            )
            result.append((self._resource_name("store"), duration, f"{stage.name}:store"))
        return result

    def _build_operations(
        self, profile: KernelProfile, workload: Workload, waves: int
    ) -> list[_Operation]:
        operations: list[_Operation] = []

        def add_operation(resource: str, duration: float, tag: int, wave: int, name: str) -> int:
            operation_id = len(operations)
            operations.append(
                _Operation(
                    operation_id=operation_id,
                    resource=resource,
                    duration=max(duration, 1e-9),
                    tag=tag,
                    wave=wave,
                    name=name,
                )
            )
            return operation_id

        def add_dependency(parent: int, child: int) -> None:
            operations[parent].dependents.append(child)
            operations[child].dependencies += 1

        for wave in range(waves):
            prior_block_tail: int | None = None
            for stage in profile.stages:
                block_head: int | None = None
                block_tail: int | None = None
                for resource, duration, name in self._durations(stage, workload, waves):
                    operation_id = add_operation(resource, duration, stage.tag, wave, name)
                    if block_head is None:
                        block_head = operation_id
                    if block_tail is not None:
                        add_dependency(block_tail, operation_id)
                    block_tail = operation_id
                if block_head is None or block_tail is None:
                    raise AssertionError(f"stage {stage.name} compiled to no operations")
                if prior_block_tail is not None:
                    add_dependency(prior_block_tail, block_head)
                prior_block_tail = block_tail
        return operations

    def _run_scheduler(self, operations: list[_Operation]) -> _Schedule:
        ready: dict[str, dict[int, deque[int]]] = defaultdict(lambda: defaultdict(deque))
        remaining_by_tag: dict[int, int] = defaultdict(int)
        for operation in operations:
            remaining_by_tag[operation.tag] += 1
            if operation.dependencies == 0:
                ready[operation.resource][operation.tag].append(operation.operation_id)

        resources = sorted({operation.resource for operation in operations})
        resource_busy_until = {resource: 0.0 for resource in resources}
        resource_running: dict[str, int | None] = {resource: None for resource in resources}
        resource_last_tag = {resource: -1 for resource in resources}
        busy: dict[str, float] = defaultdict(float)
        completions: list[tuple[float, int, str]] = []
        trace: list[dict[str, Any]] = []
        completed = 0
        current_time = 0.0
        event_count = 0

        def minimum_incomplete_tag() -> int:
            return min(tag for tag, count in remaining_by_tag.items() if count > 0)

        def pop_ready(resource: str) -> int | None:
            if completed == len(operations):
                return None
            minimum = minimum_incomplete_tag()
            maximum = minimum + self.hardware.active_tags - 1
            candidates = [
                tag for tag, queue in ready[resource].items() if queue and minimum <= tag <= maximum
            ]
            if not candidates:
                return None
            last = resource_last_tag[resource]
            after = sorted(tag for tag in candidates if tag > last)
            tag = after[0] if after else min(candidates)
            resource_last_tag[resource] = tag
            return ready[resource][tag].popleft()

        while completed < len(operations):
            dispatched = False
            for resource in resources:
                if resource_running[resource] is not None:
                    continue
                operation_id = pop_ready(resource)
                if operation_id is None:
                    continue
                operation = operations[operation_id]
                start = max(current_time, resource_busy_until[resource])
                end = start + operation.duration
                resource_running[resource] = operation_id
                resource_busy_until[resource] = end
                busy[resource] += operation.duration
                heapq.heappush(completions, (end, operation_id, resource))
                if len(trace) < self.trace_limit:
                    trace.append(
                        {
                            "event": "dispatch",
                            "operation": operation.name,
                            "resource": resource,
                            "tag": operation.tag,
                            "wave": operation.wave,
                            "start": start,
                            "end": end,
                        }
                    )
                dispatched = True

            if dispatched:
                continue
            if not completions:
                raise RuntimeError("scheduler deadlock: no ready or running operations")

            current_time = completions[0][0]
            same_time: list[tuple[float, int, str]] = []
            while completions and math.isclose(completions[0][0], current_time, abs_tol=1e-12):
                same_time.append(heapq.heappop(completions))
            for _, operation_id, resource in same_time:
                operation = operations[operation_id]
                resource_running[resource] = None
                completed += 1
                event_count += 1
                remaining_by_tag[operation.tag] -= 1
                for dependent_id in operation.dependents:
                    dependent = operations[dependent_id]
                    dependent.dependencies -= 1
                    if dependent.dependencies == 0:
                        ready[dependent.resource][dependent.tag].append(dependent_id)
                if len(trace) < self.trace_limit:
                    trace.append(
                        {
                            "event": "complete",
                            "operation": operation.name,
                            "resource": resource,
                            "tag": operation.tag,
                            "wave": operation.wave,
                            "time": current_time,
                        }
                    )

        return _Schedule(
            cycles=current_time,
            busy=dict(busy),
            event_count=event_count,
            trace=trace,
        )
