"""Target-free two-resource compute/DMA overlap model for H108."""

from __future__ import annotations

import heapq
import math
from collections import deque
from typing import Any


def balanced_integer(total: int, count: int) -> list[int]:
    if total <= 0 or count <= 0:
        raise ValueError("balanced partition dimensions must be positive")
    quotient, remainder = divmod(total, count)
    if quotient == 0:
        raise ValueError("partition count exceeds total cycles")
    return [quotient + (index < remainder) for index in range(count)]


def balanced_aligned(total: int, count: int, alignment: int = 32) -> list[int]:
    if total <= 0 or count <= 0 or alignment <= 0 or total % alignment:
        raise ValueError("aligned partition dimensions are invalid")
    units, remainder = divmod(total // alignment, count)
    if units == 0:
        raise ValueError("partition count exceeds aligned byte units")
    return [
        (units + (index < remainder)) * alignment for index in range(count)
    ]


def _overlap(intervals_a: list[tuple[int, int]], intervals_b: list[tuple[int, int]]) -> int:
    left = right = 0
    total = 0
    while left < len(intervals_a) and right < len(intervals_b):
        start = max(intervals_a[left][0], intervals_b[right][0])
        end = min(intervals_a[left][1], intervals_b[right][1])
        total += max(0, end - start)
        if intervals_a[left][1] <= intervals_b[right][1]:
            left += 1
        else:
            right += 1
    return total


def simulate_overlap(
    *,
    compute_cycles_by_tile: list[int],
    input_bytes_by_tile: list[int],
    output_bytes_by_tile: list[int],
    bandwidth_bytes_per_cycle: int,
) -> dict[str, Any]:
    tile_count = len(compute_cycles_by_tile)
    if (
        tile_count == 0
        or len(input_bytes_by_tile) != tile_count
        or len(output_bytes_by_tile) != tile_count
        or bandwidth_bytes_per_cycle <= 0
    ):
        raise ValueError("overlap schedule dimensions are inconsistent")
    fill_duration = [
        math.ceil(value / bandwidth_bytes_per_cycle)
        for value in input_bytes_by_tile
    ]
    drain_duration = [
        math.ceil(value / bandwidth_bytes_per_cycle)
        for value in output_bytes_by_tile
    ]
    ready = [False] * tile_count
    computed = [False] * tile_count
    drained = [False] * tile_count
    fill_end: list[int | None] = [None] * tile_count
    compute_start: list[int | None] = [None] * tile_count
    compute_end: list[int | None] = [None] * tile_count
    drain_start: list[int | None] = [None] * tile_count
    drain_end: list[int | None] = [None] * tile_count
    dma_queue: deque[tuple[str, int]] = deque(
        [("fill", tile) for tile in range(min(2, tile_count))]
    )
    events: list[tuple[int, int, int, str, str, int]] = []
    dma_intervals: list[tuple[int, int]] = []
    compute_intervals: list[tuple[int, int]] = []
    dma_busy = False
    compute_busy = False
    next_compute = 0
    sequence = 0
    time = 0

    def launch(now: int) -> None:
        nonlocal dma_busy, compute_busy, sequence
        if not dma_busy and dma_queue:
            kind, tile = dma_queue.popleft()
            duration = (
                fill_duration[tile] if kind == "fill" else drain_duration[tile]
            )
            finish = now + duration
            if kind == "drain":
                drain_start[tile] = now
            dma_intervals.append((now, finish))
            heapq.heappush(
                events, (finish, 0, sequence, "dma", kind, tile)
            )
            sequence += 1
            dma_busy = True
        if (
            not compute_busy
            and next_compute < tile_count
            and ready[next_compute]
        ):
            tile = next_compute
            finish = now + compute_cycles_by_tile[tile]
            compute_start[tile] = now
            compute_intervals.append((now, finish))
            heapq.heappush(
                events, (finish, 1, sequence, "compute", "compute", tile)
            )
            sequence += 1
            compute_busy = True

    launch(0)
    while not all(drained):
        if not events:
            raise RuntimeError("compute/DMA overlap schedule deadlocked")
        time = events[0][0]
        completions = []
        while events and events[0][0] == time:
            completions.append(heapq.heappop(events))
        for _, _, _, resource, kind, tile in completions:
            if resource == "dma":
                dma_busy = False
                if kind == "fill":
                    if ready[tile] or fill_end[tile] is not None:
                        raise RuntimeError("tile filled more than once")
                    ready[tile] = True
                    fill_end[tile] = time
                else:
                    if not computed[tile] or drained[tile]:
                        raise RuntimeError("tile drained outside dependency order")
                    drained[tile] = True
                    drain_end[tile] = time
                    next_tile = tile + 2
                    if next_tile < tile_count:
                        dma_queue.append(("fill", next_tile))
            else:
                compute_busy = False
                if tile != next_compute or not ready[tile] or computed[tile]:
                    raise RuntimeError("tile computed outside ascending order")
                computed[tile] = True
                compute_end[tile] = time
                dma_queue.append(("drain", tile))
                next_compute += 1
        launch(time)

    compute_work = sum(compute_cycles_by_tile)
    dma_work = sum(fill_duration) + sum(drain_duration)
    serial_cycles = compute_work + dma_work
    ideal_cycles = max(compute_work, dma_work)
    dependencies = all(
        fill_end[tile] is not None
        and compute_start[tile] is not None
        and compute_end[tile] is not None
        and drain_start[tile] is not None
        and drain_end[tile] is not None
        and fill_end[tile] <= compute_start[tile]
        and compute_start[tile] < compute_end[tile]
        and compute_end[tile] <= drain_start[tile]
        and drain_start[tile] < drain_end[tile]
        for tile in range(tile_count)
    )
    dma_nonoverlap = all(
        dma_intervals[index][1] <= dma_intervals[index + 1][0]
        for index in range(len(dma_intervals) - 1)
    )
    compute_nonoverlap = all(
        compute_intervals[index][1] <= compute_intervals[index + 1][0]
        for index in range(len(compute_intervals) - 1)
    )
    return {
        "tile_count": tile_count,
        "compute_cycles": compute_work,
        "dma_cycles": dma_work,
        "ideal_cycles": ideal_cycles,
        "pipeline_cycles": time,
        "serial_cycles": serial_cycles,
        "overlap_cycles": _overlap(dma_intervals, compute_intervals),
        "fill_count": sum(value is not None for value in fill_end),
        "compute_count": sum(computed),
        "drain_count": sum(drained),
        "checks": {
            "dependencies": dependencies,
            "dma_nonoverlap": dma_nonoverlap,
            "compute_nonoverlap": compute_nonoverlap,
            "counts": all(
                value == tile_count
                for value in (
                    sum(value is not None for value in fill_end),
                    sum(computed),
                    sum(drained),
                )
            ),
            "bounds": ideal_cycles <= time <= serial_cycles,
        },
    }


def compose_point(
    *,
    key: str,
    h102_cycles: int,
    path: dict[str, Any],
    bandwidth: int,
    peak_effective_ops_per_cycle: float,
) -> dict[str, Any]:
    tile_count = int(path["tile_count"])
    compute_by_tile = balanced_integer(h102_cycles, tile_count)
    input_by_tile = balanced_aligned(
        int(path["selected_read_bytes"]), tile_count
    )
    output_by_tile = balanced_aligned(
        int(path["selected_write_bytes"]), tile_count
    )
    schedule = simulate_overlap(
        compute_cycles_by_tile=compute_by_tile,
        input_bytes_by_tile=input_by_tile,
        output_bytes_by_tile=output_by_tile,
        bandwidth_bytes_per_cycle=bandwidth,
    )
    effective_flops = int(path["effective_flops"])
    oi = float(path["selected_oi_flop_per_byte"])
    roof = min(peak_effective_ops_per_cycle, oi * bandwidth)
    throughput = {
        name: effective_flops / schedule[f"{name}_cycles"]
        for name in ("serial", "pipeline", "ideal")
    }
    utilization = {name: value / roof for name, value in throughput.items()}
    selected_mlx_bandwidth = None
    figure25_reproduction = None
    return {
        "key": key,
        "family": path["family"],
        "bandwidth_bytes_per_cycle": bandwidth,
        "bandwidth_classification": (
            "historical_dpu_sensitivity"
            if bandwidth == 64
            else "power_of_two_sensitivity"
        ),
        "tile_count": tile_count,
        "effective_flops": effective_flops,
        "operational_intensity": oi,
        "peak_effective_ops_per_cycle": peak_effective_ops_per_cycle,
        "roofline_denominator_ops_per_cycle": roof,
        "compute_cycles_by_tile": compute_by_tile,
        "schedule": schedule,
        "throughput_effective_ops_per_cycle": throughput,
        "roofline_utilization_sensitivity": utilization,
        "selected_mlx_bandwidth_bytes_per_cycle": selected_mlx_bandwidth,
        "figure25_reproduction": figure25_reproduction,
        "checks": {
            "compute_sum": sum(compute_by_tile) == h102_cycles,
            "oi_invariant_source": math.isfinite(oi) and oi > 0,
            "throughput_order": (
                throughput["serial"]
                <= throughput["pipeline"]
                <= throughput["ideal"]
            ),
            "utilization_order": (
                utilization["serial"]
                <= utilization["pipeline"]
                <= utilization["ideal"]
            ),
            "utilization_bounded": all(
                math.isfinite(value) and 0 < value <= 1 for value in utilization.values()
            ),
            "roofline": roof
            == min(peak_effective_ops_per_cycle, oi * bandwidth),
            "schedule": all(schedule["checks"].values()),
            "null_selection": selected_mlx_bandwidth is None
            and figure25_reproduction is None,
        },
    }


__all__ = [
    "balanced_aligned",
    "balanced_integer",
    "compose_point",
    "simulate_overlap",
]
