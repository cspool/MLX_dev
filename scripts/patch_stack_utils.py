"""Helpers for exact replay of the overlay latency/physical patch boundary."""

from __future__ import annotations

from pathlib import Path

_SINGLE_SEPARATOR = (
    b'    }\n  }\n\n  const auto &functional = root["functional_execution"];'
)
_DOUBLE_SEPARATOR = (
    b'    }\n  }\n\n\n  const auto &functional = root["functional_execution"];'
)
_SINGLE_SUMMARY_SEPARATOR = b'        << "}";\n  }\n  out\n'
_DOUBLE_SUMMARY_SEPARATOR = b'        << "}";\n  }\n\n  out\n'


def _rewrite_boundaries(
    payload: bytes, boundaries: tuple[tuple[bytes, bytes], ...]
) -> bytes | None:
    for before, after in boundaries:
        if payload.count(before) == 1:
            payload = payload.replace(before, after, 1)
        elif payload.count(after) != 1:
            return None
    return payload


def collapse_physical_latency_separator(source: Path) -> bool:
    """Restore the one-line boundaries expected by the latency patch."""
    payload = source.read_bytes()
    rewritten = _rewrite_boundaries(
        payload,
        (
            (_DOUBLE_SEPARATOR, _SINGLE_SEPARATOR),
            (_DOUBLE_SUMMARY_SEPARATOR, _SINGLE_SUMMARY_SEPARATOR),
        ),
    )
    if rewritten is None:
        return False
    source.write_bytes(rewritten)
    return True


def restore_physical_latency_separator(source: Path) -> bool:
    """Restore the two-line preimages expected by the physical patch."""
    payload = source.read_bytes()
    rewritten = _rewrite_boundaries(
        payload,
        (
            (_SINGLE_SEPARATOR, _DOUBLE_SEPARATOR),
            (_SINGLE_SUMMARY_SEPARATOR, _DOUBLE_SUMMARY_SEPARATOR),
        ),
    )
    if rewritten is None:
        return False
    source.write_bytes(rewritten)
    return True
