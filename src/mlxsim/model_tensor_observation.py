"""Bounded, output-only numerical diagnostics, separate from executable assets."""

import hashlib
import json

import torch


class TensorObserver:
    def __init__(self, operator_ids, per_tensor_bytes=1024 * 1024, total_bytes=32 * 1024 * 1024):
        self.selected = set(operator_ids)
        if any(type(value) is not int or value < 0 for value in self.selected):
            raise ValueError("observation IDs must be nonnegative integers")
        self.per_tensor_bytes = per_tensor_bytes
        self.total_bytes = total_bytes
        self.bytes = 0
        self.snapshots = {}

    def __call__(self, event, value):
        identifier = event["operator_id"]
        if identifier not in self.selected:
            return
        if not isinstance(value, torch.Tensor):
            raise RuntimeError("numerical observation requires a single tensor output")
        size = value.numel() * value.element_size()
        if size > self.per_tensor_bytes or self.bytes + size > self.total_bytes:
            raise RuntimeError("numerical observation byte budget exceeded")
        if identifier in self.snapshots:
            raise RuntimeError("duplicate diagnostic operator ID")
        # Clone on the original device; perform no per-op CPU transfer/sync.
        self.snapshots[identifier] = (dict(event), value.detach().clone())
        self.bytes += size

    def save(self, directory):
        if set(self.snapshots) != self.selected:
            raise RuntimeError("some requested numerical observation IDs did not execute")
        directory.mkdir(parents=True, exist_ok=False)
        rows = []
        for identifier, (event, value) in sorted(self.snapshots.items()):
            path = directory / f"v{identifier}.bin"
            data = value.cpu().contiguous().numpy().tobytes()
            path.write_bytes(data)
            rows.append({
                "value_id": f"v{identifier}",
                "source_operator_id": identifier,
                "source_operator": event["operator"],
                "request_id": event["request_id"],
                "forward_id": event["forward_id"],
                "layer_idx": event["layer_idx"],
                "module_path": event["module_path"],
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "file": str(path.resolve()),
                "sha256": hashlib.sha256(data).hexdigest(),
            })
        report = {"classification": "reference_output_only_diagnostics", "bytes": self.bytes, "observations": rows}
        (directory / "observations.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
