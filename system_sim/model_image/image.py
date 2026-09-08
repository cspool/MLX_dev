"""Compose explicit initial RAM assets and independently checked result data."""
import hashlib
from pathlib import Path

from system_sim.physical_host.asset_source import source_manifest


def result_reference(value):
    if "outputs" in value:
        if not value["outputs"] or len({row["forward_id"] for row in value["outputs"]})!=len(value["outputs"]):raise ValueError("empty/duplicate forward result reference")
        return value
    if "reference_checks" not in value:
        raise ValueError("result reference is neither a result file nor a reference inventory")
    outputs=[]
    for row in value["reference_checks"]:
        path=Path(row["logits_file"])
        with path.open("rb") as source:digest=hashlib.file_digest(source,"sha256").hexdigest()
        if digest!=row["logits_sha256"]:raise ValueError("reference logits file fingerprint mismatch")
        if path.suffix!=".bin" or ".f16." not in path.name:raise ValueError("inventory result reference requires registered f16 logits")
        outputs.append({"forward_id":row["forward_id"],"dtype":"f16","shape":row["logits_shape"],"tokens":[row["token_id"]],"logits_file":str(path.resolve())})
    if not outputs or len({r["forward_id"] for r in outputs})!=len(outputs):raise ValueError("empty/duplicate forward result reference")
    return {"outputs":outputs}


def initial_segments(program,plan,directory):
    entries,manifest=source_manifest(program,directory)
    by_name={row["value"]:row for row in entries};segments=[]
    for row in manifest["regions"]:
        name=row["name"];binding=plan["assets"][name]
        if binding["bytes"]!=row["bytes"]:raise ValueError("initial asset extent differs from graph binding")
        segments.append({"name":name,"path":row["path"],"file_offset":row["file_offset"],"file_bytes":row["bytes"],"memory_bytes":row["bytes"],"address":binding["base"],"sha256":by_name[name]["sha256"]})
    return segments
