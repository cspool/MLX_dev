import copy
import hashlib
import json

import pytest

from test_spike_graph_runtime import ROOT,graph,native
from scripts.run_mlx_clocked_chipyard import prepare,simulation_command
from system_sim.model_image.image import result_reference


def test_resident_image_contains_commands_not_an_embedded_weight_payload(graph,tmp_path):
    original,program,_=graph
    case={"program":str(original/"program.json"),"lifetimes":str(original/"life.json"),"reference":str(original/"out/result.json")}
    out=tmp_path/"image";plan,_=prepare(case,out,graph_base=0x88000000,graph_bytes=16*2**30-128*2**20,memory_bytes=16*2**30,preload_assets=True)
    image=json.loads((out/"image.json").read_text());segments=json.loads((out/"segments.json").read_text())
    assert plan["source_calls"]==45 and image["cpu_asset_copy_bytes"]==0
    assert set(row["name"] for row in segments)==set(program["assets"])
    assert not (out/"payload_blob.bin").exists() and (out/"test.elf").exists()
    assert '.asset_count=0' in (out/"test.c").read_text()
    assert all(s["address"]==plan["assets"][s["name"]]["base"] for s in segments)
    assert image["initial_asset_bytes"]==sum(s["file_bytes"] for s in segments)
    assert not image["full_model_execution_verified"]


def test_reference_inventory_adapter_requires_bound_logits(tmp_path):
    file=tmp_path/"logits-0.f16.bin";file.write_bytes(bytes(8));digest=hashlib.sha256(file.read_bytes()).hexdigest()
    value={"reference_checks":[{"forward_id":0,"token_id":1,"logits_shape":[1,4],"logits_file":str(file),"logits_sha256":digest}]}
    reference=result_reference(value)
    assert reference["outputs"][0]["dtype"]=="f16" and reference["outputs"][0]["tokens"]==[1]
    duplicate=copy.deepcopy(value);duplicate["reference_checks"]*=2
    with pytest.raises(ValueError,match="duplicate"):result_reference(duplicate)
    file.write_bytes(bytes([1])*8)
    with pytest.raises(ValueError,match="fingerprint"):result_reference(value)


@pytest.mark.parametrize("value",[{"outputs":[]},{"outputs":[{"forward_id":0},{"forward_id":0}]},{"reference_checks":[]}])
def test_empty_or_ambiguous_reference_is_rejected(value):
    with pytest.raises(ValueError):result_reference(value)


def test_asset_manifest_and_elf_share_explicit_simulator_forwarding():
    command=simulation_command("sim","model.elf",True,"segments.json",10**13)
    assert command==["sim","+max-cycles=10000000000000","+permissive","+loadmem=model.elf","+mlx_memory_segments=segments.json","+permissive-off","model.elf"]
