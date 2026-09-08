import hashlib
import json
import struct
import subprocess
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
BASE=0x80000000
CAPACITY=16*2**30


def test_elf_preload_plusarg_is_forwarded_to_memory_and_tsi():
    from scripts.run_mlx_clocked_chipyard import simulation_command
    assert simulation_command(Path("sim"),Path("model.elf"),True)==["sim","+max-cycles=20000000","+permissive","+loadmem=model.elf","+permissive-off","model.elf"]
    assert simulation_command(Path("sim"),Path("model.elf"),False)==["sim","+max-cycles=20000000","model.elf"]


@pytest.fixture(scope="module")
def binary():
    build=ROOT/"build/mlx-wide-memory"
    for command in (["cmake","-S",str(ROOT/"system_sim/wide_memory"),"-B",str(build),"-DCMAKE_BUILD_TYPE=Release"],
                    ["cmake","--build",str(build),"--target","wide-memory-contract","-j4"]):
        p=subprocess.run(command,capture_output=True,text=True,timeout=120);assert p.returncode==0,p.stdout+p.stderr
    return build/"wide-memory-contract"


def execute(binary,tmp_path,job=None,error=False):
    command=[str(binary)]
    if job is not None:
        path=tmp_path/"job.json";path.write_text(json.dumps(job));command.append(str(path))
    result=subprocess.run(command,capture_output=True,text=True,timeout=60)
    if error:
        assert result.returncode!=0,result.stdout
        return
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads(result.stdout);(tmp_path/"report.json").write_text(json.dumps(report,indent=2)+"\n");return report


def test_axi_address_width_lanes_bursts_backpressure_and_rejection(binary,tmp_path):
    result=execute(binary,tmp_path)
    assert result["base"]==BASE and result["bytes"]==CAPACITY and result["idle"]
    assert result["max_read_address"]==BASE+CAPACITY-8
    assert result["max_write_address"]==BASE+CAPACITY-8
    assert result["aw_requests"]==result["write_responses"]


def elf(path,address,*,filesz=19,memsz=40,machine=243):
    ident=b"\x7fELF\x02\x01\x01"+bytes(9)
    header=struct.pack("<16sHHIQQQIHHHHHH",ident,2,machine,1,address,64,0,0,64,56,1,64,0,0)
    phdr=struct.pack("<IIQQQQQQ",1,7,256,address,address,filesz,memsz,8)
    payload=bytes(range(19));path.write_bytes(header+phdr+bytes(256-len(header)-len(phdr))+payload);return payload


def test_high_address_elf_file_data_and_zero_tail_are_preclock_initialization(binary,tmp_path):
    path=tmp_path/"test.elf";address=BASE+2**32+4096;raw=elf(path,address)
    report=execute(binary,tmp_path,{"base":BASE,"bytes":CAPACITY,"elf":str(path),"inspect":[{"address":address,"bytes":48}]})
    assert report["observed"]==[list(raw)+[0]*29]
    assert report["cycle"]==report["ar_requests"]==report["aw_requests"]==0
    assert report["initialized_segments"][0]["sha256"]==hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("damage",["machine","file_extent","memory_extent","outside"])
def test_elf_metadata_rejection(binary,tmp_path,damage):
    path=tmp_path/"bad.elf";options={"machine":62} if damage=="machine" else {"filesz":10000} if damage=="file_extent" else {"memsz":8} if damage=="memory_extent" else {}
    elf(path,BASE-4096 if damage=="outside" else BASE+4096,**options)
    execute(binary,tmp_path,{"base":BASE,"bytes":CAPACITY,"elf":str(path)},error=True)


@pytest.mark.parametrize("damage",[None,"overlap","digest","range"])
def test_checked_file_segments_preserve_offsets_and_digests(binary,tmp_path,damage):
    file=tmp_path/"input.bin";raw=bytes(range(31));file.write_bytes(b"prefix"+raw+b"tail")
    segment={"name":"weights","path":str(file),"address":BASE+2**32+8192,"file_offset":6,"file_bytes":31,"memory_bytes":32,"sha256":hashlib.sha256(raw).hexdigest()};segments=[segment]
    if damage=="overlap":segments.append({**segment,"name":"other","address":segment["address"]+8})
    elif damage=="digest":segment["sha256"]="0"*64
    elif damage=="range":segment["address"]=BASE+CAPACITY-16
    job={"base":BASE,"bytes":CAPACITY,"segments":segments,"inspect":[{"address":segment["address"],"bytes":32}]}
    report=execute(binary,tmp_path,job,error=damage is not None)
    if report:assert report["observed"]==[list(raw)+[0]] and report["ar_requests"]==0
