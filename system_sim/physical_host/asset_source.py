"""Bind input assets to CPU-readable file regions without a giant ELF payload."""
import hashlib
import math
from pathlib import Path

from .graph_lowering import literal_bytes
from .lowering import BYTES

SOURCE_BASE=2**38


def source_manifest(program,directory,*,source_offset=4096):
    if source_offset<0 or source_offset%64:raise ValueError("asset source offset must be nonnegative and 64-byte aligned")
    directory=Path(directory).resolve();literal_file=directory/"literal_assets.bin"
    entries=[];regions=[];at=source_offset;literal_offset=0
    with literal_file.open("wb") as output:
        for name,spec in sorted(program["assets"].items()):
            count=math.prod(spec["shape"])*BYTES[spec["dtype"]]
            if count<0 or count>2**40-SOURCE_BASE-at:raise ValueError("asset source mapping exceeds supported address range")
            digest=hashlib.sha256()
            if spec["kind"]=="literal":
                data=literal_bytes(spec);path=literal_file;offset=literal_offset;output.write(data);digest.update(data);literal_offset+=len(data)
            elif spec["kind"]=="mapped_file":
                path=Path(spec["path"]).resolve();offset=spec["byte_offset"]
                if spec["bytes"]!=count or offset<0 or offset+count>path.stat().st_size:raise ValueError("asset source file extent invalid")
                with path.open("rb") as input:
                    input.seek(offset);remaining=count
                    while remaining:
                        data=input.read(min(remaining,8*2**20))
                        if not data:raise ValueError("asset source file ended early")
                        digest.update(data);remaining-=len(data)
            else:raise ValueError("unsupported asset source kind")
            regions.append({"name":name,"offset":at,"bytes":count,"path":str(path),"file_offset":offset})
            entries.append({"value":name,"source_address":SOURCE_BASE+at,"bytes":count,"sha256":digest.hexdigest()})
            at+=max(count,1);at+=(-at)%64
    capacity=max(4096,(at+4095)//4096*4096)
    if capacity>2**40-SOURCE_BASE:raise ValueError("rounded asset source mapping exceeds supported address range")
    return entries,{"version":1,"bytes":capacity,"regions":regions}
