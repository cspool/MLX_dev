"""Durable process ownership and source snapshots for long system attempts."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as source:return hashlib.file_digest(source,"sha256").hexdigest()


def record(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2)+"\n");temporary.replace(path)


def snapshot_sources(root,destination,sources):
    root=Path(root).resolve();destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    for name,expected in sources.items():
        relative=Path(name)
        if relative.is_absolute() or ".." in relative.parts:raise ValueError("source snapshot requires repository-relative paths")
        source=root/relative;target=destination/relative
        if digest(source)!=expected:raise RuntimeError("source changed before snapshot")
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        if digest(target)!=expected or digest(source)!=expected:raise RuntimeError("source changed while snapshotting")


def launch_metadata(program,plan):
    nodes={node["source_operator_id"]:node for node in program["nodes"]};rows=[]
    for task in plan["tasks"]:
        if task["kind"] not in (2,3):continue
        node=nodes[task["source_id"]]
        rows.append({"launch_ordinal":len(rows),"source_operator_id":task["source_id"],"source_ordinal":task["source_ordinal"],"batch_index":task["batch_index"],"batch_count":task["batch_count"],
            "forward_id":node["forward_id"],"layer_idx":node["layer_idx"],"phase":node.get("phase"),"kind":node["kind"],"family":task["family"],"shape":node["output"]["shape"],"dtype":node["output"]["dtype"]})
        if task["kind"]==3:
            consumer=program["nodes"][task["consumer_ordinal"]]
            rows[-1]["consumer"]={"source_ordinal":task["consumer_ordinal"],"source_operator_id":consumer["source_operator_id"],"kind":consumer["kind"],"forward_id":consumer["forward_id"],"layer_idx":consumer["layer_idx"],"phase":consumer.get("phase"),"shape":consumer["output"]["shape"],"dtype":consumer["output"]["dtype"]}
    return rows


def linked_libraries(binary):
    result=subprocess.run(["ldd",str(binary)],capture_output=True,text=True,check=True,timeout=30)
    if "not found" in result.stdout:raise RuntimeError("system executable has unresolved runtime libraries")
    paths=set()
    for line in result.stdout.splitlines():
        match=re.search(r"(?:=>\s+)?(/[^\s]+)\s+\(",line)
        if match:paths.add(str(Path(match.group(1)).resolve()))
    return {path:digest(path) for path in sorted(paths)}


def run_process(command,log,record_path,*,timeout,env=None,metadata=None,cwd=None):
    if timeout<=0:raise ValueError("positive process watchdog required")
    state={"classification":"owned_system_process_attempt_not_success_certificate",**(metadata or {}),"command":list(map(str,command)),"runner_pid":os.getpid(),"status":"starting","watchdog_seconds":timeout}
    record(record_path,state);start=time.monotonic()
    with Path(log).open("w") as output:
        try:process=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,env=env,cwd=cwd)
        except Exception as error:
            state.update(status="spawn_failed",error=str(error));record(record_path,state);raise
        state.update(status="running",pid=process.pid);record(record_path,state)
        try:process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # The pinned emulator's SIGTERM handler dereferences a DTM pointer
            # even for TSI runs. Stop this owned process without assuming a
            # graceful device drain; progress remains diagnostic, not success.
            process.kill();process.wait();state.update(status="watchdog",exit_code=process.returncode,host_elapsed_seconds=time.monotonic()-start);record(record_path,state);raise
    state.update(status="exited",exit_code=process.returncode,host_elapsed_seconds=time.monotonic()-start);record(record_path,state)
    if process.returncode:raise subprocess.CalledProcessError(process.returncode,command)
    return state
