import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.mlx_system_attempt import digest,run_process,snapshot_sources


@pytest.mark.parametrize("code",[0,7])
def test_exit_is_recorded_without_claiming_validation(tmp_path,code):
    record=tmp_path/"execution.json";command=[sys.executable,"-c",f"print('ran');raise SystemExit({code})"]
    if code:
        with pytest.raises(subprocess.CalledProcessError):run_process(command,tmp_path/"log",record,timeout=30)
    else:run_process(command,tmp_path/"log",record,timeout=30)
    state=json.loads(record.read_text());assert state["status"]=="exited" and state["exit_code"]==code and state["pid"]>0
    assert "validation" not in state and (tmp_path/"log").read_text().strip()=="ran"


def test_watchdog_records_terminal_owned_process(tmp_path):
    record=tmp_path/"execution.json"
    with pytest.raises(subprocess.TimeoutExpired):run_process([sys.executable,"-c","import time;time.sleep(10)"],tmp_path/"log",record,timeout=0.05)
    state=json.loads(record.read_text());assert state["status"]=="watchdog" and state["exit_code"]!=0


def test_snapshots_preserve_source_bytes_and_reject_changes(tmp_path):
    root=tmp_path/"root";root.mkdir();file=root/"source.cc";file.write_text("int value=1;\n");source={"source.cc":digest(file)}
    snapshot_sources(root,tmp_path/"snapshot",source);file.write_text("int value=2;\n")
    assert (tmp_path/"snapshot/source.cc").read_text()=="int value=1;\n"
    with pytest.raises(RuntimeError):snapshot_sources(root,tmp_path/"changed",source)
    with pytest.raises(ValueError):snapshot_sources(root,tmp_path/"unsafe",{"../source.cc":"bad"})
