import copy
import json
import subprocess

import pytest

from test_clocked_rocc import binary
from test_clocked_device import chain


def execute(binary,directory,job):
    directory.mkdir();(directory/"job.json").write_text(json.dumps(job));process=subprocess.run([str(binary),str(directory/"job.json"),str(directory/"report.json")],capture_output=True,text=True,timeout=120)
    assert process.returncode==0,process.stdout+process.stderr
    return json.loads((directory/"report.json").read_text())


@pytest.mark.parametrize("mode",["normal","limited","io_failure"])
def test_observation_preserves_data_and_all_target_counters(binary,tmp_path,mode):
    job=chain();plain=execute(binary,tmp_path/"plain",job);observed=copy.deepcopy(job)
    path=tmp_path/("missing/progress.json" if mode=="io_failure" else "progress.json")
    observed["progress"]=str(path);observed["progress_period"]=5
    observed["progress_map"]=[{"launch_ordinal":i,"source_operator_id":i+101,"source_ordinal":i,"forward_id":0,"layer_idx":2,"phase":"prefill","kind":kind,"family":family,"batch_index":0,"batch_count":1,"shape":[1,4],"dtype":"f16"} for i,(kind,family) in enumerate((("linear","matrix"),("mul","vector"),("cast","memory")))]
    if mode=="limited":observed["progress_limit"]=3
    result=execute(binary,tmp_path/"observed",observed);status=result.pop("progress_observer")
    assert result==plain
    if mode=="io_failure":assert status["failed"];return
    assert not status["failed"]
    latest=json.loads(path.read_text());events=[json.loads(line) for line in path.with_suffix(".jsonl").read_text().splitlines()]
    assert latest["final_snapshot"] and latest["source"]["source_operator_id"]==103
    if mode=="limited":assert len(events)==3 and latest["event_log_truncated"]
    else:
        assert sum(e["launch_event"] for e in events)==sum(e["terminal_event"] for e in events)==3
        assert [e["source"]["source_operator_id"] for e in events if e["launch_event"]]==[101,102,103]
