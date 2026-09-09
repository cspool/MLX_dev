import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('status,code,mode',[('running',None,'physical'),('exited',1,'physical'),('exited',0,'paired')])
def test_package_rejects_nonaccepted_execution_before_copy(tmp_path,status,code,mode):
    run=tmp_path/'run';run.mkdir()
    (run/'execution.json').write_text(json.dumps(dict(status=status,exit_code=code,mode=mode)))
    (run/'comparison.json').write_text('{}')
    output=tmp_path/'package'
    command=[sys.executable,'-m','scripts.package_mlx_physical_qa_result','--run',str(run),
             '--reference',str(tmp_path/'reference'),'--inventory',str(tmp_path/'inventory.json'),'--output',str(output)]
    result=subprocess.run(command,capture_output=True,text=True,timeout=30,cwd=Path(__file__).resolve().parents[1])
    assert result.returncode!=0 and 'not completed physical QA' in result.stderr
    assert not output.exists()
