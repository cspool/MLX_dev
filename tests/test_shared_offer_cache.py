from pathlib import Path
import subprocess


ROOT=Path(__file__).resolve().parents[1]


def test_failed_offer_cache_preserves_capacity_edges_and_caller_checks():
    build=ROOT/'build/host-opt-candidate'
    for command in (['cmake','-S',ROOT/'simulator_ext/model_system','-B',build,'-DCMAKE_BUILD_TYPE=Release'],['cmake','--build',build,'--target','shared-offer-cache-contract','-j4']):
        result=subprocess.run(list(map(str,command)),capture_output=True,text=True,timeout=180)
        assert result.returncode==0,result.stdout+result.stderr
    result=subprocess.run([str(build/'shared-offer-cache-contract')],capture_output=True,text=True,timeout=30)
    assert result.returncode==0 and 'SHARED_OFFER_CACHE_CONTRACT_PASS' in result.stdout,result.stdout+result.stderr
