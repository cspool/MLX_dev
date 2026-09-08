"""Integer oracle tests independent of the MLX FU and dot-product code."""
import bisect
import json
from pathlib import Path
import random
import struct
import subprocess

import numpy as np
import pytest

from mlxsim.model_matrix_reference import kasc_reference

ROOT=Path(__file__).resolve().parents[1]


def scaled(h):
    value=struct.unpack('<e',struct.pack('<H',h))[0];n,d=value.as_integer_ratio();return n*(2**48//d)


POSITIVE=[scaled(h) for h in range(0x7c00)]+[65536*2**48]


def nearest(value):
    if not value:return 0
    sign=0x8000 if value<0 else 0;x=abs(value);at=bisect.bisect_left(POSITIVE,x)
    if at==0:return sign
    if at>=len(POSITIVE):return sign|0x7c00
    low,high=POSITIVE[at-1],POSITIVE[at]
    chosen=at-1 if x-low<high-x else at if x-low>high-x else (at if at%2==0 else at-1)
    return sign|chosen


@pytest.fixture(scope='module')
def binary():
    build=ROOT/'build/mlx-exact-numeric'
    for command in (['cmake','-S',ROOT/'simulator_ext/exact_numeric','-B',build,'-DCMAKE_BUILD_TYPE=Release'],['cmake','--build',build,'-j4']):
        p=subprocess.run(list(map(str,command)),capture_output=True,text=True,timeout=120);assert p.returncode==0,p.stdout+p.stderr
    return build/'mlx-exact-f16-dot'


def execute(binary,path,job,success=True):
    path.mkdir();file=path/'job.json';file.write_text(json.dumps(job));p=subprocess.run([str(binary),str(file),str(path/'out')],capture_output=True,text=True,timeout=120)
    if not success:
        assert p.returncode!=0 and not (path/'out/report.json').exists();return
    assert p.returncode==0,p.stdout+p.stderr
    return json.loads((path/'out/report.json').read_text())


def test_every_finite_half_roundtrips_in_the_exact_lattice(binary,tmp_path):
    bits=[h for h in range(65536) if (h>>10)&31!=31];values=[scaled(h) for h in bits]
    result=execute(binary,tmp_path/'all-half',{'mode':'round','values':list(map(str,values))})
    assert result['rounded']==[h if h!=0x8000 else 0 for h in bits]


@pytest.mark.parametrize('negative',[False,True])
def test_all_halfway_points_and_adjacent_exact_values(binary,tmp_path,negative):
    sign=-1 if negative else 1;values=[]
    for a,b in zip(POSITIVE,POSITIVE[1:]):
        middle=(a+b)//2;values.extend(sign*(middle+d) for d in (-1,0,1))
    result=execute(binary,tmp_path/'midpoints',{'mode':'round','values':list(map(str,values))})
    assert result['rounded']==[nearest(v) for v in values]


def test_random_wide_signed_exact_values(binary,tmp_path):
    rng=random.Random(70219);values=[rng.getrandbits(rng.randrange(1,121))*(-1 if rng.randrange(2) else 1) for _ in range(4000)]
    result=execute(binary,tmp_path/'wide',{'mode':'round','values':list(map(str,values))})
    assert result['rounded']==[nearest(v) for v in values]


@pytest.mark.parametrize('k',[1,7,64,4096])
def test_exact_dot_matches_python_integers_and_separate_fp32_control(binary,tmp_path,k):
    rng=np.random.default_rng(123+k);a=rng.integers(0,0x7c00,size=(3,k),dtype=np.uint16);b=rng.integers(0,0x7c00,size=(5,k),dtype=np.uint16)
    a|=rng.integers(0,2,size=a.shape,dtype=np.uint16)<<15;b|=rng.integers(0,2,size=b.shape,dtype=np.uint16)<<15
    af=tmp_path/'a.bin';bf=tmp_path/'b.bin';af.write_bytes(b'pad!'+a.tobytes());bf.write_bytes(b'prefix!!'+b.tobytes())
    directory=tmp_path/'dot';execute(binary,directory,{'mode':'linear_f16_no_bias','m':3,'n':5,'k':k,'a':{'file':str(af),'offset':4},'b':{'file':str(bf),'offset':8}})
    # Each operand has units 2^-24. Python unlimited integers form an
    # independent control/reference for the C++ bounded int128 accumulator.
    ai=[[scaled(int(h))//2**24 for h in row] for row in a];bi=[[scaled(int(h))//2**24 for h in row] for row in b]
    expected=[nearest(sum(x*y for x,y in zip(ar,br))) for ar in ai for br in bi]
    actual=np.fromfile(directory/'out/exact.f16.bin',dtype=np.uint16);assert actual.tolist()==expected
    with np.errstate(over='ignore',invalid='ignore'):
        fp=kasc_reference(a.view(np.float16),b.view(np.float16),transposed_b=True,output_dtype=np.float16).view(np.uint16)
    assert np.array_equal(np.fromfile(directory/'out/kasc.f16.bin',dtype=np.uint16).reshape(3,5),fp)


@pytest.mark.parametrize('bad',['nan','inf','bounds','dimension','mode','integer'])
def test_invalid_inputs_do_not_produce_a_reference(binary,tmp_path,bad):
    file=tmp_path/'input.bin';file.write_bytes(struct.pack('<H',0x7e00 if bad=='nan' else 0x7c00 if bad=='inf' else 0x3c00))
    job={'mode':'linear_f16_no_bias','m':1,'n':1,'k':1,'a':{'file':str(file),'offset':0},'b':{'file':str(file),'offset':0}}
    if bad=='bounds':job['a']['offset']=2
    elif bad=='dimension':job['k']=2**21
    elif bad=='mode':job['mode']='arbitrary'
    elif bad=='integer':job={'mode':'round','values':[str(2**121)]}
    execute(binary,tmp_path/'reject',job,False)
