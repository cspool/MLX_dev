import pytest

from scripts.verify_mlx_system_model import reference_argmax


@pytest.mark.parametrize("shape,values,tokens",[
    ([1,4],[1.,4.,2.,0.],[1]),
    ([2,2],[5.,-1.,0.,9.],[0,1]),
    ([2,1,2],[3.,3.,-2.,-1.],[0,1]),
    ([3,1],[7.,2.,-8.],[0,0,0]),
])
def test_reference_argmax_checks_every_last_axis_row(shape,values,tokens):
    assert reference_argmax(values,shape)==tokens


@pytest.mark.parametrize("shape,values",[([],[]),([0,2],[]),([True,2],[1.,2.]),([1.,2],[1.,2.]),([2,2],[1.,2.]),([1,2],[float('nan'),1.]),([1,1],[float('inf')])])
def test_reference_argmax_rejects_invalid_or_nonfinite_rows(shape,values):
    with pytest.raises(RuntimeError):reference_argmax(values,shape)
