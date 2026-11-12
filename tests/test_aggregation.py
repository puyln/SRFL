from copy import deepcopy
import numpy as np
import torch
from srfl.config import DEFAULTS
from srfl.aggregation import aggregate
from srfl.core_numpy import gua_aggregate


def test_gua_matches_reference_when_unnormalized():
    cfg={**DEFAULTS['aggregation'],'normalize_direction':False,'global_step':0.02,'tolerance':1e-12,'iterations':1000}
    initial={'w':torch.zeros(3,dtype=torch.float64)}
    clients=[{'w':torch.tensor(v,dtype=torch.float64)} for v in [[1,2,3],[-2,1,4]]]
    actual,diagnostic=aggregate(initial,clients,[1,1],cfg)
    expected=gua_aggregate(np.zeros(3),np.array([[1,2,3],[-2,1,4]]),global_step=0.02,solver_iterations=1000)
    np.testing.assert_allclose(actual['w'].numpy(),expected.parameters,atol=1e-7)
    assert abs(sum(diagnostic['weights'])-1)<1e-10


def test_safeguard_decreases_fixed_client_distances():
    cfg={**DEFAULTS['aggregation'],'safeguard':True,'global_step':10.,'iterations':1000}
    initial={'w':torch.zeros(2,dtype=torch.float64)}
    clients=[{'w':torch.tensor(v,dtype=torch.float64)} for v in [[1.,1.],[2.,1.]]]
    _,report=aggregate(initial,clients,[1,1],cfg)
    assert report['applied_step']>0
    assert min(report['relative_deviation_change'])>0


def test_identical_clients_make_no_update():
    state={'w':torch.arange(3,dtype=torch.float64)}
    updated,report=aggregate(state,[deepcopy(state),deepcopy(state)],[1,1],DEFAULTS['aggregation'])
    torch.testing.assert_close(updated['w'],state['w'])
    assert report['raw_direction_norm']==0


def test_fedavg_uses_explicit_slice_counts():
    cfg={**DEFAULTS['aggregation'],'method':'fedavg'}
    state={'w':torch.zeros(1)}
    updated,_=aggregate(state,[{'w':torch.tensor([1.])},{'w':torch.tensor([3.])}],[1,3],cfg)
    assert updated['w'].item()==2.5
