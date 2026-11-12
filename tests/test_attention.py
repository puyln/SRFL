from copy import deepcopy
import types
import pytest
import torch
from srfl.models.attention import sdpa_attention_forward


def test_sdpa_preserves_nonzero_relative_bias_and_gradients():
    pytest.importorskip('segment_anything')
    from segment_anything.modeling.image_encoder import Attention
    torch.manual_seed(91)
    original=Attention(dim=32,num_heads=4,qkv_bias=True,use_rel_pos=True,input_size=(8,8)).double()
    with torch.no_grad():
        original.rel_pos_h.normal_(0,.1);original.rel_pos_w.normal_(0,.1)
    candidate=deepcopy(original)
    candidate.forward=types.MethodType(sdpa_attention_forward,candidate)
    x=torch.randn(2,8,8,32,dtype=torch.double,requires_grad=True)
    y=x.detach().clone().requires_grad_()
    reference=original(x);actual=candidate(y)
    torch.testing.assert_close(reference,actual,rtol=1e-10,atol=1e-10)
    reference.square().sum().backward();actual.square().sum().backward()
    torch.testing.assert_close(x.grad,y.grad,rtol=1e-9,atol=1e-10)
    for a,b in zip(original.parameters(),candidate.parameters()):
        torch.testing.assert_close(a.grad,b.grad,rtol=1e-9,atol=1e-10)
