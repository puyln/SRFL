from copy import deepcopy
import numpy as np
import pytest
import torch
from srfl.config import DEFAULTS, merge
from srfl.models import SRFL
from srfl.models.encoder import attach_adapters, AdaptedOperation
from srfl.losses import partial_segmentation_loss
from srfl.usgr import USGRLoss
from srfl.core_numpy import solve_unbalanced_transport


def tiny_config():
    return merge(DEFAULTS,{'model':{'backbone':'tiny','image_size':32,'num_classes':3,'adapter_dim':8,
        'decoder_channels':32,'decoder_depth':1,'query_grid':4,'sam_normalization':False}})['model']


def test_two_adapters_per_block_receive_gradients_and_base_is_frozen():
    torch.set_num_threads(1)
    torch.manual_seed(5)
    model=SRFL(tiny_config())
    wrappers=[m for m in model.encoder.modules() if isinstance(m,AdaptedOperation)]
    assert len(wrappers)==2*len(model.encoder.blocks)
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    optimizer=torch.optim.Adam(model.trainable_parameters(),lr=1e-3)
    model.train()
    out=model(torch.rand(2,3,32,32))
    loss=partial_segmentation_loss(out.logits,torch.randint(0,3,(2,32,32)),torch.ones(2,3,dtype=torch.bool))
    loss=loss+0.05*USGRLoss(random_sample_count=8,max_features=16,tau_alpha=1000,tau_beta=1000)(out.usgr_features).loss
    loss.backward()
    for wrapper in wrappers:
        assert wrapper.adapter.up.weight.grad.abs().sum()>0
    for n,p in model.named_parameters():
        if not p.requires_grad:
            assert p.grad is None
    optimizer.step()
    for n,p in model.named_parameters():
        if n in frozen:
            torch.testing.assert_close(p,frozen[n],rtol=0,atol=0)
    with torch.no_grad():
        assert not model(torch.rand(1,3,32,32)).logits.requires_grad


def test_official_sam_window_and_global_blocks_preserve_zero_adapter_output():
    pytest.importorskip('segment_anything')
    from segment_anything.modeling import ImageEncoderViT
    torch.set_num_threads(1)
    original=ImageEncoderViT(img_size=32,patch_size=4,embed_dim=32,depth=2,num_heads=4,
        out_chans=16,use_rel_pos=True,window_size=3,global_attn_indexes=(1,)).eval()
    adapted=attach_adapters(deepcopy(original),8).eval()
    image=torch.randn(1,3,32,32)
    with torch.no_grad():
        torch.testing.assert_close(original(image),adapted(image))
    adapted(image).square().mean().backward()
    for operation in (b.attn for b in adapted.blocks):
        assert operation.adapter.up.weight.grad.abs().sum()>0
    for operation in (b.mlp for b in adapted.blocks):
        assert operation.adapter.up.weight.grad.abs().sum()>0


def test_all_ignored_supervision_is_zero_with_finite_zero_gradient():
    logits=torch.randn(2,3,4,4,requires_grad=True)
    loss=partial_segmentation_loss(logits,torch.full((2,4,4),-1),torch.ones(2,3,dtype=torch.bool))
    loss.backward()
    assert loss.item()==0
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum()==0


def test_marginal_background_preserves_unknown_class_probability():


    p=torch.tensor([0.2,0.3,0.5]).reshape(1,3,1,1)
    q=torch.tensor([0.4,0.3,0.3]).reshape(1,3,1,1)
    target=torch.zeros(1,1,1,dtype=torch.long)
    annotated=torch.tensor([[True,True,False]])
    torch.testing.assert_close(partial_segmentation_loss(p.log(),target,annotated),
                               partial_segmentation_loss(q.log(),target,annotated))


def test_complete_marginal_labels_equal_standard_ignore_mode():
    logits=torch.randn(2,3,4,4)
    target=torch.randint(0,3,(2,4,4))
    target[:,0,0]=-1
    annotated=torch.ones(2,3,dtype=torch.bool)
    torch.testing.assert_close(partial_segmentation_loss(logits,target,annotated,mode='marginal'),
                               partial_segmentation_loss(logits,target,annotated,mode='ignore'))


def test_usgr_matches_numpy_and_produces_nonzero_feature_gradients():
    torch.manual_seed(4)
    features=torch.randn(10,8,dtype=torch.float64,requires_grad=True)
    module=USGRLoss(random_sample_count=7,transport_iterations=100,tau_alpha=1000,tau_beta=1000)
    actual=module(features)
    expected=solve_unbalanced_transport(features.detach().numpy(),random_samples=actual.random_samples.numpy(),
                                         iterations=100,tau_alpha=1000,tau_beta=1000)
    np.testing.assert_allclose(actual.coupling.numpy(),expected.coupling,atol=1e-10)
    np.testing.assert_allclose(actual.loss.item(),expected.loss,atol=1e-10)
    actual.loss.backward()
    assert torch.isfinite(features.grad).all() and features.grad.abs().sum()>0
