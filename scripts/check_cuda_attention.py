import json,types,sys
from pathlib import Path
from copy import deepcopy
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from segment_anything.modeling.image_encoder import Attention
from srfl.models.attention import sdpa_attention_forward

torch.set_num_threads(1)
torch.manual_seed(702)
device=sys.argv[1] if len(sys.argv)>1 else 'cuda:0'
original=Attention(dim=768,num_heads=12,qkv_bias=True,use_rel_pos=True,input_size=(16,16)).to(device)
with torch.no_grad():
    original.rel_pos_h.normal_(0,.05);original.rel_pos_w.normal_(0,.05)
original.requires_grad_(False)
candidate=deepcopy(original)
candidate.forward=types.MethodType(sdpa_attention_forward,candidate)
records=[]
for amp in (False,True):
    x=torch.randn(2,16,16,768,device=device,requires_grad=True)
    y=x.detach().clone().requires_grad_()
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
        expected=original(x);actual=candidate(y)
    expected.float().square().mean().backward();actual.float().square().mean().backward()
    result={'bf16':amp}
    for name,a,b in [('output',expected,actual),('input_gradient',x.grad,y.grad)]:
        error=(a.float()-b.float()).norm()/a.float().norm().clamp_min(1e-12)
        result[name+'_relative_l2']=float(error.detach())
        assert error < (.03 if amp else 1e-5),result
    records.append(result)
print(json.dumps(records,indent=2))
