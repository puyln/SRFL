from contextlib import nullcontext
import torch
import torch.nn.functional as F


def sdpa_attention_forward(self,x):
    from segment_anything.modeling.image_encoder import get_rel_pos
    b,h,w,_=x.shape
    qkv=self.qkv(x).reshape(b,h*w,3,self.num_heads,-1).permute(2,0,3,1,4)
    q,k,v=qkv.unbind(0)
    bias=None
    if self.use_rel_pos:
        rh=get_rel_pos(h,h,self.rel_pos_h)
        rw=get_rel_pos(w,w,self.rel_pos_w)
        spatial=q.reshape(b*self.num_heads,h,w,-1)
        rel_h=torch.einsum('bhwc,hkc->bhwk',spatial,rh)
        rel_w=torch.einsum('bhwc,wkc->bhwk',spatial,rw)

        bias=(rel_h[:,:,:,:,None]+rel_w[:,:,:,None,:]).reshape(b,self.num_heads,h*w,h*w)
    if x.is_cuda:
        from torch.nn.attention import sdpa_kernel,SDPBackend
        context=sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])
    else:
        context=nullcontext()
    with context:
        value=F.scaled_dot_product_attention(q,k,v,attn_mask=bias,dropout_p=0.0,scale=self.scale)
    return self.proj(value.reshape(b,self.num_heads,h,w,-1).permute(0,2,3,1,4).reshape(b,h,w,-1))
