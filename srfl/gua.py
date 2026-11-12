import torch


def project_simplex(vector, mass=1.0):

    if vector.ndim != 1 or vector.numel() == 0 or not torch.isfinite(vector).all():
        raise ValueError('vector must be a finite non-empty one-dimensional tensor')
    if not 0 < mass < float('inf'):
        raise ValueError('mass must be finite and positive')
    ordered=torch.sort(vector,descending=True).values
    cumulative=torch.cumsum(ordered,0)-mass
    indices=torch.arange(1,vector.numel()+1,device=vector.device,dtype=vector.dtype)
    valid=ordered-cumulative/indices>0
    rho=torch.nonzero(valid,as_tuple=False)[-1,0]+1
    theta=cumulative[rho-1]/rho.to(vector.dtype)
    return (vector-theta).clamp_min(0.)
