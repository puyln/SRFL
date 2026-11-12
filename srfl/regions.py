import torch
import torch.nn.functional as F


def region_targets(labels, regions):
    return torch.stack([sum((labels==v).to(torch.int8) for v in values)>0 for values in regions],dim=1)


def partial_region_loss(logits, labels, annotated, regions, bce_weight=1., dice_weight=1.):
    if logits.shape[1]!=len(regions) or annotated.shape!=logits.shape[:2]:
        raise ValueError('Region channel/annotation shape mismatch')
    target=region_targets(labels,regions).float()
    valid=(labels!=-1)[:,None].expand_as(target)
    active=annotated & valid.flatten(2).any(2)
    if not active.any():
        return logits.sum()*0
    logits=logits.float()
    bce=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    bce=(bce*valid).flatten(2).sum(2)/valid.flatten(2).sum(2).clamp_min(1)
    probability=logits.sigmoid()*valid
    target=target*valid
    intersection=(probability*target).flatten(2).sum(2)
    dice=1-(2*intersection+1e-5)/(probability.flatten(2).sum(2)+target.flatten(2).sum(2)+1e-5)

    weights=active.float()/active.sum(1,keepdim=True).clamp_min(1)
    rows=active.any(1)
    return ((bce_weight*bce+dice_weight*dice)*weights).sum(1)[rows].mean()
