import torch
from torch import nn
import torch.nn.functional as F


def partial_segmentation_loss(logits, target, annotated_classes, *, mode='marginal',
                              ce_weight=1., dice_weight=1., ignore_index=-1):
    if logits.ndim != 4 or target.shape != (logits.shape[0], *logits.shape[-2:]):
        raise ValueError('Expected logits [B,C,H,W], target [B,H,W]')
    if annotated_classes.shape != logits.shape[:2]:
        raise ValueError('annotated_classes must have shape [B,C]')
    if mode not in ('marginal', 'ignore'):
        raise ValueError('loss mode must be marginal or ignore')
    losses = []
    for sample_logits, sample_target, annotated in zip(logits, target, annotated_classes):
        valid = sample_target != ignore_index
        if not valid.any():
            losses.append(sample_logits.sum() * 0.)
            continue
        if (sample_target[valid] < 0).any() or (sample_target[valid] >= logits.shape[1]).any():
            raise ValueError('Target contains invalid class IDs')
        annotated = annotated.bool()
        if not annotated[0]:
            raise ValueError('Class zero must be included as the background/unknown union')
        positive_ids = torch.where(annotated)[0]
        if not annotated[sample_target[valid]].all():
            raise ValueError('A target class is not declared in annotated_classes')
        log_probability = sample_logits.log_softmax(dim=0)
        if mode == 'marginal':
            union = ~annotated.clone()
            union[0] = True
            log_categories = [torch.logsumexp(log_probability[union], dim=0)]
            log_categories.extend(log_probability[int(c)] for c in positive_ids if c != 0)
            log_categories = torch.stack(log_categories)
            mapping = torch.zeros(logits.shape[1], device=logits.device, dtype=torch.long)
            for index, class_id in enumerate(positive_ids):
                mapping[class_id] = index
            local_target = mapping[sample_target.clamp_min(0)]
        else:

            log_categories = log_probability
            local_target = sample_target.clamp_min(0)
        selected = log_categories.gather(0, local_target[None]).squeeze(0)
        ce = -selected[valid].mean()
        probability = log_categories.exp() * valid[None]
        one_hot = F.one_hot(local_target, log_categories.shape[0]).permute(2,0,1).to(logits.dtype) * valid[None]
        intersection = (probability * one_hot).sum(dim=(1,2))
        denominator = probability.sum(dim=(1,2)) + one_hot.sum(dim=(1,2))
        dice = 1. - ((2. * intersection + 1e-6) / (denominator + 1e-6)).mean()
        losses.append(ce_weight * ce + dice_weight * dice)
    return torch.stack(losses).mean()
