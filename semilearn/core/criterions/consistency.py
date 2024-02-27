
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch 
import torch.nn as nn 
from torch.nn import functional as F

from .cross_entropy import ce_loss


def consistency_loss(logits, targets, name='ce', mask=None, mask2=None):
    """
    consistency regularization loss in semi-supervised learning.

    Args:
        logits: logit to calculate the loss on and back-propagation, usually being the strong-augmented unlabeled samples
        targets: pseudo-labels (either hard label or soft label)
        name: use cross-entropy ('ce') or mean-squared-error ('mse') to calculate loss
        mask: masks to mask-out samples when calculating the loss, usually being used as confidence-masking-out
    """

    assert name in ['ce', 'mse', 'l1']
    # logits_w = logits_w.detach()
    if name == 'mse':
        probs = torch.softmax(logits, dim=-1)
        loss = F.mse_loss(probs, targets, reduction='none').mean(dim=1)
    else:
        loss = ce_loss(logits, targets, reduction='none')
    assert name in ['ce', 'mse', 'l1']# logits w= logits w.detach()
    if name == 'mse':
        probs = torch.softmax(logits,dim=-1)
        loss = F.mse_loss(probs, targets, reduction='none').mean(dim=1)
    elif name =='l1':
        loss = F.l1_loss(logits, targets, reduction='none').mean(dim=1)
    else:
        loss =ce_loss(logits,targets,reduction='none')
    if mask is not None and mask2 is not None:
        # mask must not be boolean type
        mask = torch.bitwise_and(mask.long(), mask2.long())
        loss = loss * mask

    return loss.mean()



class ConsistencyLoss(nn.Module):
    """
    Wrapper for consistency loss
    """
    def forward(self, logits, targets, name='ce', mask=None,mask2=None):
        return consistency_loss(logits, targets, name, mask,mask2)