# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch.nn as nn

from torch.nn import functional as F


def l1_loss(logits, target, reduction='mean', **kwargs):
    """Calculate L1 loss."""
    loss = F.l1_loss(logits, target, reduction='none')
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


def l2_loss(logits, target, reduction='mean', **kwargs):
    """Calculate MSE (L2) loss."""
    loss = F.mse_loss(logits, target, reduction='none')
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


class RegLoss(nn.Module):
    """
    Wrapper for regression loss
    """
    def __init__(self,
                 mode="l1_loss",
                 **kwargs):
        super(RegLoss, self).__init__()
        self.mode = mode
        self.loss_list = ["l2_loss", "l1_loss"]
        assert mode in self.loss_list
        self.criterion = eval(self.mode)

    def forward(self, logits, targets, reduction='mean'):
        if logits.dim() == 2 and targets.dim() < 2:
            targets = targets.view(targets.shape[0], logits.shape[1])
        return self.criterion(logits, targets, reduction=reduction)
