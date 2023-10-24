# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch
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


def focal_l1_loss(logits, target, reduction='mean', activate='sigmoid', beta=0.2, gamma=1.0, **kwargs):
    """Calculate Focal L1 loss."""
    target = target.type_as(logits)
    loss = F.l1_loss(logits, target, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(logits - target))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(logits - target)) - 1) ** gamma
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


def focal_l2_loss(logits, target, reduction='mean', activate='sigmoid', beta=0.2, gamma=1.0, **kwargs):
    """Calculate Focal L2 loss."""
    target = target.type_as(logits)
    loss = F.mse_loss(logits, target, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(logits - target))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(logits - target)) - 1) ** gamma
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


def huber_loss(logits, target, reduction='mean', beta=1.0, **kwargs):
    """Calculate Smooth L1 loss."""
    l1_loss = F.l1_loss(logits, target, reduction='none')
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
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
        self.loss_list = ["l2_loss", "l1_loss", "focal_l1_loss", "focal_l2_loss", "huber_loss"]
        assert mode in self.loss_list
        self.criterion = eval(self.mode)

    def forward(self, logits, targets, reduction='mean'):
        if logits.shape != targets.shape:
            targets = targets.view(logits.shape)
        return self.criterion(logits, targets, reduction=reduction)
