from torch import nn
import torch

class CosLoss(nn.Module):
    def __init__(self,reduction='mean'):
        super(CosLoss, self).__init__()
        self.reduction = reduction

    def forward(self,  pred, target):
        cosloss = 2*(1 - torch.cos(pred - target))
        out = cosloss
        if self.reduction == 'mean':
            out = torch.mean(out)
        elif self.reduction == 'sum':
            out = torch.sum(out)
        return out