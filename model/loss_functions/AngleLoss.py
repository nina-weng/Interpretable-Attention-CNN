from torch import nn
import torch

class AngleLoss(nn.Module):
    def __init__(self,reduction='mean'):
        super(AngleLoss, self).__init__()
        self.reduction = reduction

    def forward(self,  pred, target):
        atan = torch.atan2(torch.sin(pred - target), torch.cos(pred - target))
        out = torch.abs(atan) #using abs
        if self.reduction == 'mean':
            out = torch.mean(out)
        elif self.reduction == 'sum':
            out = torch.sum(out)
        return out