from torch import nn
import torch

class WeightedMSELoss(nn.Module):
    def __init__(self, weight,reduction='mean'):
        super(WeightedMSELoss, self).__init__()
        self.weight = torch.tensor(weight).float()
        if torch.cuda.is_available():
            self.weight = self.weight.cuda()
        self.reduction = reduction

        assert len(self.weight) == 129

    def forward(self,  pred, target, scale=None):
        square_difference = torch.square(pred - target)
        # out = torch.squeeze(square_difference,dim=1)
        mean_each_channel = torch.mean(square_difference,dim=1)
        mean_each_channel = mean_each_channel.float()
        if scale is None:
            out = torch.matmul(mean_each_channel,self.weight)
        else:
            out = mean_each_channel.mul(scale)
        if self.reduction == 'mean':
            out = torch.mean(out)
        elif self.reduction == 'sum':
            out = torch.sum(out)
        return out