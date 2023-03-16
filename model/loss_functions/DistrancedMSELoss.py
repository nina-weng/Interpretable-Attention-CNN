from torch import nn
import torch
import logging

class DistancedMSELoss(nn.Module):
    def __init__(self,reduction='mean'):
        super(DistancedMSELoss, self).__init__()
        self.reduction = reduction
        self.coeff = 1000
        logging.info(f'Distanced MSE Loss Parameter: coeff {self.coeff}')

    def forward(self,  pred, target):
        square_difference = torch.square(pred - target)
        out = torch.add(square_difference[:,0],square_difference[:,1])

        scaled_distance_to_center = abs((target[:,0]/4-100)**2 + (target[:,1]/3-100)**2 - (37.5**2 + 25**2))
        weight = (scaled_distance_to_center+self.coeff)/(scaled_distance_to_center+1)
        out= out.mul(weight)


        if self.reduction == 'mean':
            out = torch.mean(out)
        elif self.reduction == 'sum':
            out = torch.sum(out)
        return out