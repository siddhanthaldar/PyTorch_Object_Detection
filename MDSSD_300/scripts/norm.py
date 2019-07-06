import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class L2Norm(nn.Module):
    '''L2Norm layer across all channels and scale.'''
    def __init__(self, in_features,scale):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features))
        self.reset_parameters(scale)

    def reset_parameters(self, scale):
        nn.init.constant_(self.weight, scale)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        scale = self.weight[None,:,None,None]
        return scale * x