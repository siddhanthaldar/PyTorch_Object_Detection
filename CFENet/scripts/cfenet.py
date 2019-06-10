import math
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

from multibox_layer import MultiBoxLayer
from util import CFE, FFB


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


class CFENet(nn.Module):
    input_size = 300

    def __init__(self):
        super(CFENet, self).__init__()
        
        # model
        self.base = self.VGG16()
        self.norm4 = L2Norm(512, 20) # 38

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)

        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1) 

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3)

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3)

        # CFE
        self.cfe1 = CFE(512, 3)
        self.cfe2 = CFE(512, 3)
        self.cfe3 = CFE(512, 3)
        self.cfe4 = CFE(512, 3)

        # FFB
        self.ffb1 = FFB(512,512)
        self.ffb2 = FFB(512,512)

        # multibox layer
        self.multibox = MultiBoxLayer()

    def forward(self, x):
        hs = []
        ffb = []

        h = self.base(x)
        # hs.append(self.norm4(h))  # conv4_3
        ffb.append(h)
        h = self.cfe1(h)
        h = F.max_pool2d(h, kernel_size=2, stride=2, ceil_mode=True)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        ffb.append(h)
        h = self.cfe2(h)
        h = F.max_pool2d(h, kernel_size=3, padding=1, stride=1, ceil_mode=True)
        
        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        # hs.append(h)  # conv7
        h = F.relu(self.conv8_1(h))
        h = F.relu(self.conv8_2(h))
        hs.append(h)  # conv8_2
        ffb.append(h)
        h = F.relu(self.conv9_1(h))
        h = F.relu(self.conv9_2(h))
        hs.append(h)  # conv9_2
        h = F.relu(self.conv10_1(h))
        h = F.relu(self.conv10_2(h))
        hs.append(h)  # conv10_2
        h = F.relu(self.conv11_1(h))
        h = F.relu(self.conv11_2(h))
        hs.append(h)  # conv11_2

        # Feature fusion blocks followed by Comprehensive Feature Enhancement(CFE) module
        f1 = self.ffb1(ffb[0], ffb[1])
        f1 = self.cfe3(f1)
        hs.append(f1)
        f2 = self.ffb2(ffb[1], ffb[2])
        f2 = self.cfe4(f2)
        hs.append(f2)

        loc_preds, conf_preds = self.multibox(hs)
 
        return loc_preds, conf_preds

    def VGG16(self):
        '''VGG16 layers.'''
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.ReLU(True)]
                in_channels = x
        return nn.Sequential(*layers)

if __name__ == '__main__':
    t = torch.randn(1, 3, 300, 300)
    net = CFENet()
    # res = net.forward(t)
    print(net.base)
    