import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

from norm import L2Norm

class FusionBlock(nn.Module):
    def __init__(self, big_features, small_features):
        super(FusionBlock, self).__init__()
        
        # Bigger feature map
        self.conv1_1 = nn.Conv2d(big_features, 256, kernel_size=3, padding=1, dilation=1)
        self.Norm1 = L2Norm(256, 20)

        # Smaller feature map
        self.deconv2_1 = nn.ConvTranspose2d(small_features, 256, 2, stride=2, dilation=1)
        self.conv2_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1)
        self.bn2_1 = nn.BatchNorm2d(256)
        self.deconv2_2 = nn.ConvTranspose2d(256, 256, 2, stride=2, dilation=1)
        self.conv2_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1)
        self.bn2_2 = nn.BatchNorm2d(256)
        self.deconv2_3 = nn.ConvTranspose2d(256, 256, 3, stride=2, dilation=1)
        self.conv2_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1)
        self.Norm2 = L2Norm(256, 20)

        # Common
        self.conv3_1 = nn.Conv2d(256, big_features, kernel_size=3, padding=1, dilation=1)

        
    def forward(self, big, small):
        h1 = self.conv1_1(big)
        h1 = self.Norm1(h1)

        h2 = self.deconv2_1(small)
        # print(h2.size())
        h2 = F.relu(self.bn2_1(self.conv2_1(h2)))
        # print(h2.size())
        h2 = self.deconv2_2(h2)
        # print(h2.size())
        h2 = F.relu(self.bn2_2(self.conv2_2(h2)))
        # print(h2.size())
        h2 = self.deconv2_3(h2)
        # print(h2.size())
        h2 = self.conv2_3(h2)
        # print(h2.size())
        h2 = self.Norm2(h2)

        size = h2.size()[3]
        diff_odd = h2.size()[-1] - h1.size()[-1]
        h2 = h2[:,:,(int(diff_odd/2)+diff_odd%2):(size-int(diff_odd/2)),(int(diff_odd/2)+diff_odd%2):(size-int(diff_odd/2))]

        # print(h1.size(), h2.size())
        h = F.relu(h1+h2)
        h = F.relu(self.conv3_1(h))

        return h 

if __name__ == '__main__':
    big = torch.randn(1, 256, 128, 128)
    small = torch.rand(1,512,16,16)
    net = FusionBlock(256,512)

               
        
