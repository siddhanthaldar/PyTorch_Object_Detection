import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()

        self.f = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1)
        self.g = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1)
        self.h = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1)

    def forward(self, x):
    	f = self.f(x)
    	f = torch.transpose(f,-2,-1)
    	g = self.g(x)
    	h = self.h(x)

    	attention_map = torch.mul(f,g)
    	out = torch.mul(h, attention_map)
    	return out


if __name__ == "__main__":
	x = torch.rand(1,3,300,300)
	att = AttentionBlock(3)
	x = att(x)
	print(x.size())
