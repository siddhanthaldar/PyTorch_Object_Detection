import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

class CFE(nn.Module):
	def __init__(self, channels, k):
		super(CFE, self).__init__()

		# First branch
		self.conv1_1 = nn.Conv2d(channels, channels//2, kernel_size=1, padding=0)
		self.conv1_2 = nn.Conv2d(channels//2,channels//2, kernel_size=(k,1), padding=(k//2,0),groups=8)
		self.conv1_3 = nn.Conv2d(channels//2, channels//2, kernel_size=(1,k), padding=(0,k//2),groups=8)
		self.conv1_4 = nn.Conv2d(channels//2, channels//2, kernel_size=1, padding=0)

		# Second branch
		self.conv2_1 = nn.Conv2d(channels, channels//2, kernel_size=1, padding=0)
		self.conv2_2 = nn.Conv2d(channels//2,channels//2, kernel_size=(1,k), padding=(0,k//2),groups=8)
		self.conv2_3 = nn.Conv2d(channels//2, channels//2, kernel_size=(k,1), padding=(k//2,0),groups=8)
		self.conv2_4 = nn.Conv2d(channels//2, channels//2, kernel_size=1, padding=0)

	def forward(self, x):

		# First branch
		f = self.conv1_1(x)
		# print(f.size())
		f = self.conv1_2(f)
		# print(f.size())
		f = self.conv1_3(f)
		# print(f.size())
		f = self.conv1_4(f)
		# print(f.size())
		
		# Second branch
		s = self.conv2_1(x)
		# print(s.size())
		s = self.conv2_2(s)
		# print(s.size())
		s = self.conv2_3(s)
		# print(s.size())
		s = self.conv2_4(s)
		# print(s.size())
		
		fs = torch.cat((f,s), 1)
		# print(fs.size())
		
		return (fs+x)

class FFB(nn.Module):
	def __init__(self, c1, c2):
		super(FFB, self).__init__()

		self.conv1 = nn.Conv2d(c1, c1, kernel_size=1, padding=0)
		self.conv2 = nn.Conv2d(c2, c1, kernel_size=1, padding=0)
		self.deconv1 = nn.ConvTranspose2d(c1,c1, kernel_size=3, stride=2, padding=(1,1))

	def forward(self, x1, x2):

		f = self.conv1(x1)
		# print(f.size())
		s = self.conv2(x2)
		# print(s.size())
		# s = F.upsample(s, scale_factor = 2)
		s = self.deconv1(s)
		# print(s.size())

		# return(f+s) 
		return f

if __name__ == "__main__":
	x1 = torch.rand(1,512,38,38)        
	x2 = torch.rand(1,1024,19,19)        

	# model = CFE(512,3)
	model = FFB(512,1024)
	# x = model(x1)
	x = model(x1,x2)	
	print(x.size())


	x1 = torch.rand(1,512,19,19)        
	x2 = torch.rand(1,1024,10,10)        

	model = FFB(512,1024)
	x = model(x1,x2)	
	print(x.size())
