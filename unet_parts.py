import numpy as np 
import pandas as pandas
import torch
import torchvision 
from torchvision import models, datasets, transforms
from torch.utils.data import dataloader
import torch.nn.functional as F 
from torch import nn, optim 

## Double Convolution block
class double_conv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.double_conv = nn.Sequential(
			# 1st Convolution
			nn.Conv2d(in_channels, out_channels, kernel_size = 3 , padding = 1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace = True),
			# 2nd Convolution
			nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace = True)
			)

	def forward(self, x):
		return self.double_conv(x)

## Downwards blocks
class down(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.down = nn.Sequential(
			nn.MaxPool2d(kernel_size =2), 
			double_conv(in_channels, out_channels)
			)

	def forward(self, x):
		return self.down(x)


## Upwards Blocks
class up(nn.Module):
	def __init__(self, in_channels, out_channels, bilinear = True):
		super().__init__()
		if bilinear:
			self.up = nn.Upsample(scale_factor =2, mode = 'bilinear', align_corners = True	)
		else:
			self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 2, stride = 2)
		self.conv = double_conv(in_channels, out_channels)

	def forward(self, x1, x2):
		x1 = self.up(x1)

		x = torch.cat([x2,x1], dim = 1)

		return self.conv(x)



## One by One convolution
class one_conv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(one_conv, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1)

	def forward(self, x):
		return self.conv(x)



